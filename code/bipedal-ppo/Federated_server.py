import socket
import threading
import pickle
import struct
import torch
import gymnasium as gym
import argparse
import time
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


class FederatedLearningServer:
    def __init__(self, host='localhost', port=5000, num_clients=2, aggregation_method='fedavg'):
        """
        Initialize the FL server with centralized logging
        """
        self.host = host
        self.port = port
        self.num_clients = num_clients  # Fixed number of expected clients
        self.aggregation_method = aggregation_method

        # Client data storage
        self.client_weights = {}
        self.client_samples = {}
        self.clients_connected = 0

        # Global model
        self.global_model = None
        self.global_weights = None

        # Thread locks
        self.lock = threading.Lock()
        self.condition = threading.Condition()

        # Server socket
        self.server_socket = None

        # FL round tracking
        self.current_round = 0
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Centralized logging
        self.experiment_dir = f"./experiments/exp_{self.experiment_id}"
        self.client_logs_dir = os.path.join(self.experiment_dir, "client_logs")
        self.aggregated_weights_dir = os.path.join(self.experiment_dir, "aggregated_weights")
        self.analysis_dir = os.path.join(self.experiment_dir, "analysis")

        # Create all directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.client_logs_dir, exist_ok=True)
        os.makedirs(self.aggregated_weights_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        # Statistics tracking
        self.round_statistics = []
        self.client_statistics = defaultdict(list)

        print(f"Experiment ID: {self.experiment_id}")
        print(f"Experiment directory: {self.experiment_dir}")

    def create_socket(self):
        """Create and configure server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

    def initialize_global_model(self):
        """Initialize a global model template"""
        try:
            # Create a dummy environment
            env = gym.make("BipedalWalker-v3")

            # Import here to avoid circular imports
            from stable_baselines3 import PPO

            # Create a dummy PPO model to get structure
            dummy_model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99
            )
            env.close()
            self.global_model = dummy_model
            self.global_weights = self.extract_weights(dummy_model)
            print("Global model initialized successfully")

            # Save initial model info
            self.save_model_info()

        except Exception as e:
            print(f"Error initializing global model: {e}")
            import traceback
            traceback.print_exc()
            self.global_weights = {}

    def save_model_info(self):
        """Save model architecture information"""
        model_info = {
            'experiment_id': self.experiment_id,
            'model_type': 'PPO',
            'policy': 'MlpPolicy',
            'environment': 'BipedalWalker-v3',
            'num_clients': self.num_clients,
            'aggregation_method': self.aggregation_method,
            'creation_time': datetime.now().isoformat(),
            'model_parameters': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99
            }
        }

        info_file = os.path.join(self.experiment_dir, "model_info.json")
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Model info saved to {info_file}")

    def extract_weights(self, model):
        """Extract model weights as a dictionary"""
        weights = {}
        try:
            for name, param in model.policy.state_dict().items():
                if 'weight' in name or 'bias' in name:
                    weights[name] = param.clone()
            print(f"Extracted {len(weights)} weight tensors from model")
        except Exception as e:
            print(f"Error extracting weights: {e}")
        return weights

    def receive_all(self, sock, n, timeout=30):
        """Helper function to receive exactly n bytes with timeout"""
        data = bytearray()
        sock.settimeout(timeout)
        start_time = time.time()

        while len(data) < n:
            try:
                time_remaining = timeout - (time.time() - start_time)
                if time_remaining <= 0:
                    raise socket.timeout()

                sock.settimeout(time_remaining)
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout:
                return None
            except Exception as e:
                return None

        sock.settimeout(None)
        return data

    def send_message(self, sock, message):
        """Send a message with length prefix"""
        try:
            # Pickle the message
            data = pickle.dumps(message)

            # Send length prefix (8 bytes for long long)
            length = struct.pack('!Q', len(data))
            sock.sendall(length)

            # Send the actual data
            sock.sendall(data)
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def receive_message(self, sock, timeout=300):
        """Receive a message with length prefix"""
        try:
            # Receive the length prefix (8 bytes)
            length_data = self.receive_all(sock, 8, timeout)
            if not length_data:
                return None

            # Unpack length
            length = struct.unpack('!Q', length_data)[0]

            # Limit message size
            if length > 100 * 1024 * 1024:
                print(f"Message too large: {length} bytes")
                return None

            # Receive the actual data
            data = self.receive_all(sock, length, timeout)
            if not data:
                return None

            # Unpickle
            return pickle.loads(data)
        except socket.timeout:
            print(f"Timeout receiving message")
            return None
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None

    def save_client_training_data(self, client_id, round_num, training_data):
        """Save training data received from client"""
        client_log_dir = os.path.join(self.client_logs_dir, f"client_{client_id}")
        os.makedirs(client_log_dir, exist_ok=True)

        # Save training data
        data_file = os.path.join(client_log_dir, f"round_{round_num}_training.json")
        with open(data_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"Saved training data for client {client_id}, round {round_num}")

        # Update client statistics
        self.update_client_statistics(client_id, round_num, training_data)

    def update_client_statistics(self, client_id, round_num, training_data):
        """Update client statistics with new training data"""
        stats = {
            'client_id': client_id,
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'samples': training_data.get('sample_count', 0),
            'episodes': training_data.get('episodes', 0),
            'avg_reward': training_data.get('avg_reward', 0),
            'avg_distance': training_data.get('avg_distance', 0),
            'max_distance': training_data.get('max_distance', 0)
        }

        self.client_statistics[client_id].append(stats)

        # Save to CSV for easy analysis
        stats_file = os.path.join(self.analysis_dir, f"client_{client_id}_stats.csv")
        df = pd.DataFrame(self.client_statistics[client_id])
        df.to_csv(stats_file, index=False)

    def aggregate_weights_fedavg(self):
        """Aggregate weights using Federated Average"""
        if not self.client_weights:
            print("No client weights to aggregate")
            return self.global_weights

        # Calculate total samples
        total_samples = sum(self.client_samples.values())
        print(f"Aggregating weights from {len(self.client_weights)} clients (total samples: {total_samples})")

        # Initialize aggregated weights
        aggregated_weights = {}
        first_client = list(self.client_weights.keys())[0]

        for key in self.client_weights[first_client].keys():
            aggregated_weights[key] = torch.zeros_like(
                self.client_weights[first_client][key]
            )

        # Weighted average
        for client_id in self.client_weights:
            client_weight = self.client_weights[client_id]
            weight_factor = self.client_samples[client_id] / total_samples

            for key in aggregated_weights:
                aggregated_weights[key] += client_weight[key] * weight_factor

        return aggregated_weights

    def aggregate_weights_fedsgd(self):
        """Aggregate weights using Federated SGD (simple average)"""
        if not self.client_weights:
            print("No client weights to aggregate")
            return self.global_weights

        print(f"Aggregating weights from {len(self.client_weights)} clients using FedSGD")

        # Initialize aggregated weights
        aggregated_weights = {}
        first_client = list(self.client_weights.keys())[0]

        for key in self.client_weights[first_client].keys():
            aggregated_weights[key] = torch.zeros_like(
                self.client_weights[first_client][key]
            )

        # Simple average
        num_clients = len(self.client_weights)
        for client_id in self.client_weights:
            client_weight = self.client_weights[client_id]

            for key in aggregated_weights:
                aggregated_weights[key] += client_weight[key] / num_clients

        return aggregated_weights

    def save_aggregated_weights(self, round_num):
        """Save aggregated weights to file for analysis"""
        weights_file = os.path.join(self.aggregated_weights_dir, f'round_{round_num}.pkl')

        try:
            # Convert tensors to CPU for saving
            weights_to_save = {}
            for key, tensor in self.global_weights.items():
                weights_to_save[key] = tensor.cpu().clone()

            with open(weights_file, 'wb') as f:
                pickle.dump(weights_to_save, f)

            print(f"Saved aggregated weights for round {round_num} to {weights_file}")

            # Calculate and save weight statistics
            self.save_weight_statistics(round_num)

        except Exception as e:
            print(f"Error saving aggregated weights: {e}")

    def save_weight_statistics(self, round_num):
        """Calculate and save weight statistics"""
        try:
            # First, load the weights we just saved
            weights_file = os.path.join(self.aggregated_weights_dir, f'round_{round_num}.pkl')

            if not os.path.exists(weights_file):
                print(f"Weights file not found: {weights_file}")
                return

            with open(weights_file, 'rb') as f:
                weights_to_save = pickle.load(f)

            weight_stats = {
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'layers': {}
            }

            total_norm = 0.0
            for layer_name, tensor in weights_to_save.items():
                if 'weight' in layer_name:
                    # Convert to numpy and ensure float64 for JSON serialization
                    if isinstance(tensor, torch.Tensor):
                        flat_weights = tensor.numpy().flatten().astype(np.float64)
                    else:
                        flat_weights = np.array(tensor).flatten().astype(np.float64)

                    layer_norm = float(np.linalg.norm(flat_weights))

                    # Convert all numpy values to Python native types
                    weight_stats['layers'][layer_name] = {
                        'norm': layer_norm,
                        'mean': float(flat_weights.mean()),
                        'std': float(flat_weights.std()),
                        'min': float(flat_weights.min()),
                        'max': float(flat_weights.max())
                    }
                    total_norm += layer_norm

            weight_stats['total_norm'] = total_norm

            # Save statistics - use custom serializer for numpy types
            stats_file = os.path.join(self.analysis_dir, f"weight_stats_round_{round_num}.json")

            # Custom JSON encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)

            with open(stats_file, 'w') as f:
                json.dump(weight_stats, f, indent=2, cls=NumpyEncoder)

            print(f"Saved weight statistics for round {round_num}")

            # Update round statistics
            self.round_statistics.append({
                'round': round_num,
                'clients_participated': len(self.client_weights),
                'total_samples': sum(self.client_samples.values()),
                'total_norm': total_norm,
                'timestamp': datetime.now().isoformat()
            })

            # Save round statistics to CSV
            round_stats_file = os.path.join(self.analysis_dir, "round_statistics.csv")
            if self.round_statistics:
                df = pd.DataFrame(self.round_statistics)
                df.to_csv(round_stats_file, index=False)

        except Exception as e:
            print(f"Error saving weight statistics: {e}")
            import traceback
            traceback.print_exc()

    def handle_client(self, client_socket, client_address):
        """Handle communication with a single client"""
        client_id = None
        try:
            print(f"New connection from {client_address}")

            # Set socket timeout
            client_socket.settimeout(300.0)

            # Receive client info
            client_info = self.receive_message(client_socket, timeout=30)
            if not client_info:
                print(f"No client info received from {client_address}")
                return

            client_id = client_info['client_id']
            sample_count = client_info.get('sample_count', 100000)

            with self.lock:
                self.clients_connected += 1
                self.client_samples[client_id] = sample_count
                print(f"Client {client_id} connected. Total connected: {self.clients_connected}/{self.num_clients}")

            # Send initial global model weights
            init_data = {
                'weights': self.global_weights,
                'message': 'INITIAL_WEIGHTS',
                'current_round': self.current_round,
                'experiment_id': self.experiment_id,
                'num_clients': self.num_clients
            }
            if not self.send_message(client_socket, init_data):
                print(f"Failed to send initial weights to client {client_id}")
                return
            print(f"Sent initial weights to client {client_id} (Round {self.current_round})")

            # Main communication loop
            while True:
                client_data = self.receive_message(client_socket, timeout=600)

                if not client_data:
                    print(f"Client {client_id} disconnected (timeout)")
                    break

                if client_data.get('message') == 'WEIGHTS_UPDATE':
                    weights = client_data['weights']
                    client_round = client_data.get('round', 0)

                    # Save client training data if provided
                    if 'training_data' in client_data:
                        self.save_client_training_data(client_id, client_round, client_data['training_data'])

                    with self.lock:
                        self.client_weights[client_id] = weights
                        print(f"Received weights from client {client_id} (Round {client_round})")
                        print(f"Waiting for {self.num_clients - len(self.client_weights)} more clients...")

                    # Check if ALL clients have sent weights
                    if len(self.client_weights) == self.num_clients:
                        with self.condition:
                            self.condition.notify_all()

                    # Wait for aggregation to complete
                    with self.condition:
                        print(f"Client {client_id} waiting for aggregation...")
                        self.condition.wait()  # Wait indefinitely until notified

                    # Send aggregated weights back to client
                    response_data = {
                        'weights': self.global_weights,
                        'round_complete': True,
                        'message': 'AGGREGATED_WEIGHTS',
                        'global_round': self.current_round
                    }
                    if not self.send_message(client_socket, response_data):
                        print(f"Failed to send aggregated weights to client {client_id}")
                        break
                    print(f"Sent aggregated weights to client {client_id} (Round {self.current_round})")

                elif client_data.get('message') == 'TRAINING_DATA':
                    # Client is sending training statistics
                    training_data = client_data.get('data', {})
                    client_round = client_data.get('round', 0)
                    self.save_client_training_data(client_id, client_round, training_data)

                    # Send acknowledgement
                    self.send_message(client_socket, {'message': 'DATA_RECEIVED'})

                elif client_data.get('message') == 'DISCONNECT':
                    print(f"Client {client_id} requested disconnect")
                    break

        except socket.timeout:
            print(f"Client {client_id}: Connection timeout")
        except (ConnectionError, EOFError, BrokenPipeError) as e:
            print(f"Connection error with client {client_id}: {e}")
        except Exception as e:
            print(f"Unexpected error with client {client_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.lock:
                if client_id and client_id in self.client_weights:
                    del self.client_weights[client_id]
                if client_id and client_id in self.client_samples:
                    del self.client_samples[client_id]
                self.clients_connected -= 1
                print(f"Client {client_id} disconnected. Remaining: {self.clients_connected}")
            try:
                client_socket.close()
            except:
                pass

    def aggregation_thread(self):
        """Thread for aggregating weights from all clients - waits for ALL clients"""
        print("Aggregation thread started")

        while True:
            try:
                # Wait for ALL clients to send weights
                with self.condition:
                    while len(self.client_weights) < self.num_clients:
                        print(f"Waiting for weights: {len(self.client_weights)}/{self.num_clients} clients")
                        self.condition.wait()  # Wait until all clients send weights

                self.current_round += 1
                print(f"\n{'=' * 60}")
                print(
                    f"Aggregating weights from {len(self.client_weights)}/{self.num_clients} clients (Round {self.current_round})...")
                print('=' * 60)

                # Aggregate weights
                if self.aggregation_method == 'fedavg':
                    aggregated_weights = self.aggregate_weights_fedavg()
                else:  # fedsgd
                    aggregated_weights = self.aggregate_weights_fedsgd()

                # Update global weights
                self.global_weights = aggregated_weights

                # Save aggregated weights and statistics
                self.save_aggregated_weights(self.current_round)

                print("Aggregation complete. Broadcasting updated weights...")
                print(f"{'=' * 60}\n")

                # Perform analysis after each round
                self.perform_round_analysis(self.current_round)

                # Notify all waiting clients
                with self.condition:
                    self.condition.notify_all()
                    time.sleep(0.1)  # Small delay to ensure all clients get notified
                    self.client_weights.clear()  # Clear for next round

            except Exception as e:
                print(f"Error in aggregation thread: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def perform_round_analysis(self, round_num):
        """Perform analysis after each round"""
        try:
            print(f"Performing analysis for round {round_num}...")

            # Generate analysis report
            self.generate_analysis_report(round_num)

            # Plot training progress
            self.plot_training_progress()

            # Plot weight convergence
            self.plot_weight_convergence()

            print(f"Analysis complete for round {round_num}")

        except Exception as e:
            print(f"Error performing analysis: {e}")

    def generate_analysis_report(self, round_num):
        """Generate comprehensive analysis report"""
        report_file = os.path.join(self.analysis_dir, f"analysis_report_round_{round_num}.txt")

        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"FEDERATED LEARNING ANALYSIS REPORT\n")
            f.write(f"Round: {round_num}\n")
            f.write(f"Experiment: {self.experiment_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            # Client participation
            f.write("CLIENT PARTICIPATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Expected clients: {self.num_clients}\n")
            f.write(f"Clients participated this round: {len(self.client_weights)}\n")
            f.write(f"Total samples: {sum(self.client_samples.values())}\n\n")

            # Client statistics
            if self.client_statistics:
                f.write("CLIENT PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                for client_id, stats_list in self.client_statistics.items():
                    if stats_list:
                        latest = stats_list[-1]
                        f.write(f"Client {client_id} (Round {latest.get('round', 'N/A')}):\n")
                        f.write(f"  Episodes: {latest.get('episodes', 0)}\n")
                        f.write(f"  Avg Distance: {latest.get('avg_distance', 0):.2f}\n")
                        f.write(f"  Max Distance: {latest.get('max_distance', 0):.2f}\n")
                        f.write(f"  Avg Reward: {latest.get('avg_reward', 0):.2f}\n\n")

            # Weight statistics
            f.write("MODEL STATISTICS:\n")
            f.write("-" * 40 + "\n")
            if self.round_statistics:
                latest_round = self.round_statistics[-1]
                f.write(f"Total weight norm: {latest_round.get('total_norm', 0):.2f}\n")
                f.write(f"Clients participated: {latest_round.get('clients_participated', 0)}\n")

        print(f"Analysis report saved to {report_file}")

    def plot_training_progress(self):
        """Plot training progress across clients using per-round aggregated data."""
        try:
            import matplotlib.pyplot as plt

            if not self.client_statistics:
                print("No client statistics available for plotting.")
                return

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # =============================
            # Plot 1 — Avg Distance per Client
            # =============================
            for client_id, stats_list in self.client_statistics.items():
                rounds = [s.get('round', 0) for s in stats_list]
                avg_dist = [s.get('avg_distance', 0) for s in stats_list]
                axes[0, 0].plot(rounds, avg_dist, marker='o', label=f'Client {client_id}')

            axes[0, 0].set_title("Average Distance per Client (per Round)")
            axes[0, 0].set_xlabel("Round")
            axes[0, 0].set_ylabel("Avg Distance")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

            # =============================
            # Plot 2 — Max Distance per Client
            # =============================
            for client_id, stats_list in self.client_statistics.items():
                rounds = [s.get('round', 0) for s in stats_list]
                max_dist = [s.get('max_distance', 0) for s in stats_list]
                axes[0, 1].plot(rounds, max_dist, marker='s', label=f'Client {client_id}')

            axes[0, 1].set_title("Max Distance per Client (per Round)")
            axes[0, 1].set_xlabel("Round")
            axes[0, 1].set_ylabel("Max Distance")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

            # =============================
            # Plot 3 — Reward per Round (Performance)
            # =============================
            for client_id, stats_list in self.client_statistics.items():
                rounds = [s.get('round', 0) for s in stats_list]
                rewards = [s.get('avg_reward', 0) for s in stats_list]
                axes[1, 0].plot(rounds, rewards, marker='D', label=f'Client {client_id}')

            axes[1, 0].set_title("Average Reward per Client (per Round)")
            axes[1, 0].set_xlabel("Round")
            axes[1, 0].set_ylabel("Avg Reward")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

            plt.tight_layout()
            out_file = os.path.join(self.analysis_dir, f"training_progress_round_{self.current_round}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()

            print(f"[PLOT] Training progress saved: {out_file}")

        except Exception as e:
            print(f"Error in plot_training_progress: {e}")

    def plot_weight_convergence(self):
        """Plot weight norm and norm deltas for each federated round."""
        try:
            import matplotlib.pyplot as plt

            weight_norms = []
            rounds = []

            for round_num in range(1, self.current_round + 1):
                stats_file = os.path.join(self.analysis_dir, f"weight_stats_round_{round_num}.json")
                if not os.path.exists(stats_file):
                    continue

                try:
                    with open(stats_file, 'r') as f:
                        data = json.load(f)
                    if "total_norm" in data:
                        weight_norms.append(float(data["total_norm"]))
                        rounds.append(round_num)
                except Exception as e:
                    print(f"Failed to read {stats_file}: {e}")

            if len(weight_norms) < 2:
                print("[WARN] Not enough rounds to generate weight convergence plot.")
                return

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # =============================
            # Plot 1 — Weight Norm
            # =============================
            axes[0].plot(rounds, weight_norms, marker='o', linewidth=2)
            axes[0].set_title("Model Weight Norm Over Rounds")
            axes[0].set_xlabel("Round")
            axes[0].set_ylabel("Weight L2 Norm")
            axes[0].grid(True, alpha=0.3)

            # =============================
            # Plot 2 — Relative Norm Change
            # =============================
            norm_changes = [
                abs(weight_norms[i] - weight_norms[i - 1]) / max(weight_norms[i - 1], 1e-8) * 100
                for i in range(1, len(weight_norms))
            ]

            axes[1].bar(rounds[1:], norm_changes)
            axes[1].set_title("Relative Weight Norm Change (%) per Round")
            axes[1].set_xlabel("Round")
            axes[1].set_ylabel("Change (%)")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            out_file = os.path.join(self.analysis_dir, f"weight_convergence_round_{self.current_round}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()

            print(f"[PLOT] Weight convergence saved: {out_file}")

        except Exception as e:
            print(f"Error in plot_weight_convergence: {e}")

    def start(self):
        """Start the FL server"""
        self.create_socket()
        self.initialize_global_model()

        # Start aggregation thread
        aggregation_thread = threading.Thread(target=self.aggregation_thread, daemon=True)
        aggregation_thread.start()

        print(f"\n{'*' * 60}")
        print(f"FEDERATED LEARNING SERVER STARTED")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Host: {self.host}:{self.port}")
        print(f"Expected clients: {self.num_clients}")
        print(f"Aggregation Method: {self.aggregation_method}")
        print(f"Data directory: {self.experiment_dir}")
        print(f"{'*' * 60}\n")

        # Accept client connections
        try:
            while True:
                client_socket, client_address = self.server_socket.accept()
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()

        except KeyboardInterrupt:
            print("\nShutting down server...")
            self.finalize_experiment()
        except Exception as e:
            print(f"Server error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.server_socket.close()
            print("Server closed")

    def finalize_experiment(self):
        """Finalize experiment and generate final report with detailed round-by-round analysis"""
        print("\n" + "=" * 60)
        print("FINALIZING EXPERIMENT")
        print("=" * 60)

        try:
            # Generate final analysis report
            final_report_file = os.path.join(self.experiment_dir, "final_analysis_report.txt")

            with open(final_report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("FINAL FEDERATED LEARNING ANALYSIS REPORT\n")
                f.write(f"Experiment ID: {self.experiment_id}\n")
                f.write(f"Total Rounds: {self.current_round}\n")
                f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                # Summary statistics
                f.write("EXPERIMENT SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total rounds completed: {self.current_round}\n")
                f.write(f"Expected clients: {self.num_clients}\n")
                f.write(f"Unique clients participated: {len(self.client_statistics)}\n")
                f.write(f"Aggregation method: {self.aggregation_method.upper()}\n")

                if self.round_statistics:
                    total_samples = sum(r.get('total_samples', 0) for r in self.round_statistics)
                    f.write(f"Total training samples: {total_samples}\n")

                # ROUND-BY-ROUND PERFORMANCE
                f.write("\n" + "=" * 80 + "\n")
                f.write("ROUND-BY-ROUND PERFORMANCE TRACKING\n")
                f.write("=" * 80 + "\n\n")

                # Organize data by round
                for round_num in range(1, self.current_round + 1):
                    f.write(f"\nROUND {round_num}:\n")
                    f.write("-" * 80 + "\n")

                    # Get round statistics
                    round_stat = next((r for r in self.round_statistics if r['round'] == round_num), None)
                    if round_stat:
                        f.write(f"Total samples this round: {round_stat.get('total_samples', 0)}\n")
                        f.write(f"Model weight norm: {round_stat.get('total_norm', 0):.2f}\n")
                        f.write(f"Clients participated: {round_stat.get('clients_participated', 0)}\n\n")

                    # Client performance this round
                    for client_id in sorted(self.client_statistics.keys()):
                        client_rounds = self.client_statistics[client_id]
                        round_data = next((r for r in client_rounds if r['round'] == round_num), None)

                        if round_data:
                            f.write(f"  Client {client_id}:\n")
                            f.write(f"    Episodes: {round_data.get('episodes', 0)}\n")
                            f.write(f"    Avg Reward: {round_data.get('avg_reward', 0):.2f}\n")
                            f.write(f"    Avg Distance: {round_data.get('avg_distance', 0):.2f}\n")
                            f.write(f"    Max Distance: {round_data.get('max_distance', 0):.2f}\n")
                            f.write(f"    Training Samples: {round_data.get('samples', 0)}\n")
                            f.write("\n")

                # CLIENT PERFORMANCE SUMMARY (Overall)
                f.write("\n" + "=" * 80 + "\n")
                f.write("CLIENT PERFORMANCE SUMMARY (ALL ROUNDS)\n")
                f.write("=" * 80 + "\n\n")

                for client_id in sorted(self.client_statistics.keys()):
                    stats_list = self.client_statistics[client_id]
                    if stats_list:
                        total_episodes = sum(s.get('episodes', 0) for s in stats_list)
                        avg_rewards = [s.get('avg_reward', 0) for s in stats_list]
                        avg_distances = [s.get('avg_distance', 0) for s in stats_list]
                        max_distances = [s.get('max_distance', 0) for s in stats_list]

                        f.write(f"Client {client_id}:\n")
                        f.write(f"  Rounds participated: {len(stats_list)}\n")
                        f.write(f"  Total episodes: {total_episodes}\n")
                        f.write(f"  Average reward across rounds: {np.mean(avg_rewards):.2f}\n")
                        f.write(f"  Average distance across rounds: {np.mean(avg_distances):.2f}\n")
                        f.write(f"  Best distance achieved: {max(max_distances):.2f}\n")
                        f.write(f"  Improvement (first → last avg distance): "
                                f"{avg_distances[0]:.2f} → {avg_distances[-1]:.2f} "
                                f"({((avg_distances[-1] - avg_distances[0]) / max(abs(avg_distances[0]), 0.01) * 100):.1f}%)\n")
                        f.write("\n")

                # FEDERATED LEARNING INSIGHTS
                f.write("=" * 80 + "\n")
                f.write("FEDERATED LEARNING INSIGHTS\n")
                f.write("=" * 80 + "\n\n")

                # Check if clients are improving together
                all_avg_distances = []
                for client_id in sorted(self.client_statistics.keys()):
                    stats_list = self.client_statistics[client_id]
                    avg_distances = [s.get('avg_distance', 0) for s in stats_list]
                    all_avg_distances.append(avg_distances)

                if len(all_avg_distances) >= 2 and len(all_avg_distances[0]) > 1:
                    # Calculate correlation between client improvements
                    client1_improvement = all_avg_distances[0]
                    client2_improvement = all_avg_distances[1]

                    f.write("Collaborative Learning Analysis:\n")
                    f.write("-" * 80 + "\n")

                    # Check if both improved
                    client1_improved = client1_improvement[-1] > client1_improvement[0]
                    client2_improved = client2_improvement[-1] > client2_improvement[0]

                    if client1_improved and client2_improved:
                        f.write("✓ Both clients showed improvement through federated learning!\n")
                    else:
                        f.write("⚠ Not all clients improved - may need more rounds or tuning\n")

                    # Average improvement
                    avg_initial = np.mean([d[0] for d in all_avg_distances])
                    avg_final = np.mean([d[-1] for d in all_avg_distances])
                    overall_improvement = ((avg_final - avg_initial) / max(abs(avg_initial), 0.01) * 100)

                    f.write(f"\nAverage performance across all clients:\n")
                    f.write(f"  Initial (Round 1): {avg_initial:.2f}\n")
                    f.write(f"  Final (Round {self.current_round}): {avg_final:.2f}\n")
                    f.write(f"  Overall improvement: {overall_improvement:.1f}%\n")

                # Model convergence
                f.write("\n" + "=" * 80 + "\n")
                f.write("MODEL CONVERGENCE ANALYSIS\n")
                f.write("=" * 80 + "\n\n")

                if len(self.round_statistics) > 1:
                    first_norm = self.round_statistics[0].get('total_norm', 0)
                    last_norm = self.round_statistics[-1].get('total_norm', 0)
                    total_change = ((last_norm - first_norm) / max(first_norm, 0.01) * 100)

                    f.write(f"Initial model weight norm (Round 1): {first_norm:.2f}\n")
                    f.write(f"Final model weight norm (Round {self.current_round}): {last_norm:.2f}\n")
                    f.write(f"Total weight norm change: {total_change:.1f}%\n")
                    f.write(f"Model converged: {'Yes' if abs(total_change) < 10 else 'No (still learning)'}\n")

                    # Weight norm progression
                    f.write(f"\nWeight norm by round:\n")
                    for i, stat in enumerate(self.round_statistics, 1):
                        f.write(f"  Round {i}: {stat.get('total_norm', 0):.2f}\n")

                # Recommendations
                f.write("\n" + "=" * 80 + "\n")
                f.write("RECOMMENDATIONS\n")
                f.write("=" * 80 + "\n\n")

                if self.current_round < 10:
                    f.write("• Consider running more rounds (10-20) for better convergence\n")

                if len(self.round_statistics) > 1:
                    recent_norms = [r.get('total_norm', 0) for r in self.round_statistics[-3:]]
                    if len(recent_norms) >= 2:
                        norm_variance = np.std(recent_norms)
                        if norm_variance > first_norm * 0.05:
                            f.write("• Model weights are still changing significantly - more training needed\n")
                        else:
                            f.write("• Model weights are stabilizing - good convergence!\n")

                # Check performance trend
                if all_avg_distances and len(all_avg_distances[0]) > 2:
                    recent_trend = all_avg_distances[0][-2:]
                    if recent_trend[-1] < recent_trend[-2]:
                        f.write("• Recent performance declined - consider adjusting learning rate\n")

                f.write("\n" + "=" * 80 + "\n")

            print(f"Final analysis report saved to {final_report_file}")

            # Print summary to console
            print("\n" + "=" * 80)
            print("EXPERIMENT COMPLETE - KEY METRICS:")
            print("=" * 80)
            print(f"Total Rounds: {self.current_round}")
            print(f"Total Clients: {len(self.client_statistics)}")

            for client_id in sorted(self.client_statistics.keys()):
                stats_list = self.client_statistics[client_id]
                if stats_list:
                    avg_distances = [s.get('avg_distance', 0) for s in stats_list]
                    print(f"Client {client_id}: {avg_distances[0]:.2f} → {avg_distances[-1]:.2f} (Avg Distance)")

            print("=" * 80)

            # Save experiment metadata
            metadata = {
                'experiment_id': self.experiment_id,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_rounds': self.current_round,
                'num_clients': self.num_clients,
                'aggregation_method': self.aggregation_method,
                'client_ids': list(self.client_statistics.keys()),
                'directory_structure': {
                    'experiment_dir': self.experiment_dir,
                    'client_logs': self.client_logs_dir,
                    'aggregated_weights': self.aggregated_weights_dir,
                    'analysis': self.analysis_dir
                }
            }

            metadata_file = os.path.join(self.experiment_dir, "experiment_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Experiment metadata saved to {metadata_file}")

        except Exception as e:
            print(f"Error finalizing experiment: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Server.')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--clients', type=int, default=3, help='Expected number of clients')
    parser.add_argument('--aggregation', type=str, default='fedavg',
                        choices=['fedavg', 'fedsgd'], help='Aggregation method')

    args = parser.parse_args()

    server = FederatedLearningServer(
        host=args.host,
        port=args.port,
        num_clients=args.clients,
        aggregation_method=args.aggregation
    )
    server.start()


if __name__ == "__main__":
    main()
