import socket
import pickle
import struct
import torch
import gymnasium as gym
import argparse
import time
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class FederatedLearningClient:
    def __init__(self, client_id, server_host='localhost', server_port=5000):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.model = None
        self.env = None
        self.total_timesteps = 100000
        self.num_rounds = 12

    def connect_to_server(self):
        """Connect to the server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30)
            self.socket.connect((self.server_host, self.server_port))
            print(f"Client {self.client_id} connected to server")
            return True
        except Exception as e:
            print(f"Client {self.client_id}: Connection failed: {e}")
            return False

    def send_message(self, message):
        """Send a message to server"""
        try:
            data = pickle.dumps(message)
            length = struct.pack('!Q', len(data))
            self.socket.sendall(length)
            self.socket.sendall(data)
            return True
        except Exception as e:
            print(f"Client {self.client_id}: Error sending message: {e}")
            return False

    def receive_message(self):
        """Receive message from server"""
        try:
            # Receive length
            length_data = self.socket.recv(8)
            if not length_data:
                return None
            length = struct.unpack('!Q', length_data)[0]

            # Receive data
            data = b''
            while len(data) < length:
                packet = self.socket.recv(length - len(data))
                if not packet:
                    return None
                data += packet

            return pickle.loads(data)
        except Exception as e:
            print(f"Client {self.client_id}: Error receiving message: {e}")
            return None

    def make_env(self):
        """Create environment"""

        def _init():
            env = gym.make("BipedalWalker-v3")
            return env

        return _init

    def collect_training_statistics(self, round_num):
        """Collect training statistics to send to server"""
        # In practice, you'd track this during training
        # For now, simulate some statistics
        return {
            'round': round_num,
            'sample_count': self.total_timesteps,
            'episodes': np.random.randint(50, 100),  # Simulated
            'avg_reward': np.random.uniform(-50, 50),  # Simulated
            'avg_distance': np.random.uniform(0, 100),  # Simulated
            'max_distance': np.random.uniform(50, 200)  # Simulated
        }

    def participate_in_fl(self):
        """Participate in federated learning"""
        if not self.connect_to_server():
            return

        # Send client info
        client_info = {
            'client_id': self.client_id,
            'sample_count': self.total_timesteps,
            'message': 'CLIENT_INFO'
        }
        self.send_message(client_info)

        # Receive initial weights
        server_data = self.receive_message()
        if not server_data or server_data.get('message') != 'INITIAL_WEIGHTS':
            print(f"Client {self.client_id}: No initial weights received")
            return

        # Initialize model
        self.env = DummyVecEnv([self.make_env()])
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99
        )

        initial_weights = server_data['weights']
        if initial_weights:
            model_dict = self.model.policy.state_dict()
            model_dict.update(initial_weights)
            self.model.policy.load_state_dict(model_dict)

        # FL rounds
        for round_num in range(1, self.num_rounds + 1):
            print(f"\nClient {self.client_id}: Starting round {round_num}")

            # Train locally
            self.model.learn(
                total_timesteps=self.total_timesteps,
                reset_num_timesteps=False,
                tb_log_name=f"client_{self.client_id}_round_{round_num}"
            )

            # Extract weights
            weights = {}
            for name, param in self.model.policy.state_dict().items():
                if 'weight' in name or 'bias' in name:
                    weights[name] = param.clone().cpu()

            # Collect training statistics
            training_stats = self.collect_training_statistics(round_num)

            # Send weights and statistics to server
            update_data = {
                'weights': weights,
                'training_data': training_stats,
                'client_id': self.client_id,
                'round': round_num,
                'message': 'WEIGHTS_UPDATE'
            }
            self.send_message(update_data)

            # Receive aggregated weights
            response = self.receive_message()
            if response and response.get('message') == 'AGGREGATED_WEIGHTS':
                aggregated_weights = response['weights']
                model_dict = self.model.policy.state_dict()
                model_dict.update(aggregated_weights)
                self.model.policy.load_state_dict(model_dict)
                print(f"Client {self.client_id}: Updated with aggregated weights")

            time.sleep(1)

        # Disconnect
        self.send_message({'message': 'DISCONNECT', 'client_id': self.client_id})
        self.socket.close()
        self.env.close()
        print(f"Client {self.client_id}: FL complete")


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--id', type=int, required=True, help='Client ID')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--rounds', type=int, default=12, help='FL rounds')
    parser.add_argument('--timesteps', type=int, default=100000, help='Timesteps per round')

    args = parser.parse_args()

    client = FederatedLearningClient(
        client_id=args.id,
        server_host=args.host,
        server_port=args.port
    )
    client.total_timesteps = args.timesteps
    client.num_rounds = args.rounds

    client.participate_in_fl()


if __name__ == "__main__":
    main()