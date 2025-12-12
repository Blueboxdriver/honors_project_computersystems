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
from stable_baselines3.common.callbacks import BaseCallback


# ========================================================================
# TRAINING STATISTICS CALLBACK (unchanged)
# ========================================================================

class TrainingStatsCallback(BaseCallback):
    """Custom callback to track training statistics"""

    def __init__(self, verbose=0):
        super(TrainingStatsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.current_episode_distance = 0

    def _on_step(self) -> bool:
        # Track episode info from SB3 monitor
        for info in self.locals.get("infos", []):
            if "episode" in info:
                reward = info["episode"]["r"]
                length = info["episode"]["l"]

                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)

                # Crude distance estimate based on reward
                estimated_distance = max(0, reward + 100)
                self.episode_distances.append(estimated_distance)

        return True

    def get_statistics(self):
        if len(self.episode_rewards) == 0:
            return {
                "episodes": 0,
                "avg_reward": 0.0,
                "avg_distance": 0.0,
                "max_distance": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "std_reward": 0.0,
                "total_steps": 0
            }

        return {
            "episodes": len(self.episode_rewards),
            "avg_reward": float(np.mean(self.episode_rewards)),
            "avg_distance": float(np.mean(self.episode_distances)),
            "max_distance": float(np.max(self.episode_distances)),
            "min_reward": float(np.min(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "total_steps": sum(self.episode_lengths)
        }

    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []


# ========================================================================
# EPISODE LIMIT CALLBACK
# ========================================================================

class EpisodeLimitCallback(BaseCallback):
    """Stop training once a target number of episodes is reached."""

    def __init__(self, max_episodes, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episodes = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episodes += 1
                if self.verbose:
                    print(f"[EpisodeLimit] Episode {self.episodes}/{self.max_episodes}")

                if self.episodes >= self.max_episodes:
                    if self.verbose:
                        print(f"[EpisodeLimit] Reached limit, stopping round")
                    return False  # SB3 stops training
        return True

    def reset(self):
        self.episodes = 0


# ========================================================================
# FEDERATED CLIENT
# ========================================================================

class FederatedLearningClient:
    def __init__(self, client_id, server_host='localhost', server_port=5000):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.model = None
        self.env = None

        self.num_rounds = 12
        self.episodes_per_round = 20  # NEW: override using args
        self.stats_callback = None

    # -------------- NETWORK HELPERS -------------------------------------

    def connect_to_server(self):
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
        try:
            length_data = self.socket.recv(8)
            if not length_data:
                return None

            length = struct.unpack('!Q', length_data)[0]

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

    # -------------- ENVIRONMENT -----------------------------------------

    def make_env(self):
        def _init():
            env = gym.make("BipedalWalker-v3")
            return Monitor(env)
        return _init

    # -------------- MAIN FEDERATED LOOP ---------------------------------

    def participate_in_fl(self):
        if not self.connect_to_server():
            return

        # Send client info
        self.send_message({
            "client_id": self.client_id,
            "sample_count": self.episodes_per_round,
            "message": "CLIENT_INFO"
        })

        # Receive initial weights
        server_data = self.receive_message()
        if not server_data or server_data.get("message") != "INITIAL_WEIGHTS":
            print("No initial weights received.")
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

        # Load initial weights
        initial_weights = server_data["weights"]
        model_dict = self.model.policy.state_dict()
        model_dict.update(initial_weights)
        self.model.policy.load_state_dict(model_dict)

        self.stats_callback = TrainingStatsCallback()

        # ==========================
        # FL ROUNDS
        # ==========================

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n====================== ROUND {round_num}/{self.num_rounds} ======================")

            self.stats_callback.reset()
            episode_limit_cb = EpisodeLimitCallback(self.episodes_per_round, verbose=1)
            episode_limit_cb.reset()

            # Train for **episodes**, not timesteps**
            self.model.learn(
                total_timesteps=int(1e12),  # arbitrary huge number so PPO doesn't stop early
                reset_num_timesteps=False,
                callback=[self.stats_callback, episode_limit_cb],
                tb_log_name=f"client_{self.client_id}_round_{round_num}"
            )

            # Extract client stats
            training_stats = self.stats_callback.get_statistics()
            training_stats["round"] = round_num
            training_stats["client_id"] = self.client_id
            training_stats["sample_count"] = self.episodes_per_round

            print(f"\n[Client {self.client_id}] Round {round_num} Stats:")
            print(f"  Episodes: {training_stats['episodes']}")
            print(f"  Avg Reward: {training_stats['avg_reward']:.2f}")
            print(f"  Reward Range: [{training_stats['min_reward']:.2f}, {training_stats['max_reward']:.2f}]")
            print(f"  Avg Distance: {training_stats['avg_distance']:.2f}")
            print(f"  Max Distance: {training_stats['max_distance']:.2f}")

            # Extract weights to send to server
            weights = {
                name: param.clone().cpu()
                for name, param in self.model.policy.state_dict().items()
                if "weight" in name or "bias" in name
            }

            update = {
                "client_id": self.client_id,
                "round": round_num,
                "weights": weights,
                "training_data": training_stats,
                "message": "WEIGHTS_UPDATE"
            }

            print("Sending weights to server...")
            self.send_message(update)

            # Receive AGGREGATED WEIGHTS
            response = self.receive_message()
            if response and response.get("message") == "AGGREGATED_WEIGHTS":
                aggregated = response["weights"]
                model_dict = self.model.policy.state_dict()
                model_dict.update(aggregated)
                self.model.policy.load_state_dict(model_dict)
                print(f"Client {self.client_id}: Updated with aggregated weights.")
            else:
                print("Warning: Did not receive aggregated weights!")

            time.sleep(1)

        print("\nClient finished all rounds. Disconnecting...")
        self.send_message({"client_id": self.client_id, "message": "DISCONNECT"})
        self.socket.close()
        self.env.close()
        print("Client done.")


# ========================================================================
# MAIN ENTRY
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--id", type=int, required=True, help="Client ID")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--rounds", type=int, default=12, help="FL rounds")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per round")  # NEW

    args = parser.parse_args()

    client = FederatedLearningClient(
        client_id=args.id,
        server_host=args.host,
        server_port=args.port
    )

    client.num_rounds = args.rounds
    client.episodes_per_round = args.episodes

    client.participate_in_fl()


if __name__ == "__main__":
    main()
