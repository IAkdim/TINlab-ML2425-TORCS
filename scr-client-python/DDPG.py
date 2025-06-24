#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import socket
import select
import time

from run_continuous_model import parse_sensors, create_control_string

STATE_SIZE = 35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPEED_ACTIONS = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.8]]  # accelerate, coast, brake

def parse_sensors(sensor_string):
    """Parse TORCS sensor string into state vector."""
    parts = sensor_string.replace('(', ' ').replace(')', ' ').split()

    state = {}
    i = 0
    while i < len(parts):
        if parts[i] in ['angle', 'speedX', 'speedY', 'speedZ', 'trackPos', 'rpm', 'gear', 'damage', 'fuel', 'racePos', 'distRaced']:
            if i + 1 < len(parts):
                state[parts[i]] = float(parts[i + 1])
                i += 2
            else:
                i += 1
        elif parts[i] == 'track':
            track_values = []
            for j in range(1, 20):
                if i + j < len(parts):
                    track_values.append(float(parts[i + j]))
            state['track'] = track_values
            i += 20
        else:
            i += 1

    angle = state.get('angle', 0)
    speed_x = state.get('speedX', 0)
    track_pos = state.get('trackPos', 0)

    track = state.get('track', [200] * 19)
    track_sensors = [min(t / 50.0, 1.0) if t > 0 else 1.0 for t in track]

    left_sensors = track_sensors[:9]
    right_sensors = track_sensors[10:]
    front_sensor = track_sensors[9]

    left_min = min(left_sensors) if left_sensors else 1.0
    right_min = min(right_sensors) if right_sensors else 1.0
    left_avg = sum(left_sensors) / len(left_sensors) if left_sensors else 1.0
    right_avg = sum(right_sensors) / len(right_sensors) if right_sensors else 1.0

    vector = [
        angle,
        speed_x,
        state.get('speedY', 0),
        track_pos,
        state.get('rpm', 0) / 10000.0,
        state.get('gear', 1),
        state.get('damage', 0) / 100.0,
        front_sensor,
        left_min,
        right_min,
        left_avg,
        right_avg,
        left_min - right_min,
        left_avg - right_avg,
        abs(track_pos) + abs(angle),
        *track_sensors,
        0.0  # Padding to reach 35
    ]

    return np.array(vector, dtype=np.float32), state


def create_control_string(steer, accel, brake, gear=1):
    """Create TORCS control command string."""
    return f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)"

class Actor(nn.Module):
    def __init__(self, state_size=STATE_SIZE, hidden_size=128):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.steer_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())  # [-1, 1]
        self.speed_head = nn.Sequential(nn.Linear(hidden_size, 3))  # accel, coast, brake
        self.gear_head = nn.Sequential(nn.Linear(hidden_size, 3))   # gear -1, 1, 2 (no 0)

    def forward(self, x):
        x = self.fc(x)
        steer = self.steer_head(x)
        speed_logits = self.speed_head(x)
        gear_logits = self.gear_head(x)
        return steer, speed_logits, gear_logits


class Critic(nn.Module):
    def __init__(self, state_size=STATE_SIZE, action_size=1 + 1 + 1, hidden_size=128):
        super(Critic, self).__init__()
        self.state_fc = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU())
        self.action_fc = nn.Sequential(nn.Linear(action_size, hidden_size), nn.ReLU())
        self.q_head = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, state, action):
        s = self.state_fc(state)
        a = self.action_fc(action)
        return self.q_head(torch.cat([s, a], dim=1))


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )
    def __len__(self):
        return len(self.buffer)


class OUNoise:
    def __init__(self, mu=0.0, theta=0.6, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn()
        self.state += dx
        return self.state


def train_ddpg(num_episodes=100000, max_steps=2000):
    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    target_actor = Actor().to(DEVICE)
    target_critic = Critic().to(DEVICE)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    memory = ReplayBuffer()
    noise = OUNoise()

    gamma = 0.99
    tau = 0.001

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    host, port = "localhost", 3001
    server_address = (host, port)
    client_id = "SCR"

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode+1} ===")
        noise.reset()
        stuck_counter = 0
        prev_dist = 0

        # Connect
        init_msg = f"{client_id}(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)"
        sock.sendto(init_msg.encode(), server_address)
        _ = sock.recvfrom(1000)

        data, _ = sock.recvfrom(1000)
        state_vector, _ = parse_sensors(data.decode())
        total_reward = 0

        for step in range(max_steps):
            state = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                steer_out, speed_logits, gear_logits = actor(state)
                steer = steer_out[0, 0].cpu().item() + noise.sample()
                speed_idx = torch.argmax(speed_logits, dim=1).item()
                gear_idx = torch.argmax(gear_logits, dim=1).item()

            # Map gear index: 0 -> -1, 1 -> 1, 2 -> 2
            gear = [-1, 1, 2][gear_idx]
            accel, brake = SPEED_ACTIONS[speed_idx]
            steer = float(np.clip(steer, -1, 1))

            control_str = create_control_string(steer, accel, brake, gear)
            sock.sendto(control_str.encode(), server_address)

            data, _ = sock.recvfrom(1000)
            if "***" in data.decode():
                break

            next_state_vector, state_dict = parse_sensors(data.decode())

            # Reward shaping
            # Reward shaping
            speed_x = state_dict.get("speedX", 0)
            angle = abs(state_dict.get("angle", 0))
            track_pos = abs(state_dict.get("trackPos", 0))
            track = state_dict.get("track", [200] * 19)
            dist_raced = state_dict.get("distRaced", 0)
            stuck = speed_x < 0.1

            # Forward motion reward
            reward = speed_x * 100

            # Penalize bad alignment
            reward -= angle * 5
            reward -= track_pos * 5
            reward -= abs(steer) * 2

            # Encourage seeing further ahead
            reward += min(track[9], 100) * 0.1

            # Escalating penalty for being stuck
            if stuck:
                stuck_counter += 1
                reward -= stuck_counter * 10
            else:
                stuck_counter = 0

            # Directly reward forward track progress
            progress = dist_raced - prev_dist
            reward += max(progress, 0) * 10  # Avoid negative progress
            prev_dist = dist_raced

            done = False
            if stuck_counter > 50 or track_pos > 1.2:
                reward -= 100
                done = True

            total_reward += reward

            # Normalize and store transition
            action_array = [steer, speed_idx / 2.0, (gear + 1) / 3.0]
            memory.add((state_vector, action_array, reward, next_state_vector, done))

            state_vector = next_state_vector

            if done:
                break

            # Learning step
            if len(memory) > 128:
                s_batch, a_batch, r_batch, s2_batch, d_batch = memory.sample(64)

                with torch.no_grad():
                    steer2, speed2, gear2 = target_actor(s2_batch)
                    speed_idx2 = torch.argmax(speed2, dim=1, keepdim=True).float() / 2.0
                    gear_idx2 = torch.argmax(gear2, dim=1, keepdim=True).float() / 3.0
                    a2 = torch.cat([steer2, speed_idx2, gear_idx2], dim=1)
                    q_target = r_batch + gamma * target_critic(s2_batch, a2) * (1 - d_batch)

                q_val = critic(s_batch, a_batch)
                critic_loss = nn.MSELoss()(q_val, q_target)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                # Actor loss
                steer_a, speed_a, gear_a = actor(s_batch)
                speed_a_idx = torch.argmax(speed_a, dim=1, keepdim=True).float() / 2.0
                gear_a_idx = torch.argmax(gear_a, dim=1, keepdim=True).float() / 3.0
                a_pred = torch.cat([steer_a, speed_a_idx, gear_a_idx], dim=1)

                actor_loss = -critic(s_batch, a_pred).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # Soft update
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        print(f"Episode {episode+1} Total Reward: {total_reward:.2f}")

        if (episode + 1) % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(actor.state_dict(), f"models/ddpg_actor_ep{episode+1}.pth")


if __name__ == "__main__":
    train_ddpg(num_episodes=100000)
