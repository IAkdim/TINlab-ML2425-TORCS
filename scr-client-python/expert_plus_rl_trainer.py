#!/usr/bin/env python3
"""
Expert + RL hybrid trainer.
Start with expert model, then fine-tune with reinforcement learning.
"""
import socket
import select
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import joblib


class ExpertDrivingNet(nn.Module):
    """Same architecture as supervised model."""
    
    def __init__(self, input_size=21, hidden_size=256, dropout=0.2):
        super(ExpertDrivingNet, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        # Separate heads for each action
        self.acceleration_head = nn.Sequential(
            nn.Linear(hidden_size//2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output [0, 1]
        )
        
        self.brake_head = nn.Sequential(
            nn.Linear(hidden_size//2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output [0, 1]
        )
        
        self.steering_head = nn.Sequential(
            nn.Linear(hidden_size//2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output [-1, 1]
        )
    
    def forward(self, x):
        features = self.shared(x)
        
        accel = self.acceleration_head(features)
        brake = self.brake_head(features)
        steering = self.steering_head(features)
        
        return torch.cat([accel, brake, steering], dim=1)


def parse_sensors(sensor_string):
    """Parse TORCS sensor string into the format expected by expert model."""
    # Remove parentheses and split by spaces
    parts = sensor_string.replace('(', ' ').replace(')', ' ').split()
    
    state = {}
    i = 0
    while i < len(parts):
        if parts[i] in ['angle', 'speedX', 'speedY', 'speedZ', 'trackPos', 'rpm', 'gear', 'damage', 'fuel', 'racePos']:
            if i + 1 < len(parts):
                state[parts[i]] = float(parts[i + 1])
                i += 2
            else:
                i += 1
        elif parts[i] == 'track':
            # Get 19 track sensor values
            track_values = []
            for j in range(1, 20):
                if i + j < len(parts):
                    track_values.append(float(parts[i + j]))
            state['track'] = track_values
            i += 20
        else:
            i += 1
    
    # Create feature vector matching the training data format
    speed = state.get('speedX', 0)
    track_pos = state.get('trackPos', 0)
    angle = state.get('angle', 0)
    
    # Get 18 track sensors (expert data has TRACK_EDGE_0 to TRACK_EDGE_17)
    track = state.get('track', [200] * 19)
    track_sensors = []
    for i in range(18):  # Only use first 18 sensors to match expert data
        if i < len(track):
            track_sensors.append(track[i])
        else:
            track_sensors.append(200.0)
    
    # Create the exact feature vector format from training data
    features = [speed, track_pos, angle] + track_sensors
    
    return np.array(features, dtype=np.float32), state


def create_control_string(accel, brake, steer, gear=1):
    """Create TORCS control string."""
    return f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)"


def calculate_reward(current_state_dict, prev_state_dict):
    """Reward function focused on recovery and progress."""
    reward = 0.0
    
    # Speed reward (encourage forward movement)
    speed = current_state_dict.get('speedX', 0) * 3.6  # km/h
    reward += speed * 0.01
    
    # Track position penalty (stay on track)
    track_pos = abs(current_state_dict.get('trackPos', 0))
    if track_pos < 0.5:
        reward += 2.0  # Big bonus for staying on track
    else:
        reward -= track_pos * 5.0  # Penalty for going off track
    
    # Progress reward
    distance = current_state_dict.get('distRaced', 0)
    prev_distance = prev_state_dict.get('distRaced', 0) if prev_state_dict else 0
    progress = distance - prev_distance
    reward += progress * 0.1
    
    # Damage penalty
    damage = current_state_dict.get('damage', 0)
    prev_damage = prev_state_dict.get('damage', 0) if prev_state_dict else 0
    new_damage = damage - prev_damage
    reward -= new_damage * 0.1
    
    # Recovery bonus (reward getting back on track)
    if prev_state_dict:
        prev_track_pos = abs(prev_state_dict.get('trackPos', 0))
        if prev_track_pos > 0.8 and track_pos < 0.5:
            reward += 10.0  # Big bonus for recovery
    
    return reward


def train_expert_plus_rl(expert_model_path, scaler_path, episodes=100):
    """Train the expert model with RL fine-tuning."""
    
    # Load expert model
    model = ExpertDrivingNet(input_size=21)
    if os.path.exists(expert_model_path):
        model.load_state_dict(torch.load(expert_model_path, map_location='cpu'))
        print(f"Loaded expert model from {expert_model_path}")
    else:
        print(f"Expert model {expert_model_path} not found, starting from scratch")
    
    # Load scaler
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for fine-tuning
    
    # Experience replay
    replay_buffer = deque(maxlen=10000)
    
    # Training parameters
    epsilon = 0.1  # Lower exploration since we start with expert
    gamma = 0.99
    batch_size = 32
    
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    print(f"Starting Expert + RL training for {episodes} episodes...")
    
    for episode in range(episodes):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (host, port)
        
        try:
            # Initialize connection
            init_msg = f"{client_id}(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)"
            sock.sendto(init_msg.encode(), server_address)
            
            # Wait for identification
            ready = select.select([sock], [], [], 10.0)
            if not ready[0]:
                print(f"Episode {episode}: Connection timeout")
                continue
            
            data, addr = sock.recvfrom(1000)
            response = data.decode().strip()
            
            if not ("identified" in response or response.startswith("(")):
                print(f"Episode {episode}: Failed to identify")
                continue
            
            print(f"Episode {episode + 1}/{episodes}: Starting...")
            
            episode_reward = 0
            step_count = 0
            prev_state_dict = None
            episode_buffer = []
            
            while step_count < 2000:  # Max steps per episode
                # Get sensor data
                ready = select.select([sock], [], [], 5.0)
                if not ready[0]:
                    break
                
                data, addr = sock.recvfrom(1000)
                sensors = data.decode().strip()
                
                if sensors in ["***shutdown***", "***restart***"]:
                    break
                
                # Parse state
                features, state_dict = parse_sensors(sensors)
                
                # Normalize features if scaler available
                if scaler is not None:
                    features = scaler.transform(features.reshape(1, -1))[0]
                
                # Get action from model (with small epsilon exploration)
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    actions = model(features_tensor)[0]
                    
                    # Add small noise for exploration
                    if random.random() < epsilon:
                        noise = torch.randn_like(actions) * 0.1
                        actions = torch.clamp(actions + noise, 
                                            torch.tensor([-1.0, 0.0, -1.0]), 
                                            torch.tensor([1.0, 1.0, 1.0]))
                    
                    acceleration = float(actions[0].item())
                    brake = float(actions[1].item())
                    steering = float(actions[2].item())
                
                # Simple gear logic
                rpm = state_dict.get('rpm', 0)
                current_gear = int(state_dict.get('gear', 1))
                if rpm > 6000:
                    gear = min(6, current_gear + 1)
                elif rpm < 2500 and current_gear > 1:
                    gear = current_gear - 1
                else:
                    gear = max(1, current_gear)
                
                # Send control
                control = create_control_string(acceleration, brake, steering, gear)
                sock.sendto(control.encode(), server_address)
                
                # Calculate reward
                reward = calculate_reward(state_dict, prev_state_dict)
                episode_reward += reward
                
                # Store experience
                if prev_state_dict is not None:
                    episode_buffer.append({
                        'features': prev_features,
                        'actions': [prev_acceleration, prev_brake, prev_steering],
                        'reward': reward,
                        'next_features': features,
                        'done': False
                    })
                
                prev_state_dict = state_dict.copy()
                prev_features = features.copy()
                prev_acceleration, prev_brake, prev_steering = acceleration, brake, steering
                step_count += 1
            
            # Send shutdown to end episode properly
            try:
                sock.sendto("***shutdown***".encode(), server_address)
            except:
                pass
            
            # Mark last experience as done
            if episode_buffer:
                episode_buffer[-1]['done'] = True
                replay_buffer.extend(episode_buffer)
            
            print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, Steps={step_count}")
            
            # Wait a moment for server to reset
            time.sleep(1.0)
            
            # Train on replay buffer
            if len(replay_buffer) >= batch_size:
                # Sample batch
                batch = random.sample(replay_buffer, batch_size)
                
                # Prepare training data
                features_batch = torch.FloatTensor([exp['features'] for exp in batch])
                actions_batch = torch.FloatTensor([exp['actions'] for exp in batch])
                rewards_batch = torch.FloatTensor([exp['reward'] for exp in batch])
                next_features_batch = torch.FloatTensor([exp['next_features'] for exp in batch])
                done_batch = torch.BoolTensor([exp['done'] for exp in batch])
                
                # Current Q-values (actions from model)
                current_actions = model(features_batch)
                
                # Target actions (mix of current actions and rewards)
                with torch.no_grad():
                    next_actions = model(next_features_batch)
                    # Simple target: current action + reward signal
                    targets = actions_batch.clone()
                    
                    # Adjust actions based on reward
                    for i, reward in enumerate(rewards_batch):
                        if reward < -1.0:  # Bad outcome, adjust away from this action
                            targets[i] = targets[i] * 0.9  # Reduce action magnitude
                        elif reward > 1.0:  # Good outcome, reinforce
                            targets[i] = targets[i] * 1.1  # Slightly increase
                    
                    # Clamp to valid ranges
                    targets[:, 0] = torch.clamp(targets[:, 0], 0, 1)  # accel
                    targets[:, 1] = torch.clamp(targets[:, 1], 0, 1)  # brake
                    targets[:, 2] = torch.clamp(targets[:, 2], -1, 1)  # steering
                
                # Loss and backprop
                loss = nn.MSELoss()(current_actions, targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if episode % 10 == 0:
                    print(f"  Training loss: {loss.item():.6f}")
            
            # Decay exploration
            epsilon = max(0.01, epsilon * 0.995)
            
            # Save model periodically
            if (episode + 1) % 25 == 0:
                torch.save(model.state_dict(), f'models/expert_rl_episode_{episode+1}.pth')
                print(f"Saved model at episode {episode + 1}")
        
        except KeyboardInterrupt:
            print("Training interrupted")
            break
        except Exception as e:
            print(f"Episode {episode}: Error {e}")
        finally:
            sock.close()
    
    # Save final model
    torch.save(model.state_dict(), 'models/expert_rl_final.pth')
    print("Training completed! Final model saved to models/expert_rl_final.pth")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Expert + RL hybrid training")
    parser.add_argument("--expert-model", default="models/expert_driving_best.pth",
                       help="Path to expert model")
    parser.add_argument("--scaler", default="models/expert_scaler.pkl",
                       help="Path to feature scaler")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of RL episodes")
    
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    train_expert_plus_rl(args.expert_model, args.scaler, args.episodes)


if __name__ == "__main__":
    main()
