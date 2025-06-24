#!/usr/bin/env python3
"""
Expert + RL trainer using continuous session (like the working DQN trainers).
Load expert model and fine-tune with RL in a single long session.
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


def calculate_reward(current_raw_state, prev_raw_state):
    """Reward function focused on recovery and staying on track."""
    reward = 0.0
    
    # Basic state
    speed = current_raw_state.get('speedX', 0) * 3.6  # km/h
    track_pos = current_raw_state.get('trackPos', 0)
    angle = current_raw_state.get('angle', 0)
    damage = current_raw_state.get('damage', 0)
    
    # Previous state for comparison
    prev_damage = prev_raw_state.get('damage', 0) if prev_raw_state else 0
    prev_distance = prev_raw_state.get('distRaced', 0) if prev_raw_state else 0
    current_distance = current_raw_state.get('distRaced', 0)
    
    # 1. Progress reward (most important)
    progress = current_distance - prev_distance
    reward += progress * 0.1
    
    # 2. Track position reward (critical for staying on track)
    if abs(track_pos) < 0.3:  # Center of track
        reward += 3.0
    elif abs(track_pos) < 0.6:  # Still on track
        reward += 1.0
    elif abs(track_pos) < 1.0:  # Edge of track
        reward -= 2.0
    else:  # Off track
        reward -= 10.0
    
    # 3. Speed reward (but not too fast if off track)
    if abs(track_pos) < 0.8:  # Only reward speed when on track
        reward += min(speed * 0.01, 1.0)
    
    # 4. Damage penalty
    if damage > prev_damage:
        damage_increase = damage - prev_damage
        reward -= damage_increase * 0.2
    
    # 5. Angle penalty (stay aligned with track)
    reward -= abs(angle) * 2.0
    
    # 6. RECOVERY BONUS (key for learning from mistakes)
    if prev_raw_state:
        prev_track_pos = abs(prev_raw_state.get('trackPos', 0))
        # If we were off track and now we're back on
        if prev_track_pos > 0.8 and abs(track_pos) < 0.5:
            reward += 20.0  # Big bonus for recovery!
    
    return reward


def train_expert_rl_continuous(expert_model_path, scaler_path, num_episodes=20, max_steps=4000):
    """Train expert model with RL in continuous session."""
    
    # Load expert model
    model = ExpertDrivingNet(input_size=21)
    if os.path.exists(expert_model_path):
        model.load_state_dict(torch.load(expert_model_path, map_location='cpu'))
        print(f"✅ Loaded expert model from {expert_model_path}")
    else:
        print(f"❌ Expert model {expert_model_path} not found!")
        return
    
    # Load scaler
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ Loaded scaler from {scaler_path}")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)  # Very low LR for fine-tuning
    
    # Experience replay
    replay_buffer = deque(maxlen=5000)
    
    # Training parameters
    epsilon = 0.05  # Low exploration since we start with expert
    batch_size = 32
    
    # Connection
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (host, port)
    
    print(f"🚗 Starting Expert + RL training...")
    print(f"📊 Episodes: {num_episodes}, Max steps per episode: {max_steps}")
    print(f"🎯 Goal: Keep expert driving + learn recovery")
    
    try:
        episode = 0
        total_training_steps = 0
        
        while episode < num_episodes:
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            # Initialize connection for each episode
            init_msg = f"{client_id}(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)"
            sock.sendto(init_msg.encode(), server_address)
            
            identified = False
            ready = select.select([sock], [], [], 10.0)
            if ready[0]:
                data, addr = sock.recvfrom(1000)
                response = data.decode().strip()
                if "identified" in response or response.startswith("("):
                    identified = True
                    print("✅ Successfully identified!")
            
            if not identified:
                print("❌ Failed to identify")
                break
            
            # Episode training variables
            step_count = 0
            episode_reward = 0
            prev_raw_state = None
            prev_features = None
            prev_actions = None
            
            while step_count < max_steps:
                # Get sensor data
                ready = select.select([sock], [], [], 5.0)
                if not ready[0]:
                    break
                
                data, addr = sock.recvfrom(1000)
                sensors = data.decode().strip()
                
                if sensors == "***shutdown***":
                    episode = num_episodes  # End all training
                    break
                elif sensors == "***restart***":
                    break  # End this episode
                
                # Parse state
                features, raw_state = parse_sensors(sensors)
                
                # Normalize features if scaler available
                if scaler is not None:
                    features = scaler.transform(features.reshape(1, -1))[0]
                
                # Get action from model
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    actions = model(features_tensor)[0]
                    
                    # Add small exploration noise occasionally
                    if random.random() < epsilon:
                        noise = torch.randn_like(actions) * 0.05  # Very small noise
                        actions = torch.clamp(actions + noise, 
                                            torch.tensor([0.0, 0.0, -1.0]), 
                                            torch.tensor([1.0, 1.0, 1.0]))
                    
                    acceleration = float(actions[0].item())
                    brake = float(actions[1].item())
                    steering = float(actions[2].item())
                
                # Simple gear logic
                rpm = raw_state.get('rpm', 0)
            current_gear = int(raw_state.get('gear', 1))
            if rpm > 6000:
                gear = min(6, current_gear + 1)
            elif rpm < 2500 and current_gear > 1:
                gear = current_gear - 1
            else:
                gear = max(1, current_gear)
            
            # Send control
            control = create_control_string(acceleration, brake, steering, gear)
            sock.sendto(control.encode(), server_address)
            
            # Calculate reward and store experience
            if prev_raw_state is not None and prev_features is not None and prev_actions is not None:
                reward = calculate_reward(raw_state, prev_raw_state)
                total_reward += reward
                
                # Store experience
                experience = {
                    'state': prev_features.copy(),
                    'action': prev_actions.copy(),
                    'reward': reward,
                    'next_state': features.copy(),
                    'done': False
                }
                replay_buffer.append(experience)
                
                # Train model if we have enough experiences
                if len(replay_buffer) >= batch_size and step_count % 4 == 0:  # Train every 4 steps
                    # Sample batch
                    batch = random.sample(replay_buffer, batch_size)
                    
                    # Convert to tensors
                    states = torch.FloatTensor(np.array([exp['state'] for exp in batch]))
                    actions = torch.FloatTensor(np.array([exp['action'] for exp in batch]))
                    rewards = torch.FloatTensor([exp['reward'] for exp in batch])
                    next_states = torch.FloatTensor(np.array([exp['next_state'] for exp in batch]))
                    
                    # Current action predictions
                    current_actions = model(states)
                    
                    # Create targets: adjust actions based on rewards
                    with torch.no_grad():
                        targets = actions.clone()
                        
                        # Simple adjustment: if reward is bad, move away from action
                        for i, reward in enumerate(rewards):
                            if reward < -5.0:  # Bad situation
                                # Reduce action magnitudes slightly
                                targets[i] *= 0.95
                            elif reward > 5.0:  # Good situation
                                # Slightly reinforce current action
                                targets[i] = targets[i] * 1.02
                        
                        # Clamp to valid ranges
                        targets[:, 0] = torch.clamp(targets[:, 0], 0, 1)  # accel
                        targets[:, 1] = torch.clamp(targets[:, 1], 0, 1)  # brake
                        targets[:, 2] = torch.clamp(targets[:, 2], -1, 1)  # steering
                    
                    # Loss and update
                    loss = nn.MSELoss()(current_actions, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    
                    training_steps += 1
            
            # Store current state as previous for next iteration
            prev_raw_state = raw_state.copy()
            prev_features = features.copy()
            prev_actions = np.array([acceleration, brake, steering])
            
            step_count += 1
            
            # Progress reporting
            if step_count - last_progress_print >= 500:
                speed_kmh = raw_state.get('speedX', 0) * 3.6
                track_pos = raw_state.get('trackPos', 0)
                damage = raw_state.get('damage', 0)
                distance = raw_state.get('distRaced', 0)
                
                avg_reward = total_reward / max(1, step_count)
                
                print(f"Step {step_count:5d}: Speed={speed_kmh:5.1f}km/h, Pos={track_pos:6.3f}, "
                      f"Dist={distance:7.1f}m, Damage={damage:4.0f}, AvgReward={avg_reward:6.2f}, "
                      f"TrainingSteps={training_steps}")
                
                last_progress_print = step_count
            
            # Decay exploration
            if step_count % 1000 == 0:
                epsilon = max(0.01, epsilon * 0.98)
        
        print(f"\n🏁 Training completed!")
        print(f"📊 Total steps: {step_count}")
        print(f"🎯 Training updates: {training_steps}")
        print(f"📈 Average reward: {total_reward / max(1, step_count):.2f}")
        
        # Save fine-tuned model
        torch.save(model.state_dict(), 'models/expert_rl_finetuned.pth')
        print(f"💾 Saved fine-tuned model to: models/expert_rl_finetuned.pth")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        # Save current model
        torch.save(model.state_dict(), 'models/expert_rl_interrupted.pth')
        print("💾 Saved interrupted model to: models/expert_rl_interrupted.pth")
    except Exception as e:
        print(f"❌ Error during training: {e}")
    finally:
        sock.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Expert + RL continuous training")
    parser.add_argument("--expert-model", default="models/expert_driving_best.pth",
                       help="Path to expert model")
    parser.add_argument("--scaler", default="models/expert_scaler.pkl",
                       help="Path to feature scaler")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Maximum training steps")
    
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    train_expert_rl_continuous(args.expert_model, args.scaler, args.steps)


if __name__ == "__main__":
    main()