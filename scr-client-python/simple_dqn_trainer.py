#!/usr/bin/env python3
"""
Simple DQN training script that mimics the working simple driver.
Based on the C++ client structure but with DQN learning.
"""
import socket
import select
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os


class DQN(nn.Module):
    """Simple DQN network."""
    def __init__(self, state_size=29, action_size=14, hidden_size=64):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """DQN agent for TORCS."""
    def __init__(self, state_size=29, action_size=14, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Improved discrete actions: [steer, accel, brake]
        self.actions = [
            # High-speed actions
            [-0.3, 1.0, 0.0],   # Left + full accel
            [-0.1, 1.0, 0.0],   # Slight left + full accel
            [0.0, 1.0, 0.0],    # Straight + full accel
            [0.1, 1.0, 0.0],    # Slight right + full accel
            [0.3, 1.0, 0.0],    # Right + full accel
            
            # Medium-speed actions
            [-0.5, 0.5, 0.0],   # Sharp left + medium accel
            [0.0, 0.5, 0.0],    # Straight + medium accel
            [0.5, 0.5, 0.0],    # Sharp right + medium accel
            
            # Braking actions
            [-0.2, 0.0, 0.5],   # Left + brake
            [0.0, 0.0, 0.5],    # Straight + brake
            [0.2, 0.0, 0.5],    # Right + brake
            
            # Emergency actions
            [-0.8, 0.0, 1.0],   # Sharp left + full brake
            [0.8, 0.0, 1.0],    # Sharp right + full brake
            [0.0, 0.0, 0.0],    # Coast
        ]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def parse_sensors(sensor_string):
    """Parse TORCS sensor string into state vector."""
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
    
    # Create state vector with exactly 29 features
    vector = [
        state.get('angle', 0),
        state.get('speedX', 0),
        state.get('speedY', 0),
        state.get('trackPos', 0),
        state.get('rpm', 0) / 10000.0,  # Normalize
        state.get('gear', 1),
        state.get('damage', 0) / 100.0,  # Normalize
    ]
    
    # Add 19 track sensors (normalized)
    track = state.get('track', [200] * 19)
    for i in range(19):
        if i < len(track):
            vector.append(min(track[i] / 200.0, 1.0))  # Normalize to [0,1]
        else:
            vector.append(1.0)
    
    # Add 3 more features to reach 29 (you can add more sensor data here)
    vector.extend([
        state.get('speedZ', 0),  # speedZ
        state.get('fuel', 94) / 100.0,  # normalized fuel
        state.get('racePos', 1) / 10.0,  # normalized race position
    ])
    
    # Ensure exactly 29 features
    while len(vector) < 29:
        vector.append(0.0)
    vector = vector[:29]  # Truncate if too long
    
    return np.array(vector, dtype=np.float32), state


def calculate_reward(prev_state, current_state, prev_raw_state, current_raw_state):
    """Calculate improved reward for DQN training."""
    reward = 0.0
    
    # Extract raw values for better reward calculation
    speed = current_state[1]  # speedX
    track_pos = current_state[3]  # trackPos
    damage = current_raw_state.get('damage', 0)
    prev_damage = prev_raw_state.get('damage', 0) if prev_raw_state else 0
    
    # 1. PROGRESS REWARD (most important)
    if prev_raw_state:
        distance_progress = current_raw_state.get('distRaced', 0) - prev_raw_state.get('distRaced', 0)
        reward += distance_progress * 0.1  # Big reward for forward progress
    
    # 2. SPEED REWARD (encourage going fast)
    if abs(track_pos) < 1.0:  # Only reward speed when on track
        speed_reward = min(speed / 30.0, 1.0) * 2.0  # Max +2.0 for high speed
        reward += speed_reward
    
    # 3. TRACK POSITION REWARD (stay in center)
    if abs(track_pos) < 1.0:
        center_reward = (1.0 - abs(track_pos)) * 1.0  # Max +1.0 for center
        reward += center_reward
    else:
        # Moderate penalty for going off track (since we don't end episodes early)
        reward -= 2.0
    
    # 4. DAMAGE PENALTY (avoid crashes)
    if damage > prev_damage:
        damage_increase = damage - prev_damage
        reward -= damage_increase * 0.1  # Penalty for new damage
    
    # 5. WALL AVOIDANCE (don't get too close)
    track_sensors = current_state[7:26]  # 19 track sensors
    min_distance = min(track_sensors)
    if min_distance < 0.05:  # Very close to wall
        reward -= 2.0
    elif min_distance < 0.1:  # Close to wall
        reward -= 0.5
    
    # 6. BACKWARDS PENALTY (don't go backwards)
    if speed < -1.0:
        reward -= 2.0
    
    # 7. RECOVERY REWARD (getting back on track)
    if prev_raw_state:
        prev_track_pos = abs(prev_state[3])
        current_track_pos = abs(track_pos)
        if prev_track_pos > 1.0 and current_track_pos < prev_track_pos:
            # Reward for getting closer to track
            reward += (prev_track_pos - current_track_pos) * 5.0
    
    # 8. LAP COMPLETION BONUS
    if prev_raw_state:
        current_lap_time = current_raw_state.get('curLapTime', 0)
        prev_lap_time = prev_raw_state.get('curLapTime', 0)
        
        # If lap time reset to 0, we completed a lap
        if prev_lap_time > 10.0 and current_lap_time < 5.0:
            reward += 100.0  # Huge bonus for completing lap
            print(f"LAP COMPLETED! Bonus: +100.0")
    
    # 9. STUCK DETECTION (not moving)
    if abs(speed) < 0.1 and abs(track_pos) < 1.0:
        reward -= 0.2  # Small penalty for being stuck
    
    return reward


def create_control_string(steer, accel, brake, gear=1):
    """Create TORCS control string."""
    return f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)"


def main():
    """Main training function."""
    print("Starting DQN TORCS training...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Initialize DQN agent
    agent = DQNAgent()
    
    # Connection parameters
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    # Training parameters
    num_episodes = 50
    max_steps = 10000
    
    # Socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (host, port)
    
    print(f"Connecting to {host}:{port}")
    
    try:
        episode = 0
        while episode < num_episodes:
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            # Identification phase
            print("Identifying to server...")
            init_msg = f"{client_id}(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)"
            
            # Send identification message once
            sock.sendto(init_msg.encode(), server_address)
            print(f"Sent: {init_msg}")
            
            # Wait for identification confirmation
            identified = False
            ready = select.select([sock], [], [], 10.0)
            if ready[0]:
                data, addr = sock.recvfrom(1000)
                response = data.decode().strip()
                print(f"Received: {response}")
                
                if "identified" in response:
                    identified = True
                    print("Successfully identified! Starting episode...")
                elif response.startswith("("):
                    # Already in game, this is sensor data
                    print("Already in game, starting episode...")
                    identified = True
                else:
                    print(f"DEBUG: Unexpected response: '{response}'")
            else:
                print("No response from server")
            
            print(f"DEBUG: identified = {identified}")
            if not identified:
                print("Failed to identify to server")
                break
            
            print("Starting episode loop...")
            
            # Episode loop
            step_count = 0
            episode_reward = 0
            prev_state = None
            prev_action = None
            prev_raw_state = None
            
            while step_count < max_steps:
                # Receive sensor data
                ready = select.select([sock], [], [], 5.0)
                if not ready[0]:
                    print("No sensor data received, ending episode")
                    break
                
                data, addr = sock.recvfrom(1000)
                sensors = data.decode().strip()
                
                # Check for special messages
                if sensors == "***shutdown***":
                    print("Server shutdown")
                    sock.sendto(b"", server_address)
                    episode = num_episodes  # End training
                    break
                elif sensors == "***restart***":
                    print("Episode restart")
                    sock.sendto(b"", server_address)
                    break
                
                # Parse state
                current_state, current_raw_state = parse_sensors(sensors)
                
                # Get action from agent
                action_idx = agent.act(current_state, training=True)
                steer, accel, brake = agent.actions[action_idx]
                
                # Simple gear logic
                rpm = current_state[4] * 10000  # Denormalize
                if rpm > 6000:
                    gear = min(6, max(1, int(current_state[5]) + 1))
                elif rpm < 2500:
                    gear = max(1, int(current_state[5]) - 1)
                else:
                    gear = max(1, int(current_state[5]))
                
                # Create and send control command
                if step_count < max_steps - 1:
                    control = create_control_string(steer, accel, brake, gear)
                else:
                    control = "(meta 1)"  # End episode
                
                sock.sendto(control.encode(), server_address)
                
                # Calculate reward and store experience
                if prev_state is not None:
                    reward = calculate_reward(prev_state, current_state, prev_raw_state, current_raw_state)
                    episode_reward += reward
                    
                    # Print reward details every 50 steps for debugging
                    if step_count % 50 == 0:
                        print(f"Step {step_count}: Reward = {reward:.3f}, Speed = {current_state[1]*3.6:.1f}km/h, TrackPos = {current_state[3]:.3f}")
                    
                    # Check for crash/stuck conditions (apply penalties but don't end episode)
                    done = False
                    if abs(current_state[3]) > 1.0:  # Off track
                        reward -= 5.0  # Crash penalty (but continue episode)
                        if step_count % 50 == 0:  # Don't spam the message
                            print(f"OFF TRACK: TrackPos = {current_state[3]:.3f}")
                    elif current_raw_state.get('damage', 0) > 50:  # Too much damage
                        reward -= 10.0
                        if step_count % 50 == 0:
                            print(f"DAMAGED: Damage = {current_raw_state.get('damage', 0)}")
                    elif step_count > 100 and abs(current_state[1]) < 0.1:  # Stuck (not moving)
                        reward -= 2.0
                        if step_count % 50 == 0:
                            print("STUCK: Not moving!")
                    
                    # Only mark as done at the very end of episode
                    if step_count >= max_steps - 1:
                        done = True
                    
                    agent.remember(prev_state, prev_action, reward, current_state, done)
                    
                    # Train every 4 steps
                    if step_count % 4 == 0:
                        agent.replay()
                    
                    # Update target network every 100 steps
                    if step_count % 100 == 0:
                        agent.update_target_network()
                
                prev_state = current_state
                prev_raw_state = current_raw_state
                prev_action = action_idx
                step_count += 1
            
            print(f"Episode {episode + 1} completed: Reward = {episode_reward:.2f}, Steps = {step_count}, Epsilon = {agent.epsilon:.3f}")
            
            # Save model every 10 episodes
            if (episode + 1) % 10 == 0:
                model_path = f"models/dqn_simple_episode_{episode + 1}.pth"
                torch.save(agent.q_network.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            
            episode += 1
            time.sleep(1)  # Brief pause between episodes
        
        # Final model save
        torch.save(agent.q_network.state_dict(), "models/dqn_simple_final.pth")
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
