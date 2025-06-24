#!/usr/bin/env python3
"""
Continuous control DQN trainer with fluid motions and recovery.
Uses hybrid discrete-continuous actions for better driving.
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


class ContinuousDQN(nn.Module):
    """DQN network with continuous steering output."""
    def __init__(self, state_size=35, hidden_size=128):
        super(ContinuousDQN, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Steering head (continuous: -1 to +1)
        self.steering_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Speed action head (discrete: accelerate, coast, brake)
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [accel, coast, brake]
        )
    
    def forward(self, x):
        features = self.features(x)
        steering = self.steering_head(features)
        speed_q = self.speed_head(features)
        return steering, speed_q


class HybridDQNAgent:
    """Hybrid agent with continuous steering + discrete speed control."""
    def __init__(self, state_size=35, lr=0.001):
        self.state_size = state_size
        self.memory = deque(maxlen=15000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.q_network = ContinuousDQN(state_size)
        self.target_network = ContinuousDQN(state_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Speed actions: [accel, brake]
        self.speed_actions = [
            [1.0, 0.0],    # Full accelerate
            [0.0, 0.0],    # Coast
            [0.0, 0.8],    # Brake
        ]
    
    def remember(self, state, action, reward, next_state, done):
        # action = (steering, speed_action_idx)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Get hybrid action: continuous steering + discrete speed."""
        
        if training and np.random.random() <= self.epsilon:
            # Guided exploration using heuristic steering + some randomness
            heuristic_steering = self._get_heuristic_steering(state)
            
            if np.random.random() < 0.7:  # 70% heuristic, 30% random
                steering = heuristic_steering + np.random.normal(0, 0.2)  # Add noise
            else:
                steering = np.random.uniform(-1.0, 1.0)  # Pure random
                
            speed_action = random.randrange(3)
        else:
            # Policy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                steering_out, speed_q = self.q_network(state_tensor)
                steering = float(steering_out[0].item())
                speed_action = int(torch.argmax(speed_q[0]).item())
        
        return steering, speed_action
    
    def _get_heuristic_steering(self, state):
        """Calculate heuristic steering based on track sensors."""
        # Extract key features
        angle = state[0]           # angle to track
        track_pos = state[3]       # position on track
        left_min = state[8]        # closest left obstacle
        right_min = state[9]       # closest right obstacle
        steering_bias = state[12]  # left_min - right_min
        
        # Basic steering logic
        steering = 0.0
        
        # 1. Correct track position (get back to center)
        if abs(track_pos) > 0.1:
            steering -= track_pos * 2.0  # Steer toward center
        
        # 2. Correct angle to track
        steering -= angle * 1.0
        
        # 3. Use track sensors for wall avoidance
        if left_min < 0.3 or right_min < 0.3:  # Close to walls
            steering += steering_bias * 3.0  # Steer away from closer wall
        
        # 4. Anticipate turns based on sensor asymmetry
        steering += steering_bias * 0.5
        
        # Clamp to valid range
        return max(-1.0, min(1.0, steering))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = [e[1] for e in batch]  # (steering, speed_action_idx) tuples
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q-values
        steering_out, speed_q = self.q_network(states)
        
        # Target Q-values
        with torch.no_grad():
            next_steering, next_speed_q = self.target_network(next_states)
            next_speed_values = next_speed_q.max(1)[0]
        
        # Loss for speed actions (standard DQN)
        speed_actions = torch.LongTensor([a[1] for a in actions])
        current_speed_q = speed_q.gather(1, speed_actions.unsqueeze(1))
        target_speed_q = rewards + (0.99 * next_speed_values * ~dones)
        speed_loss = nn.MSELoss()(current_speed_q.squeeze(), target_speed_q)
        
        # Loss for steering (regression to rewarded actions)
        target_steering = torch.FloatTensor([a[0] for a in actions])
        steering_loss = nn.MSELoss()(steering_out.squeeze(), target_steering)
        
        # Combined loss
        total_loss = speed_loss + 0.5 * steering_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
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
    
    # Create enhanced state vector focusing on steering cues
    angle = state.get('angle', 0)
    speed_x = state.get('speedX', 0)
    track_pos = state.get('trackPos', 0)
    
    # Get 19 track sensors and process them for steering cues
    track = state.get('track', [200] * 19)
    track_sensors = []
    for i in range(19):
        if i < len(track) and track[i] > 0:
            track_sensors.append(min(track[i] / 50.0, 1.0))  # Closer normalization for better sensitivity
        else:
            track_sensors.append(1.0)
    
    # Calculate steering cues from track sensors
    left_sensors = track_sensors[:9]   # Sensors 0-8 (left side)
    right_sensors = track_sensors[10:] # Sensors 10-18 (right side)  
    front_sensor = track_sensors[9]    # Sensor 9 (straight ahead)
    
    left_min = min(left_sensors) if left_sensors else 1.0
    right_min = min(right_sensors) if right_sensors else 1.0
    left_avg = sum(left_sensors) / len(left_sensors) if left_sensors else 1.0
    right_avg = sum(right_sensors) / len(right_sensors) if right_sensors else 1.0
    
    # Create state vector with exactly 35 features
    vector = [
        # Basic driving state (7 features)
        angle,                              # 0: angle to track
        speed_x,                           # 1: forward speed
        state.get('speedY', 0),            # 2: lateral speed  
        track_pos,                         # 3: position on track
        state.get('rpm', 0) / 10000.0,     # 4: normalized RPM
        state.get('gear', 1),              # 5: current gear
        state.get('damage', 0) / 100.0,    # 6: normalized damage
        
        # Key steering cues (8 features)
        front_sensor,                      # 7: distance straight ahead
        left_min,                          # 8: closest obstacle on left
        right_min,                         # 9: closest obstacle on right
        left_avg,                          # 10: average left clearance
        right_avg,                         # 11: average right clearance
        left_min - right_min,              # 12: steering bias (-1=steer right, +1=steer left)
        (left_avg - right_avg),            # 13: track curvature estimate
        abs(track_pos) + abs(angle),       # 14: total misalignment
        
        # Critical track sensors (20 features: keep all 19 + 1 padding)
        *track_sensors,                    # 15-33: all 19 track sensors
        0.0,                               # 34: padding to reach 35
    ]
    
    return np.array(vector, dtype=np.float32), state


def calculate_reward(prev_state, current_state, prev_raw_state, current_raw_state):
    """Enhanced reward function for continuous control."""
    reward = 0.0
    
    speed = current_state[1]  # speedX
    track_pos = current_state[3]  # trackPos
    angle = current_state[0]  # angle to track
    damage = current_raw_state.get('damage', 0)
    prev_damage = prev_raw_state.get('damage', 0) if prev_raw_state else 0
    
    # 1. PROGRESS REWARD (most important)
    if prev_raw_state:
        distance_progress = current_raw_state.get('distRaced', 0) - prev_raw_state.get('distRaced', 0)
        reward += distance_progress * 0.2  # Higher progress reward
    
    # 2. SPEED REWARD (only when on track and aligned)
    if abs(track_pos) < 1.0 and abs(angle) < 0.5:  # On track and reasonably aligned
        speed_reward = min(speed / 25.0, 1.0) * 3.0  # Max +3.0 for high speed
        reward += speed_reward
    
    # 3. TRACK POSITION REWARD 
    if abs(track_pos) < 1.0:
        center_reward = (1.0 - abs(track_pos)) * 1.5  # Stronger center reward
        reward += center_reward
    else:
        # Penalty for being off track
        reward -= min(abs(track_pos), 3.0)  # Cap penalty at -3.0
    
    # 4. ALIGNMENT REWARD (encourage pointing toward track center)
    if abs(track_pos) < 1.0:
        # Reward for being aligned with track direction
        alignment_reward = (1.0 - abs(angle)) * 0.5
        reward += alignment_reward
    
    # 5. RECOVERY REWARD (getting back on track)
    if prev_raw_state:
        prev_track_pos = abs(prev_state[3])
        current_track_pos = abs(track_pos)
        if prev_track_pos > 1.0 and current_track_pos < prev_track_pos:
            # Big reward for getting closer to track
            recovery_reward = (prev_track_pos - current_track_pos) * 10.0
            reward += recovery_reward
            if current_track_pos < 1.0:
                reward += 5.0  # Bonus for getting back on track
    
    # 6. DAMAGE PENALTY
    if damage > prev_damage:
        damage_increase = damage - prev_damage
        reward -= damage_increase * 0.2
    
    # 7. WALL AVOIDANCE
    track_sensors = current_state[7:26]  # 19 track sensors
    min_distance = min(track_sensors)
    if min_distance < 0.05:
        reward -= 3.0
    elif min_distance < 0.1:
        reward -= 1.0
    
    # 8. BACKWARDS PENALTY
    if speed < -2.0:
        reward -= 3.0
    
    # 9. LAP COMPLETION BONUS
    if prev_raw_state:
        current_lap_time = current_raw_state.get('curLapTime', 0)
        prev_lap_time = prev_raw_state.get('curLapTime', 0)
        
        if prev_lap_time > 10.0 and current_lap_time < 5.0:
            reward += 200.0  # Huge bonus for completing lap
            print(f"LAP COMPLETED! Bonus: +200.0")
    
    # 10. STUCK PENALTY
    if abs(speed) < 0.5 and abs(track_pos) < 1.0:
        reward -= 0.3
    
    return reward


def create_control_string(steer, accel, brake, gear=1):
    """Create TORCS control string."""
    return f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)"


def main():
    """Main training function."""
    print("Starting Continuous DQN TORCS training...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Initialize agent
    agent = HybridDQNAgent()
    
    # Connection parameters
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    # Training parameters
    num_episodes = 100
    max_steps = 3000  # Longer episodes for better learning
    
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
                    print("Already in game, starting episode...")
                    identified = True
            else:
                print("No response from server")
            
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
                    episode = num_episodes
                    break
                elif sensors == "***restart***":
                    print("Episode restart")
                    sock.sendto(b"", server_address)
                    break
                
                # Parse state
                current_state, current_raw_state = parse_sensors(sensors)
                
                # Get action from agent
                steering, speed_action_idx = agent.act(current_state, training=True)
                accel, brake = agent.speed_actions[speed_action_idx]
                
                # Clamp steering to valid range
                steering = max(-1.0, min(1.0, steering))
                
                # Simple gear logic
                rpm = current_raw_state.get('rpm', 0)
                current_gear = int(current_raw_state.get('gear', 1))
                if rpm > 6000:
                    gear = min(6, current_gear + 1)
                elif rpm < 2500 and current_gear > 1:
                    gear = current_gear - 1
                else:
                    gear = max(1, current_gear)
                
                # Create and send control command
                if step_count < max_steps - 1:
                    control = create_control_string(steering, accel, brake, gear)
                else:
                    control = "(meta 1)"  # End episode
                
                sock.sendto(control.encode(), server_address)
                
                # Calculate reward and store experience
                if prev_state is not None:
                    reward = calculate_reward(prev_state, current_state, prev_raw_state, current_raw_state)
                    episode_reward += reward
                    
                    # Print detailed progress every 100 steps
                    if step_count % 100 == 0:
                        speed_kmh = current_state[1] * 3.6
                        track_pos = current_state[3]
                        angle = current_state[0]
                        distance = current_raw_state.get('distRaced', 0)
                        
                        print(f"Step {step_count:4d}: Steer={steering:6.3f} Speed={speed_kmh:6.1f}km/h "
                              f"Pos={track_pos:6.3f} Angle={angle:6.3f} Dist={distance:7.1f}m "
                              f"Reward={reward:6.2f}")
                    
                    # Mark as done only at episode end
                    done = (step_count >= max_steps - 1)
                    
                    agent.remember(prev_state, prev_action, reward, current_state, done)
                    
                    # Train every 4 steps
                    if step_count % 4 == 0:
                        agent.replay()
                    
                    # Update target network every 200 steps
                    if step_count % 200 == 0:
                        agent.update_target_network()
                
                prev_state = current_state
                prev_raw_state = current_raw_state
                prev_action = (steering, speed_action_idx)
                step_count += 1
            
            final_distance = current_raw_state.get('distRaced', 0) if 'current_raw_state' in locals() else 0
            print(f"Episode {episode + 1} completed: Reward = {episode_reward:.2f}, Steps = {step_count}, Distance = {final_distance:.1f}m, Epsilon = {agent.epsilon:.3f}")
            
            # Save model every 20 episodes
            if (episode + 1) % 20 == 0:
                model_path = f"models/continuous_dqn_episode_{episode + 1}.pth"
                torch.save(agent.q_network.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            
            episode += 1
            time.sleep(1)
        
        # Final model save
        torch.save(agent.q_network.state_dict(), "models/continuous_dqn_final.pth")
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
