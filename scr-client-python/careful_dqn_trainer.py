#!/usr/bin/env python3
"""
Careful DQN trainer that prioritizes safe driving over pure speed.
Focuses on staying on track and avoiding crashes.
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


class CarefulDQN(nn.Module):
    """DQN network optimized for safe driving."""
    def __init__(self, state_size=35, hidden_size=128):
        super(CarefulDQN, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Steering head (continuous: -1 to +1)
        self.steering_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Speed action head (more conservative options)
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # [light_accel, medium_accel, coast, light_brake, hard_brake]
        )
    
    def forward(self, x):
        features = self.features(x)
        steering = self.steering_head(features)
        speed_q = self.speed_head(features)
        return steering, speed_q


class CarefulDQNAgent:
    """Careful agent that prioritizes safety over speed."""
    def __init__(self, state_size=35, lr=0.0005):  # Lower learning rate for stability
        self.state_size = state_size
        self.memory = deque(maxlen=20000)
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Higher minimum for more exploration
        self.epsilon_decay = 0.998  # Slower decay for longer exploration
        
        self.q_network = CarefulDQN(state_size)
        self.target_network = CarefulDQN(state_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Conservative speed actions: [accel, brake]
        self.speed_actions = [
            [0.3, 0.0],    # Light accelerate (30%)
            [0.6, 0.0],    # Medium accelerate (60%)
            [0.0, 0.0],    # Coast
            [0.0, 0.3],    # Light brake
            [0.0, 0.8],    # Hard brake
        ]
        
        self.speed_names = ["Light Accel", "Medium Accel", "Coast", "Light Brake", "Hard Brake"]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Get careful action with safety-first exploration."""
        
        if training and np.random.random() <= self.epsilon:
            # Safety-guided exploration
            heuristic_steering = self._get_safety_steering(state)
            heuristic_speed = self._get_safety_speed(state)
            
            if np.random.random() < 0.8:  # 80% safe heuristic, 20% random
                steering = heuristic_steering + np.random.normal(0, 0.1)  # Small noise
                speed_action = heuristic_speed
            else:
                steering = np.random.uniform(-0.5, 0.5)  # Limited random steering
                speed_action = random.randrange(5)
        else:
            # Policy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                steering_out, speed_q = self.q_network(state_tensor)
                steering = float(steering_out[0].item())
                speed_action = int(torch.argmax(speed_q[0]).item())
        
        return steering, speed_action
    
    def _get_safety_steering(self, state):
        """Calculate safe steering that prioritizes staying on track."""
        angle = state[0]           # angle to track
        track_pos = state[3]       # position on track
        left_min = state[8]        # closest left obstacle
        right_min = state[9]       # closest right obstacle
        front_sensor = state[7]    # distance ahead
        
        steering = 0.0
        
        # 1. PRIORITY: Get back on track if off
        if abs(track_pos) > 0.8:  # Close to edge
            steering -= track_pos * 3.0  # Strong correction toward center
        elif abs(track_pos) > 0.3:  # Moving away from center
            steering -= track_pos * 1.5  # Moderate correction
        
        # 2. Correct angle to track (smooth alignment)
        steering -= angle * 1.2
        
        # 3. SAFETY: Avoid walls aggressively
        if left_min < 0.4 and right_min < 0.4:  # Narrow passage
            steering += (left_min - right_min) * 4.0  # Strong avoidance
        elif left_min < 0.6 or right_min < 0.6:  # Approaching wall
            steering += (left_min - right_min) * 2.0  # Moderate avoidance
        
        # 4. Anticipate turns (gentle)
        if front_sensor > 0.8:  # Plenty of space ahead
            steering += (left_min - right_min) * 0.3  # Gentle turn anticipation
        
        # Limit steering magnitude for smooth driving
        return max(-0.7, min(0.7, steering))
    
    def _get_safety_speed(self, state):
        """Calculate safe speed action based on situation."""
        speed_x = state[1]         # current speed
        track_pos = state[3]       # position on track
        left_min = state[8]        # closest left obstacle
        right_min = state[9]       # closest right obstacle
        front_sensor = state[7]    # distance ahead
        angle = abs(state[0])      # misalignment
        
        # Safety conditions that require braking
        if abs(track_pos) > 0.9:  # Very close to edge
            return 4  # Hard brake
        elif min(left_min, right_min) < 0.2:  # Very close to wall
            return 4  # Hard brake
        elif front_sensor < 0.3:  # Obstacle ahead
            return 3  # Light brake
        elif angle > 0.3:  # Badly misaligned
            return 3  # Light brake
        elif speed_x > 80:  # Too fast (limit to ~290 km/h)
            return 2  # Coast
        elif abs(track_pos) > 0.5 or angle > 0.15:  # Somewhat off track
            return 1  # Medium accel (careful)
        else:  # Safe to accelerate
            return 0  # Light accel (conservative)
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = [e[1] for e in batch]
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q-values
        steering_out, speed_q = self.q_network(states)
        
        # Target Q-values
        with torch.no_grad():
            next_steering, next_speed_q = self.target_network(next_states)
            next_speed_values = next_speed_q.max(1)[0]
        
        # Loss for speed actions
        speed_actions = torch.LongTensor([a[1] for a in actions])
        current_speed_q = speed_q.gather(1, speed_actions.unsqueeze(1))
        target_speed_q = rewards + (0.95 * next_speed_values * ~dones)  # Lower gamma for safety
        speed_loss = nn.MSELoss()(current_speed_q.squeeze(), target_speed_q)
        
        # Loss for steering (weighted by safety)
        target_steering = torch.FloatTensor([a[0] for a in actions])
        steering_loss = nn.MSELoss()(steering_out.squeeze(), target_steering)
        
        # Combined loss with emphasis on steering accuracy
        total_loss = speed_loss + 0.8 * steering_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def parse_sensors(sensor_string):
    """Parse TORCS sensor string into enhanced state vector."""
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
    
    # Create enhanced state vector
    angle = state.get('angle', 0)
    speed_x = state.get('speedX', 0)
    track_pos = state.get('trackPos', 0)
    
    # Process track sensors for safety
    track = state.get('track', [200] * 19)
    track_sensors = []
    for i in range(19):
        if i < len(track) and track[i] > 0:
            track_sensors.append(min(track[i] / 50.0, 1.0))
        else:
            track_sensors.append(1.0)
    
    # Safety-focused features
    left_sensors = track_sensors[:9]
    right_sensors = track_sensors[10:]
    front_sensor = track_sensors[9]
    
    left_min = min(left_sensors) if left_sensors else 1.0
    right_min = min(right_sensors) if right_sensors else 1.0
    left_avg = sum(left_sensors) / len(left_sensors) if left_sensors else 1.0
    right_avg = sum(right_sensors) / len(right_sensors) if right_sensors else 1.0
    
    vector = [
        # Basic state
        angle, speed_x, state.get('speedY', 0), track_pos,
        state.get('rpm', 0) / 10000.0, state.get('gear', 1), state.get('damage', 0) / 100.0,
        
        # Safety features
        front_sensor, left_min, right_min, left_avg, right_avg,
        left_min - right_min, left_avg - right_avg, abs(track_pos) + abs(angle),
        
        # All track sensors
        *track_sensors, 0.0  # Padding to 35
    ]
    
    return np.array(vector, dtype=np.float32), state


def calculate_safety_reward(prev_state, current_state, prev_raw_state, current_raw_state):
    """Safety-focused reward function."""
    reward = 0.0
    
    speed = current_state[1]
    track_pos = current_state[3]
    angle = current_state[0]
    damage = current_raw_state.get('damage', 0)
    prev_damage = prev_raw_state.get('damage', 0) if prev_raw_state else 0
    
    # 1. SAFETY FIRST: Heavy damage penalty
    if damage > prev_damage:
        damage_increase = damage - prev_damage
        reward -= damage_increase * 0.5  # Strong damage penalty
    
    # 2. TRACK POSITION REWARD (most important)
    if abs(track_pos) < 0.3:  # In center lane
        reward += 2.0
    elif abs(track_pos) < 0.6:  # Still on track
        reward += 1.0
    elif abs(track_pos) < 1.0:  # Edge of track
        reward -= 1.0
    else:  # Off track
        reward -= 5.0
    
    # 3. ALIGNMENT REWARD
    if abs(angle) < 0.1:  # Well aligned
        reward += 1.0
    elif abs(angle) < 0.3:  # Reasonably aligned
        reward += 0.5
    else:  # Badly aligned
        reward -= 2.0
    
    # 4. CONTROLLED SPEED REWARD (not maximum speed)
    if abs(track_pos) < 0.5 and abs(angle) < 0.2:  # Only reward speed when safe
        if 10 < speed < 60:  # Sweet spot: 36-216 km/h
            reward += min(speed / 40.0, 1.5)  # Max +1.5
        elif speed > 80:  # Too fast (>288 km/h)
            reward -= (speed - 80) * 0.02  # Penalty for excessive speed
    
    # 5. PROGRESS REWARD (when safe)
    if prev_raw_state and abs(track_pos) < 1.0:
        distance_progress = current_raw_state.get('distRaced', 0) - prev_raw_state.get('distRaced', 0)
        reward += distance_progress * 0.1
    
    # 6. WALL AVOIDANCE
    track_sensors = current_state[15:34]  # Track sensors
    min_distance = min(track_sensors)
    if min_distance < 0.1:
        reward -= 3.0
    elif min_distance < 0.3:
        reward -= 1.0
    
    # 7. STABILITY BONUS (reward smooth driving)
    if prev_raw_state:
        prev_track_pos = prev_state[3]
        prev_angle = prev_state[0]
        
        # Reward for stable track position
        if abs(track_pos - prev_track_pos) < 0.1:
            reward += 0.5
        
        # Reward for stable angle
        if abs(angle - prev_angle) < 0.1:
            reward += 0.5
    
    # 8. RECOVERY REWARD
    if prev_raw_state and abs(prev_state[3]) > 1.0 and abs(track_pos) < 1.0:
        reward += 10.0  # Big bonus for getting back on track
    
    return reward


def create_control_string(steer, accel, brake, gear=1):
    """Create TORCS control string."""
    return f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)"


def main():
    """Main training function."""
    print("Starting Careful DQN TORCS training...")
    print("Focus: Safe driving, track following, crash avoidance")
    
    os.makedirs("models", exist_ok=True)
    
    agent = CarefulDQNAgent()
    
    # Connection parameters
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    # Training parameters
    num_episodes = 30  # More episodes for careful learning
    max_steps = 4000   # Longer episodes
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (host, port)
    
    print(f"Connecting to {host}:{port}")
    
    try:
        episode = 0
        while episode < num_episodes:
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            # Identification
            init_msg = f"{client_id}(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)"
            sock.sendto(init_msg.encode(), server_address)
            
            identified = False
            ready = select.select([sock], [], [], 10.0)
            if ready[0]:
                data, addr = sock.recvfrom(1000)
                response = data.decode().strip()
                if "identified" in response or response.startswith("("):
                    identified = True
                    print("Successfully identified!")
            
            if not identified:
                print("Failed to identify")
                break
            
            # Episode loop
            step_count = 0
            episode_reward = 0
            prev_state = None
            prev_action = None
            prev_raw_state = None
            total_damage = 0
            
            while step_count < max_steps:
                ready = select.select([sock], [], [], 5.0)
                if not ready[0]:
                    break
                
                data, addr = sock.recvfrom(1000)
                sensors = data.decode().strip()
                
                if sensors == "***shutdown***":
                    episode = num_episodes
                    break
                elif sensors == "***restart***":
                    break
                
                current_state, current_raw_state = parse_sensors(sensors)
                steering, speed_action_idx = agent.act(current_state, training=True)
                accel, brake = agent.speed_actions[speed_action_idx]
                
                # Clamp steering for safety
                steering = max(-0.8, min(0.8, steering))
                
                # Gear logic
                rpm = current_raw_state.get('rpm', 0)
                current_gear = int(current_raw_state.get('gear', 1))
                if rpm > 5500:  # Lower shift point for safety
                    gear = min(6, current_gear + 1)
                elif rpm < 2000 and current_gear > 1:
                    gear = current_gear - 1
                else:
                    gear = max(1, current_gear)
                
                control = create_control_string(steering, accel, brake, gear)
                sock.sendto(control.encode(), server_address)
                
                # Training
                if prev_state is not None:
                    reward = calculate_safety_reward(prev_state, current_state, prev_raw_state, current_raw_state)
                    episode_reward += reward
                    
                    if step_count % 200 == 0:
                        speed_kmh = current_state[1] * 3.6
                        track_pos = current_state[3]
                        damage = current_raw_state.get('damage', 0)
                        
                        print(f"Step {step_count:4d}: Steer={steering:6.3f} Speed={agent.speed_names[speed_action_idx]:12s} "
                              f"Speed={speed_kmh:6.1f}km/h Pos={track_pos:6.3f} Damage={damage:4.0f} Reward={reward:6.2f}")
                    
                    done = (step_count >= max_steps - 1)
                    agent.remember(prev_state, prev_action, reward, current_state, done)
                    
                    if step_count % 4 == 0:
                        agent.replay()
                    
                    if step_count % 300 == 0:
                        agent.update_target_network()
                
                prev_state = current_state
                prev_raw_state = current_raw_state
                prev_action = (steering, speed_action_idx)
                step_count += 1
                total_damage = current_raw_state.get('damage', 0)
            
            final_distance = current_raw_state.get('distRaced', 0) if 'current_raw_state' in locals() else 0
            print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, Steps={step_count}, "
                  f"Distance={final_distance:.1f}m, Damage={total_damage:.0f}, Epsilon={agent.epsilon:.3f}")
            
            # Save model every 10 episodes
            if (episode + 1) % 10 == 0:
                model_path = f"models/careful_dqn_episode_{episode + 1}.pth"
                torch.save(agent.q_network.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            
            episode += 1
            time.sleep(1)
        
        torch.save(agent.q_network.state_dict(), "models/careful_dqn_final.pth")
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
