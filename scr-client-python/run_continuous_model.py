#!/usr/bin/env python3
"""
Run a trained continuous DQN model for evaluation.
Usage: python run_continuous_model.py [model_path]
"""
import socket
import select
import time
import numpy as np
import torch
import torch.nn as nn
import sys
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
    
    # Create enhanced state vector focusing on steering cues (same as trainer)
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


def create_control_string(steer, accel, brake, gear=1):
    """Create TORCS control string."""
    return f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)"


def load_model(model_path):
    """Load trained continuous DQN model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    model = ContinuousDQN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Loaded continuous model from {model_path}")
    return model


def run_model(model_path, max_steps=5000, verbose=False):
    """Run the trained continuous model."""
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Speed actions: [accel, brake]
    speed_actions = [
        [1.0, 0.0],    # Full accelerate
        [0.0, 0.0],    # Coast
        [0.0, 0.8],    # Brake
    ]
    
    speed_names = ["Accelerate", "Coast", "Brake"]
    
    # Connection parameters
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    # Socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (host, port)
    
    print(f"Connecting to {host}:{port}")
    print(f"Running continuous model evaluation for {max_steps} steps...")
    print("Press Ctrl+C to stop\n")
    
    try:
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
                print("Successfully identified! Starting evaluation...")
            elif response.startswith("("):
                print("Already in game, starting evaluation...")
                identified = True
        else:
            print("No response from server")
        
        if not identified:
            print("Failed to identify to server")
            return
        
        print("Starting evaluation loop...\n")
        
        # Evaluation loop
        step_count = 0
        total_distance = 0
        prev_distance = 0
        
        while step_count < max_steps:
            # Receive sensor data
            ready = select.select([sock], [], [], 5.0)
            if not ready[0]:
                print("No sensor data received, ending evaluation")
                break
            
            data, addr = sock.recvfrom(1000)
            sensors = data.decode().strip()
            
            # Check for special messages
            if sensors == "***shutdown***":
                print("Server shutdown")
                sock.sendto(b"", server_address)
                break
            elif sensors == "***restart***":
                print("Episode restart")
                sock.sendto(b"", server_address)
                break
            
            # Parse state
            state_vector, state_dict = parse_sensors(sensors)
            
            # Get action from model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                steering_out, speed_q = model(state_tensor)
                
                steering = float(steering_out[0].item())
                speed_action_idx = int(torch.argmax(speed_q[0]).item())
            
            # Clamp steering to valid range
            steering = max(-1.0, min(1.0, steering))
            accel, brake = speed_actions[speed_action_idx]
            
            # Simple gear logic
            rpm = state_dict.get('rpm', 0)
            current_gear = int(state_dict.get('gear', 1))
            if rpm > 6000:
                gear = min(6, current_gear + 1)
            elif rpm < 2500 and current_gear > 1:
                gear = current_gear - 1
            else:
                gear = max(1, current_gear)
            
            # Create and send control command
            control = create_control_string(steering, accel, brake, gear)
            sock.sendto(control.encode(), server_address)
            
            # Track progress
            current_distance = state_dict.get('distRaced', 0)
            if current_distance > prev_distance:
                total_distance = current_distance
                prev_distance = current_distance
            
            # Print status every 100 steps
            if verbose or step_count % 100 == 0:
                speed = state_dict.get('speedX', 0) * 3.6  # Convert to km/h
                track_pos = state_dict.get('trackPos', 0)
                angle = state_dict.get('angle', 0)
                damage = state_dict.get('damage', 0)
                
                print(f"Step {step_count:4d}: Steer={steering:6.3f} Speed={speed_names[speed_action_idx]:10s} "
                      f"Speed={speed:6.1f}km/h Pos={track_pos:6.3f} Angle={angle:6.3f} "
                      f"Distance={total_distance:7.1f}m Damage={damage:3.0f}")
                
                if verbose:
                    print(f"          Speed Q-values: {speed_q[0].numpy()}")
                    print(f"          Steering output: {steering:.3f}")
                    track_sensors = state_dict.get('track', [])
                    if track_sensors:
                        min_dist = min(track_sensors)
                        print(f"          Min track distance: {min_dist:.1f}m")
                    print()
            
            step_count += 1
        
        print(f"\nEvaluation completed!")
        print(f"Total steps: {step_count}")
        print(f"Total distance: {total_distance:.1f}m")
        if step_count > 0:
            avg_speed = (total_distance / (step_count * 0.02)) * 3.6  # Convert to km/h
            print(f"Average speed: {avg_speed:.1f}km/h")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained continuous DQN model")
    parser.add_argument("model_path", nargs="?", default="models/continuous_dqn_final.pth",
                       help="Path to trained model (default: models/continuous_dqn_final.pth)")
    parser.add_argument("--steps", type=int, default=5000,
                       help="Maximum steps to run (default: 5000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        print("\nAvailable models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.endswith('.pth'):
                    print(f"  {os.path.join(models_dir, f)}")
        else:
            print("  No models directory found")
        return
    
    run_model(args.model_path, args.steps, args.verbose)


if __name__ == "__main__":
    main()
