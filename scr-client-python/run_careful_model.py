#!/usr/bin/env python3
"""
Run a trained careful DQN model for evaluation.
Shows safe, controlled driving behavior.
"""
import socket
import select
import time
import numpy as np
import torch
import torch.nn as nn
import sys
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
    
    # Create enhanced state vector (same as trainer)
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


def create_control_string(steer, accel, brake, gear=1):
    """Create TORCS control string."""
    return f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)"


def load_model(model_path):
    """Load trained careful DQN model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    model = CarefulDQN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Loaded careful model from {model_path}")
    return model


def run_model(model_path, max_steps=8000, verbose=False):
    """Run the trained careful model."""
    
    model = load_model(model_path)
    if model is None:
        return
    
    # Conservative speed actions
    speed_actions = [
        [0.3, 0.0],    # Light accelerate (30%)
        [0.6, 0.0],    # Medium accelerate (60%)
        [0.0, 0.0],    # Coast
        [0.0, 0.3],    # Light brake
        [0.0, 0.8],    # Hard brake
    ]
    
    speed_names = ["Light Accel", "Medium Accel", "Coast", "Light Brake", "Hard Brake"]
    
    # Connection parameters
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (host, port)
    
    print(f"Connecting to {host}:{port}")
    print(f"Running careful model evaluation for {max_steps} steps...")
    print("Expected: Conservative speed, smooth steering, low damage\n")
    
    try:
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
                print("Successfully identified! Starting careful evaluation...")
        
        if not identified:
            print("Failed to identify to server")
            return
        
        print("Starting evaluation loop...\n")
        
        # Evaluation loop
        step_count = 0
        total_distance = 0
        prev_distance = 0
        max_speed = 0
        total_damage = 0
        
        while step_count < max_steps:
            ready = select.select([sock], [], [], 5.0)
            if not ready[0]:
                print("No sensor data received, ending evaluation")
                break
            
            data, addr = sock.recvfrom(1000)
            sensors = data.decode().strip()
            
            if sensors == "***shutdown***":
                print("Server shutdown")
                break
            elif sensors == "***restart***":
                print("Episode restart")
                break
            
            # Parse state
            state_vector, state_dict = parse_sensors(sensors)
            
            # Get action from model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                steering_out, speed_q = model(state_tensor)
                
                steering = float(steering_out[0].item())
                speed_action_idx = int(torch.argmax(speed_q[0]).item())
            
            # Clamp steering for safety
            steering = max(-0.8, min(0.8, steering))
            accel, brake = speed_actions[speed_action_idx]
            
            # Conservative gear logic
            rpm = state_dict.get('rpm', 0)
            current_gear = int(state_dict.get('gear', 1))
            if rpm > 5500:  # Lower shift point
                gear = min(6, current_gear + 1)
            elif rpm < 2000 and current_gear > 1:
                gear = current_gear - 1
            else:
                gear = max(1, current_gear)
            
            control = create_control_string(steering, accel, brake, gear)
            sock.sendto(control.encode(), server_address)
            
            # Track progress
            current_distance = state_dict.get('distRaced', 0)
            if current_distance > prev_distance:
                total_distance = current_distance
                prev_distance = current_distance
            
            current_speed_kmh = state_dict.get('speedX', 0) * 3.6
            max_speed = max(max_speed, current_speed_kmh)
            total_damage = state_dict.get('damage', 0)
            
            # Print status
            if verbose or step_count % 200 == 0:
                track_pos = state_dict.get('trackPos', 0)
                angle = state_dict.get('angle', 0)
                
                # Safety status
                safety_status = "SAFE"
                if abs(track_pos) > 0.8:
                    safety_status = "DANGER"
                elif abs(track_pos) > 0.5 or abs(angle) > 0.3:
                    safety_status = "CAUTION"
                
                print(f"Step {step_count:4d}: Steer={steering:6.3f} Speed={speed_names[speed_action_idx]:12s} "
                      f"Speed={current_speed_kmh:6.1f}km/h Pos={track_pos:6.3f} "
                      f"Distance={total_distance:7.1f}m Damage={total_damage:4.0f} [{safety_status}]")
                
                if verbose:
                    print(f"          Angle={angle:6.3f} Front={state_vector[7]:.2f} "
                          f"Left={state_vector[8]:.2f} Right={state_vector[9]:.2f}")
                    print(f"          Speed Q-values: {speed_q[0].numpy()}")
                    print()
            
            step_count += 1
        
        print(f"\n=== Careful Driving Evaluation Results ===")
        print(f"Total steps: {step_count}")
        print(f"Total distance: {total_distance:.1f}m")
        print(f"Max speed: {max_speed:.1f}km/h")
        print(f"Final damage: {total_damage:.0f}")
        print(f"Avg speed: {(total_distance/(step_count*0.02))*3.6:.1f}km/h" if step_count > 0 else "N/A")
        
        # Safety assessment
        if total_damage < 100:
            print("🟢 EXCELLENT: Very safe driving, minimal damage")
        elif total_damage < 500:
            print("🟡 GOOD: Mostly safe driving, minor incidents")
        elif total_damage < 2000:
            print("🟠 FAIR: Some crashes, needs improvement")
        else:
            print("🔴 POOR: Frequent crashes, unsafe driving")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained careful DQN model")
    parser.add_argument("model_path", nargs="?", default="models/careful_dqn_final.pth",
                       help="Path to trained model")
    parser.add_argument("--steps", type=int, default=8000,
                       help="Maximum steps to run (default: 8000)")
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
        return
    
    run_model(args.model_path, args.steps, args.verbose)


if __name__ == "__main__":
    main()
