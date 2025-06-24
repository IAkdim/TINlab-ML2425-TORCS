#!/usr/bin/env python3
"""
Run a trained expert model (supervised learning) for evaluation.
Uses behavioral cloning from expert demonstrations.
"""
import socket
import select
import time
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import joblib


class ExpertDrivingNet(nn.Module):
    """Neural network for imitating expert driving behavior."""
    
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
    
    # Create feature vector matching the training data format:
    # SPEED, TRACK_POSITION, ANGLE_TO_TRACK_AXIS, TRACK_EDGE_0...TRACK_EDGE_17
    
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


def load_expert_model(model_path, scaler_path=None):
    """Load trained expert model and scaler."""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None, None
    
    # Load model
    model = ExpertDrivingNet(input_size=21)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Loaded expert model from {model_path}")
    
    # Load scaler if available
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    elif scaler_path:
        print(f"Warning: Scaler file {scaler_path} not found, proceeding without normalization")
    
    return model, scaler


def run_expert_model(model_path, scaler_path=None, max_steps=8000, verbose=False):
    """Run the trained expert model."""
    
    model, scaler = load_expert_model(model_path, scaler_path)
    if model is None:
        return
    
    # Connection parameters
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (host, port)
    
    print(f"Connecting to {host}:{port}")
    print(f"Running expert model evaluation for {max_steps} steps...")
    print("Expected: Smooth expert-like driving behavior\n")
    
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
                print("Successfully identified! Starting expert evaluation...")
        
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
            features, state_dict = parse_sensors(sensors)
            
            # Normalize features if scaler is available
            if scaler is not None:
                features = scaler.transform(features.reshape(1, -1))[0]
            
            # Get action from expert model
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                actions = model(features_tensor)[0]
                
                acceleration = float(actions[0].item())
                brake = float(actions[1].item())
                steering = float(actions[2].item())
            
            # Simple gear logic (like expert would use)
            rpm = state_dict.get('rpm', 0)
            current_gear = int(state_dict.get('gear', 1))
            if rpm > 6000:
                gear = min(6, current_gear + 1)
            elif rpm < 2500 and current_gear > 1:
                gear = current_gear - 1
            else:
                gear = max(1, current_gear)
            
            # Send control command
            control = create_control_string(acceleration, brake, steering, gear)
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
                
                # Driving quality assessment
                quality = "EXCELLENT"
                if abs(track_pos) > 0.8 or total_damage > 100:
                    quality = "POOR"
                elif abs(track_pos) > 0.5 or abs(angle) > 0.3:
                    quality = "FAIR"
                elif abs(track_pos) > 0.3 or abs(angle) > 0.15:
                    quality = "GOOD"
                
                print(f"Step {step_count:4d}: Accel={acceleration:5.3f} Brake={brake:5.3f} "
                      f"Steer={steering:6.3f} Speed={current_speed_kmh:6.1f}km/h "
                      f"Pos={track_pos:6.3f} Distance={total_distance:7.1f}m "
                      f"Damage={total_damage:4.0f} [{quality}]")
                
                if verbose:
                    print(f"          Angle={angle:6.3f} RPM={rpm:4.0f} Gear={gear}")
                    # Show some track sensor values
                    track_sensors = state_dict.get('track', [])
                    if len(track_sensors) >= 3:
                        print(f"          Track sensors: L={track_sensors[3]:.1f} "
                              f"C={track_sensors[9]:.1f} R={track_sensors[15]:.1f}")
                    print()
            
            step_count += 1
        
        print(f"\n=== Expert Model Evaluation Results ===")
        print(f"Total steps: {step_count}")
        print(f"Total distance: {total_distance:.1f}m")
        print(f"Max speed: {max_speed:.1f}km/h")
        print(f"Final damage: {total_damage:.0f}")
        if step_count > 0:
            avg_speed = (total_distance / (step_count * 0.02)) * 3.6
            print(f"Average speed: {avg_speed:.1f}km/h")
        
        # Performance assessment
        if total_damage < 50 and total_distance > 1000:
            print("🏆 EXCELLENT: Expert-level driving performance!")
        elif total_damage < 200 and total_distance > 500:
            print("🥇 VERY GOOD: High-quality driving with minimal issues")
        elif total_damage < 1000:
            print("🥈 GOOD: Decent driving with some room for improvement")
        else:
            print("🥉 NEEDS WORK: Consider retraining or checking data quality")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained expert driving model")
    parser.add_argument("model_path", nargs="?", default="models/expert_driving_best.pth",
                       help="Path to trained model")
    parser.add_argument("--scaler", default="models/expert_scaler.pkl",
                       help="Path to feature scaler")
    parser.add_argument("--steps", type=int, default=8000,
                       help="Maximum steps to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        print("\nAvailable models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.endswith('.pth') and 'expert' in f:
                    print(f"  {os.path.join(models_dir, f)}")
        return
    
    run_expert_model(args.model_path, args.scaler, args.steps, args.verbose)


if __name__ == "__main__":
    main()
