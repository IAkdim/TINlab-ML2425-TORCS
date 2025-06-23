#!/usr/bin/env python3
"""
Run a trained DQN model for inspection/evaluation.
Usage: python run_dqn_model.py [model_path]
"""
import socket
import select
import time
import numpy as np
import torch
import torch.nn as nn
import sys
import os


class DQN(nn.Module):
    """Simple DQN network."""
    def __init__(self, state_size=29, action_size=9, hidden_size=64):
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
    
    # Add 3 more features to reach 29
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


def create_control_string(steer, accel, brake, gear=1):
    """Create TORCS control string."""
    return f"(accel {accel})(brake {brake})(gear {gear})(steer {steer})(clutch 0.0)"


def load_model(model_path):
    """Load trained DQN model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    model = DQN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


def run_model(model_path, max_steps=5000, verbose=False):
    """Run the trained model."""
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Discrete actions: [steer, accel, brake]
    actions = [
        [-0.5, 0.8, 0.0],  # Turn left + accelerate
        [-0.2, 0.8, 0.0],  # Slight left + accelerate  
        [0.0, 0.8, 0.0],   # Straight + accelerate
        [0.2, 0.8, 0.0],   # Slight right + accelerate
        [0.5, 0.8, 0.0],   # Turn right + accelerate
        [-0.5, 0.0, 0.3],  # Turn left + brake
        [0.0, 0.0, 0.3],   # Straight + brake
        [0.5, 0.0, 0.3],   # Turn right + brake
        [0.0, 0.0, 0.0],   # Coast
    ]
    
    action_names = [
        "Turn left + accel",
        "Slight left + accel", 
        "Straight + accel",
        "Slight right + accel",
        "Turn right + accel",
        "Turn left + brake",
        "Straight + brake", 
        "Turn right + brake",
        "Coast"
    ]
    
    # Connection parameters
    host = "localhost"
    port = 3001
    client_id = "SCR"
    
    # Socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (host, port)
    
    print(f"Connecting to {host}:{port}")
    print(f"Running model evaluation for {max_steps} steps...")
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
                q_values = model(state_tensor)
                action_idx = torch.argmax(q_values).item()
            
            steer, accel, brake = actions[action_idx]
            
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
            control = create_control_string(steer, accel, brake, gear)
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
                damage = state_dict.get('damage', 0)
                
                print(f"Step {step_count:4d}: Action={action_names[action_idx]:20s} "
                      f"Speed={speed:6.1f}km/h TrackPos={track_pos:6.3f} "
                      f"Distance={total_distance:7.1f}m Damage={damage:3.0f}")
                
                if verbose:
                    print(f"          Q-values: {q_values[0].numpy()}")
                    print(f"          State: angle={state_dict.get('angle', 0):.3f} "
                          f"rpm={state_dict.get('rpm', 0):.0f} gear={gear}")
                    track_sensors = state_dict.get('track', [])
                    if track_sensors:
                        min_dist = min(track_sensors)
                        print(f"          Min track distance: {min_dist:.1f}m")
                    print()
            
            step_count += 1
        
        print(f"\nEvaluation completed!")
        print(f"Total steps: {step_count}")
        print(f"Total distance: {total_distance:.1f}m")
        print(f"Average speed: {total_distance/max(step_count*0.02, 1)*3.6:.1f}km/h")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained DQN model for evaluation")
    parser.add_argument("model_path", nargs="?", default="models/dqn_simple_final.pth",
                       help="Path to trained model (default: models/dqn_simple_final.pth)")
    parser.add_argument("--steps", type=int, default=5000,
                       help="Maximum steps to run (default: 5000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output including Q-values")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        print("\nAvailable models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pth'):
                    print(f"  {os.path.join(models_dir, f)}")
        else:
            print("  No models directory found")
        return
    
    run_model(args.model_path, args.steps, args.verbose)


if __name__ == "__main__":
    main()
