"""
Machine Learning driver implementation.
Supports various ML approaches including reinforcement learning.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pickle
import os
from typing import List, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.base_driver import BaseDriver
from src.car_state import CarState
from src.car_control import CarControl


class SimpleNN(nn.Module):
    """Simple neural network for racing control."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 3):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Deep Q-Network agent for discrete action spaces."""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
    
    def _build_model(self) -> nn.Module:
        """Build neural network model."""
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
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
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


class MLDriver(BaseDriver):
    """Machine learning driver with multiple algorithm support."""
    
    def __init__(self, model_type: str = "neural", model_path: Optional[str] = None):
        super().__init__()
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.training = False
        
        # Experience tracking
        self.prev_state = None
        self.prev_action = None
        self.episode_reward = 0
        self.step_count = 0
        
        # Action discretization for DQN
        self.discrete_actions = [
            # [steer, accel, brake]
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
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == "neural":
            input_size = 31  # Size of state vector from CarState.to_vector()
            self.model = SimpleNN(input_size, 64, 3)  # [steer, accel, brake]
            
        elif self.model_type == "dqn":
            input_size = 31
            action_size = len(self.discrete_actions)
            self.model = DQNAgent(input_size, action_size)
            
        # Load pre-trained model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
    
    def _get_state_vector(self, cs: CarState) -> np.ndarray:
        """Convert CarState to feature vector."""
        return np.array(cs.to_vector(), dtype=np.float32)
    
    def _calculate_reward(self, cs: CarState, prev_cs: Optional[CarState] = None) -> float:
        """Calculate reward for reinforcement learning."""
        reward = 0.0
        
        # Speed reward (encourage going fast)
        speed_reward = min(cs.get_speed_x() / 100.0, 1.0)
        reward += speed_reward * 0.1
        
        # Track position penalty (stay on track)
        track_pos = abs(cs.get_track_pos())
        if track_pos > 1.0:  # Off track
            reward -= 2.0
        else:
            reward += (1.0 - track_pos) * 0.1
        
        # Progress reward
        if prev_cs:
            distance_progress = cs.get_dist_raced() - prev_cs.get_dist_raced()
            reward += distance_progress * 0.01
        
        # Damage penalty
        reward -= cs.get_damage() * 0.001
        
        # Track sensor reward (avoid walls)
        min_track_dist = min(cs.track)
        if min_track_dist < 1.0:
            reward -= (1.0 - min_track_dist) * 0.5
        
        return reward
    
    def _neural_inference(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Get action from neural network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            output = self.model(state_tensor)
            steer, accel, brake = output[0].numpy()
            
            # Ensure brake and accel are non-negative
            accel = max(0.0, accel)
            brake = max(0.0, brake)
            
            return float(steer), float(accel), float(brake)
    
    def _dqn_inference(self, state: np.ndarray, training: bool = False) -> Tuple[float, float, float]:
        """Get action from DQN agent."""
        action_idx = self.model.act(state, training)
        steer, accel, brake = self.discrete_actions[action_idx]
        return steer, accel, brake
    
    def set_training_mode(self, training: bool = True):
        """Enable/disable training mode."""
        self.training = training
    
    def save_model(self, path: str):
        """Save model to disk."""
        if self.model_type == "neural":
            torch.save(self.model.state_dict(), path)
        elif self.model_type == "dqn":
            torch.save({
                'q_network': self.model.q_network.state_dict(),
                'target_network': self.model.target_network.state_dict(),
                'epsilon': self.model.epsilon
            }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        if self.model_type == "neural":
            self.model.load_state_dict(torch.load(path))
        elif self.model_type == "dqn":
            checkpoint = torch.load(path)
            self.model.q_network.load_state_dict(checkpoint['q_network'])
            self.model.target_network.load_state_dict(checkpoint['target_network'])
            self.model.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
    
    def drive(self, sensors: str) -> str:
        """Main driving function with ML inference."""
        cs = self._parse_state(sensors)
        state = self._get_state_vector(cs)
        
        # Get action from model
        if self.model_type == "neural":
            steer, accel, brake = self._neural_inference(state)
        elif self.model_type == "dqn":
            steer, accel, brake = self._dqn_inference(state, self.training)
        else:
            # Fallback to simple behavior
            steer, accel, brake = 0.0, 0.5, 0.0
        
        # Simple gear logic (can be improved)
        if cs.get_rpm() > 6000:
            gear = min(6, cs.get_gear() + 1)
        elif cs.get_rpm() < 2500 and cs.get_gear() > 1:
            gear = cs.get_gear() - 1
        else:
            gear = max(1, cs.get_gear())
        
        # Training logic for RL
        if self.training and self.model_type == "dqn":
            reward = self._calculate_reward(cs, self.prev_state)
            self.episode_reward += reward
            
            # Store experience
            if self.prev_state is not None and self.prev_action is not None:
                prev_state_vector = self._get_state_vector(self.prev_state)
                self.model.remember(prev_state_vector, self.prev_action, 
                                  reward, state, False)
                
                # Train periodically
                if self.step_count % 4 == 0:
                    self.model.replay()
                
                # Update target network
                if self.step_count % 100 == 0:
                    self.model.update_target_network()
            
            self.prev_state = cs
            if self.model_type == "dqn":
                self.prev_action = self.model.act(state, True)
        
        self.step_count += 1
        
        # Create control command
        control = CarControl(accel, brake, gear, steer, 0.0)
        return control.to_string()
    
    def on_restart(self):
        """Called when episode restarts."""
        if self.training:
            print(f"Episode finished. Reward: {self.episode_reward:.2f}, Steps: {self.step_count}")
            self.episode_reward = 0
            self.step_count = 0
            self.prev_state = None
            self.prev_action = None
        print("ML Driver restarting!")
    
    def on_shutdown(self):
        """Called when shutting down."""
        if self.training and self.model_path:
            self.save_model(self.model_path)
        print("ML Driver bye bye!")