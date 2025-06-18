"""
OpenAI Gym environment wrapper for TORCS SCR.
Provides standard RL interface for training agents.
"""
import gym
from gym import spaces
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scr_client import SCRClient
from src.base_driver import Stage
from src.car_state import CarState
from src.car_control import CarControl


class TORCSEnv(gym.Env):
    """
    OpenAI Gym environment for TORCS racing simulation.
    
    Observation Space: 31-dimensional vector with:
    - Car state (angle, speeds, track position, etc.)
    - Track sensors (19 distance measurements)
    - Opponent sensors (5 closest)
    
    Action Space: 3-dimensional continuous control:
    - Steering: [-1, 1]
    - Acceleration: [0, 1] 
    - Braking: [0, 1]
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, host: str = "localhost", port: int = 3001,
                 client_id: str = "GYM", track_name: str = "aalborg",
                 max_steps: int = 1000, timeout: float = 1.0):
        super(TORCSEnv, self).__init__()
        
        # Environment parameters
        self.host = host
        self.port = port
        self.client_id = client_id
        self.track_name = track_name
        self.max_steps = max_steps
        self.timeout = timeout
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # State tracking
        self.client: Optional[SCRClient] = None
        self.current_state: Optional[CarState] = None
        self.step_count = 0
        self.episode_reward = 0
        self.done = False
        self.last_dist_raced = 0
        self.stuck_count = 0
        
        # Performance tracking
        self.episode_times = []
        self.episode_rewards = []
        
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.step_count = 0
        self.episode_reward = 0
        self.done = False
        self.last_dist_raced = 0
        self.stuck_count = 0
        
        # Disconnect previous client if exists
        if self.client:
            self.client.disconnect()
        
        # Create new client connection
        self.client = SCRClient(self.host, self.port, self.client_id, self.timeout)
        
        if not self.client.connect():
            raise ConnectionError("Failed to connect to TORCS server")
        
        # Initialize with dummy driver for identification
        dummy_driver = DummyDriver()
        if not self.client.identify_client(dummy_driver):
            raise ConnectionError("Failed to identify client with server")
        
        # Get initial state
        sensors = self.client.receive_message()
        if not sensors or sensors in ["***shutdown***", "***restart***"]:
            raise RuntimeError("Failed to receive initial state")
        
        self.current_state = CarState(sensors)
        self.last_dist_raced = self.current_state.get_dist_raced()
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if self.done:
            raise RuntimeError("Episode finished. Call reset() to start new episode.")
        
        # Convert action to control command
        steer, accel, brake = action
        steer = np.clip(steer, -1.0, 1.0)
        accel = np.clip(accel, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        
        # Simple gear logic
        if self.current_state.get_rpm() > 6000:
            gear = min(6, self.current_state.get_gear() + 1)
        elif self.current_state.get_rpm() < 2500 and self.current_state.get_gear() > 1:
            gear = self.current_state.get_gear() - 1
        else:
            gear = max(1, self.current_state.get_gear())
        
        # Create and send control command
        control = CarControl(accel, brake, gear, steer, 0.0)
        
        if not self.client.send_message(control.to_string()):
            self.done = True
            return self._get_observation(), -10.0, True, {"error": "Failed to send action"}
        
        # Receive response
        sensors = self.client.receive_message()
        if not sensors:
            self.done = True
            return self._get_observation(), -10.0, True, {"error": "No response from server"}
        
        # Check for episode end signals
        if sensors == "***shutdown***":
            self.done = True
            return self._get_observation(), 0.0, True, {"reason": "Server shutdown"}
        
        if sensors == "***restart***":
            self.done = True
            return self._get_observation(), 0.0, True, {"reason": "Server restart"}
        
        # Update state
        prev_state = self.current_state
        self.current_state = CarState(sensors)
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(self.current_state, prev_state)
        self.episode_reward += reward
        
        # Check termination conditions
        self.done = self._check_done()
        
        # Prepare info dict
        info = {
            "step": self.step_count,
            "episode_reward": self.episode_reward,
            "distance_raced": self.current_state.get_dist_raced(),
            "speed": self.current_state.get_speed_x(),
            "track_pos": self.current_state.get_track_pos(),
            "damage": self.current_state.get_damage()
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation vector."""
        if self.current_state is None:
            return np.zeros(31, dtype=np.float32)
        return np.array(self.current_state.to_vector(), dtype=np.float32)
    
    def _calculate_reward(self, current_state: CarState, prev_state: CarState) -> float:
        """Calculate reward for current step."""
        reward = 0.0
        
        # Progress reward - encourage forward movement
        dist_progress = current_state.get_dist_raced() - self.last_dist_raced
        reward += dist_progress * 0.01
        self.last_dist_raced = current_state.get_dist_raced()
        
        # Speed reward - encourage high speed
        speed_reward = min(current_state.get_speed_x() / 100.0, 1.0)
        reward += speed_reward * 0.1
        
        # Track position penalty - stay on track
        track_pos = abs(current_state.get_track_pos())
        if track_pos > 1.0:  # Off track
            reward -= 5.0
            self.stuck_count += 1
        else:
            reward += (1.0 - track_pos) * 0.1
            self.stuck_count = 0
        
        # Damage penalty
        if current_state.get_damage() > prev_state.get_damage():
            reward -= (current_state.get_damage() - prev_state.get_damage()) * 0.01
        
        # Wall avoidance reward
        min_track_dist = min(current_state.track)
        if min_track_dist < 2.0:
            reward -= (2.0 - min_track_dist) * 0.5
        
        # Stuck penalty
        if abs(current_state.get_angle()) > 1.0:  # Large angle indicates stuck
            reward -= 1.0
            self.stuck_count += 1
        
        return reward
    
    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # Too much damage
        if self.current_state.get_damage() > 5000:
            return True
        
        # Stuck for too long
        if self.stuck_count > 100:
            return True
        
        # Very slow for extended period (could indicate stuck)
        if self.current_state.get_speed_x() < 1.0 and self.step_count > 50:
            return True
        
        return False
    
    def close(self):
        """Clean up environment."""
        if self.client:
            self.client.disconnect()
    
    def render(self, mode='human'):
        """Render environment (TORCS handles visualization)."""
        if mode == 'human':
            if self.current_state:
                print(f"Step: {self.step_count}, "
                      f"Speed: {self.current_state.get_speed_x():.1f}, "
                      f"Position: {self.current_state.get_track_pos():.3f}, "
                      f"Reward: {self.episode_reward:.2f}")
    
    def seed(self, seed=None):
        """Set random seed."""
        np.random.seed(seed)
        return [seed]


class DummyDriver:
    """Minimal driver for environment initialization."""
    
    def __init__(self):
        self.track_name = "unknown"
        self.stage = Stage.UNKNOWN
    
    def init(self):
        """Return default sensor angles."""
        return [-90 + i * 10 for i in range(19)]


# Example usage and training script
if __name__ == "__main__":
    # Create environment
    env = TORCSEnv(host="localhost", port=3001, max_steps=1000)
    
    # Simple random agent test
    print("Testing TORCS Gym environment with random agent...")
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if env.step_count % 100 == 0:
            env.render()
    
    print(f"Episode finished. Total reward: {total_reward:.2f}")
    env.close()