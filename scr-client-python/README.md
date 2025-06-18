# SCR Python Client

Python implementation of the SCR (Simulated Car Racing) client with machine learning capabilities.

## Features

- **Rule-based driver** - Python port of the C++ SimpleDriver
- **ML driver framework** - Neural networks and reinforcement learning
- **OpenAI Gym environment** - Standard RL interface
- **Multiple ML algorithms** - Support for DQN, policy networks, etc.
- **Compatible interface** - Same command-line arguments as C++ version

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install with conda
conda install numpy torch gym matplotlib pandas
```

## Quick Start

### 1. Basic Usage (Rule-based Driver)

```bash
# Same interface as C++ version
python client.py host:localhost port:3001 id:SCR maxEpisodes:1 maxSteps:2000 track:aalborg

# Or with explicit arguments
python client.py --host localhost --port 3001 --max-episodes 1 --max-steps 2000
```

### 2. Machine Learning Driver

```bash
# Use neural network driver
DRIVER_TYPE=ml ML_MODEL_TYPE=neural python client.py host:localhost port:3001 id:ML_SCR maxEpisodes:3 maxSteps:1000

# Use DQN with training
DRIVER_TYPE=ml ML_MODEL_TYPE=dqn ML_TRAINING=true ML_MODEL_PATH=models/dqn_model.pth python client.py host:localhost port:3001 id:DQN_SCR maxEpisodes:10 maxSteps:1000
```

### 3. OpenAI Gym Environment

```python
from environments.torcs_env import TORCSEnv

# Create environment
env = TORCSEnv(host="localhost", port=3001, max_steps=1000)

# Standard Gym interface
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # or your agent's action
    obs, reward, done, info = env.step(action)
```

## Project Structure

```
scr-client-python/
├── src/                    # Core framework
│   ├── scr_client.py      # UDP client implementation
│   ├── car_state.py       # Sensor data handling
│   ├── car_control.py     # Control commands
│   ├── base_driver.py     # Driver interface
│   └── simple_parser.py   # Protocol parsing
├── drivers/               # Driver implementations
│   ├── simple_driver.py   # Rule-based driver (C++ port)
│   └── ml_driver.py       # ML driver framework
├── environments/          # RL environments
│   └── torcs_env.py       # OpenAI Gym wrapper
├── examples/              # Example scripts
├── tests/                 # Unit tests
├── client.py              # Main entry point
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Driver Types

### SimpleDriver (Rule-based)
- Exact Python port of the C++ SimpleDriver
- Same gear shifting, steering, and braking logic
- Stuck detection and recovery
- ABS braking system

### MLDriver (Machine Learning)
- **Neural Network**: Direct sensor → action mapping
- **DQN**: Deep Q-Network for discrete actions
- **Training support**: Experience replay, epsilon-greedy exploration
- **Model persistence**: Save/load trained models

## ML Training Examples

### Train DQN Agent

```python
import os
os.environ['DRIVER_TYPE'] = 'ml'
os.environ['ML_MODEL_TYPE'] = 'dqn'
os.environ['ML_TRAINING'] = 'true'
os.environ['ML_MODEL_PATH'] = 'models/dqn_driver.pth'

# Run training
# python client.py host:localhost port:3001 maxEpisodes:50 maxSteps:2000
```

### Use Stable-Baselines3

```python
from environments.torcs_env import TORCSEnv
from stable_baselines3 import PPO

# Create environment
env = TORCSEnv()

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("ppo_torcs_driver")
```

## Configuration

### Environment Variables

- `DRIVER_TYPE`: `simple` (default) or `ml`
- `ML_MODEL_TYPE`: `neural` or `dqn`
- `ML_TRAINING`: `true` to enable training mode
- `ML_MODEL_PATH`: Path to save/load model

### Action Space (ML Driver)

**Continuous (Neural Network)**:
- Steering: [-1, 1] (left to right)
- Acceleration: [0, 1] 
- Braking: [0, 1]

**Discrete (DQN)**:
- 9 predefined actions combining steering, acceleration, and braking

### Observation Space

31-dimensional vector:
- Car dynamics (7): angle, speeds, track position, RPM, gear, damage
- Track sensors (19): distance to track edges
- Opponent sensors (5): closest opponents

## Performance Comparison

| Driver Type | Language | Lines of Code | Training | Performance |
|-------------|----------|---------------|----------|-------------|
| SimpleDriver | C++ | ~300 | No | Baseline |
| SimpleDriver | Python | ~200 | No | Same as C++ |
| Neural Network | Python | ~100 | Yes | Variable |
| DQN | Python | ~150 | Yes | Improves over time |

## Advanced Usage

### Custom Reward Function

```python
def custom_reward(self, current_state, prev_state):
    reward = 0.0
    # Your custom reward logic
    return reward

# Override in MLDriver
driver.calculate_reward = custom_reward
```

### Custom Neural Network

```python
class CustomNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom architecture
        
# Use in MLDriver
driver.model = CustomNN()
```

## Debugging

### Verbose Mode

```bash
# Enable verbose sensor output (same as C++ __UDP_CLIENT_VERBOSE__)
python client.py host:localhost port:3001 --verbose
```

### Monitor Training

```python
# Log to TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/torcs_training')

# In training loop
writer.add_scalar('Reward/Episode', episode_reward, episode)
```

## Compatibility

- **Protocol**: 100% compatible with C++ client
- **Commands**: Same command-line interface
- **Performance**: Comparable to C++ for rule-based driving
- **Server**: Works with existing TORCS SCR server setup

## Known Issues

- First UDP connection may timeout (retry once)
- Gym environment requires server restart between episodes
- DQN training may need hyperparameter tuning for specific tracks

## Contributing

1. Add new drivers in `drivers/` directory
2. Extend base classes in `src/`
3. Add tests in `tests/`
4. Update documentation

## License

Same as original C++ implementation (GPL v2+)