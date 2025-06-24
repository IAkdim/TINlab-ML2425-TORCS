# SCR Python Client

Python implementation of the SCR (Simulated Car Racing) client with advanced machine learning capabilities.

## Features

- **Rule-based driver** - Python port of the C++ SimpleDriver
- **Supervised learning** - Train from expert demonstrations (RECOMMENDED)
- **Reinforcement learning** - DQN with continuous/discrete control
- **Multiple training approaches** - From careful driving to aggressive racing
- **Expert data included** - CSV files from last year's winning bot
- **OpenAI Gym environment** - Standard RL interface
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
├── src/                         # Core framework
│   ├── scr_client.py           # UDP client implementation
│   ├── car_state.py            # Sensor data handling
│   ├── car_control.py          # Control commands
│   ├── base_driver.py          # Driver interface
│   └── simple_parser.py        # Protocol parsing
├── drivers/                     # Driver implementations
│   ├── simple_driver.py        # Rule-based driver (C++ port)
│   └── ml_driver.py            # ML driver framework
├── environments/                # RL environments
│   └── torcs_env.py            # OpenAI Gym wrapper
├── trainers/                    # ML training scripts
│   ├── supervised_trainer.py   # Behavioral cloning from expert data
│   ├── simple_dqn_trainer.py   # Basic DQN implementation
│   ├── continuous_dqn_trainer.py # Hybrid continuous control
│   └── careful_dqn_trainer.py  # Conservative speed control
├── evaluators/                  # Model evaluation scripts
│   ├── run_expert_model.py     # Run supervised learning models
│   ├── run_dqn_model.py        # Run DQN models
│   ├── run_continuous_model.py # Run continuous control models
│   └── run_careful_model.py    # Run careful DQN models
├── train_data/                  # Expert demonstration data
│   └── train_data/             # CSV files from last year's winning bot
├── models/                      # Saved model files
├── client.py                   # Main entry point
├── requirements.txt            # Dependencies
└── README.md                  # This file
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

## Training Options

### 1. Supervised Learning (RECOMMENDED)

Train from expert demonstrations using behavioral cloning:

```bash
# Train from expert CSV data
python supervised_trainer.py

# Run trained expert model
python run_expert_model.py models/expert_driving_best.pth
```

**Advantages:**
- Uses proven winning strategies from last year's champion
- Fast training (~200 epochs)
- Stable, expert-level performance
- Includes data from multiple tracks

### 2. Reinforcement Learning (Advanced)

Train using various DQN approaches:

```bash
# Basic DQN with discrete actions
python simple_dqn_trainer.py

# Continuous control (hybrid steering + discrete speed)
python continuous_dqn_trainer.py

# Conservative driving (safety-first)
python careful_dqn_trainer.py
```

**Note:** RL training requires significant time and hyperparameter tuning.

### 3. Evaluation Scripts

Test any trained model:

```bash
# Run expert model (supervised learning)
python run_expert_model.py models/expert_driving_best.pth --verbose

# Run DQN models
python run_dqn_model.py models/dqn_model.pth
python run_continuous_model.py models/continuous_dqn_model.pth
python run_careful_model.py models/careful_dqn_model.pth
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

**Expert Model (Supervised Learning)**:
21-dimensional vector:
- Speed, track position, angle to track axis
- 18 track edge sensors (distance to track boundaries)

**DQN Models**:
29-35 dimensional vectors (varies by trainer):
- Car dynamics: angle, speeds, track position, RPM, gear, damage
- Track sensors: 19 distance measurements to track edges
- Additional features: steering cues, velocity vectors (continuous models)

## Performance Comparison

| Approach | Training Time | Sample Efficiency | Performance | Stability |
|----------|---------------|-------------------|-------------|-----------|
| **Expert (Supervised)** | ~30 min | ⭐⭐⭐⭐⭐ | Expert-level | ⭐⭐⭐⭐⭐ |
| Careful DQN | ~2-4 hours | ⭐⭐⭐ | Conservative | ⭐⭐⭐⭐ |
| Continuous DQN | ~4-8 hours | ⭐⭐ | Variable | ⭐⭐⭐ |
| Simple DQN | ~6-12 hours | ⭐ | Unstable | ⭐⭐ |
| Rule-based | None | N/A | Baseline | ⭐⭐⭐⭐⭐ |

**Recommendation**: Start with supervised learning using expert data for best results.

## Expert Data Format

The included training data (`train_data/train_data/*.csv`) contains:

**Actions (3 columns):**
- `ACCELERATION`: [0, 1] - Throttle input
- `BRAKE`: [0, 1] - Brake input  
- `STEERING`: [-1, 1] - Steering angle (left negative, right positive)

**Sensors (21 columns):**
- `SPEED`: Current speed (m/s)
- `TRACK_POSITION`: Position relative to track center [-1, 1]
- `ANGLE_TO_TRACK_AXIS`: Car orientation relative to track direction
- `TRACK_EDGE_0` to `TRACK_EDGE_17`: Distance to track boundaries (18 sensors)

Data extracted from last year's winning bot across multiple tracks.
Video reference: https://www.youtube.com/watch?v=pX-UDQdtmBM

## Advanced Customization

### Creating Custom Trainers

```python
# Extend base trainer classes
from trainers.supervised_trainer import ExpertDrivingNet

class CustomTrainer:
    def __init__(self):
        self.model = ExpertDrivingNet(input_size=21, hidden_size=512)
        # Your custom training logic
```

### Model Architecture Options

All trainers support customizable neural networks with different:
- Hidden layer sizes (128, 256, 512)
- Dropout rates (0.1, 0.2, 0.3)
- Activation functions (ReLU, LeakyReLU, Swish)
- Output heads (separate vs. shared)

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
- TORCS server may require restart between long training sessions
- DQN models require significant hyperparameter tuning for optimal performance
- Sensor angles must match exactly: `[-90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 75, 90]`

## Getting Started

**Quick Start (Recommended):**
1. Train expert model: `python supervised_trainer.py`
2. Test performance: `python run_expert_model.py models/expert_driving_best.pth --verbose`
3. Compare with rule-based: `python client.py host:localhost port:3001`

**For Research/Experimentation:**
1. Try different DQN approaches in `trainers/` directory
2. Modify reward functions and network architectures
3. Collect your own training data using successful runs

## Contributing

1. Add new drivers in `drivers/` directory
2. Extend base classes in `src/`
3. Add tests in `tests/`
4. Update documentation

## License

Same as original C++ implementation (GPL v2+)