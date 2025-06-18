#!/usr/bin/env python3
"""
Example using Stable-Baselines3 with TORCS environment.
Requires: pip install stable-baselines3
"""
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    print("Stable-Baselines3 not available. Install with: pip install stable-baselines3")
    STABLE_BASELINES_AVAILABLE = False

from environments.torcs_env import TORCSEnv


def create_monitored_env(host="localhost", port=3001, log_dir="./logs/"):
    """Create and wrap environment with monitoring."""
    os.makedirs(log_dir, exist_ok=True)
    
    env = TORCSEnv(host=host, port=port, max_steps=1000)
    env = Monitor(env, log_dir)
    
    return env


def train_ppo_agent(total_timesteps=100000, host="localhost", port=3001):
    """Train PPO agent."""
    print("Training PPO agent...")
    
    # Create environment
    env = create_monitored_env(host, port, "./logs/ppo/")
    
    # Check environment
    check_env(env)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Setup callbacks
    eval_env = create_monitored_env(host, port + 1, "./logs/ppo_eval/")  # Different port
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/ppo_best/",
        log_path="./logs/ppo_eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="ppo_torcs"
    )
    
    # Save final model
    model.save("models/ppo_torcs_final")
    env.close()
    eval_env.close()
    
    return model


def train_sac_agent(total_timesteps=100000, host="localhost", port=3001):
    """Train SAC agent (good for continuous control)."""
    print("Training SAC agent...")
    
    # Create environment
    env = create_monitored_env(host, port, "./logs/sac/")
    
    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Setup callbacks
    eval_env = create_monitored_env(host, port + 1, "./logs/sac_eval/")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/sac_best/",
        log_path="./logs/sac_eval/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="sac_torcs"
    )
    
    # Save final model
    model.save("models/sac_torcs_final")
    env.close()
    eval_env.close()
    
    return model


def evaluate_model(model_path, algorithm="ppo", host="localhost", port=3001, n_episodes=10):
    """Evaluate a trained model."""
    print(f"Evaluating {algorithm} model from {model_path}")
    
    # Create environment
    env = TORCSEnv(host=host, port=port, max_steps=2000)
    
    # Load model
    if algorithm.lower() == "ppo":
        model = PPO.load(model_path)
    elif algorithm.lower() == "sac":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}")
    
    env.close()
    
    # Print statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.1f}")
    print(f"Best episode: {max(episode_rewards):.2f}")
    
    return episode_rewards


def hyperparameter_tuning():
    """Example of hyperparameter tuning with Optuna."""
    try:
        import optuna
    except ImportError:
        print("Optuna not available. Install with: pip install optuna")
        return
    
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_int('n_steps', 512, 4096, step=512)
        batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
        
        # Create environment
        env = create_monitored_env("localhost", 3001, f"./logs/optuna_trial_{trial.number}/")
        
        # Create model with suggested hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=0
        )
        
        # Train for shorter time during optimization
        model.learn(total_timesteps=50000)
        
        # Evaluate
        eval_env = TORCSEnv("localhost", 3002, max_steps=1000)
        episode_rewards = []
        
        for _ in range(5):  # Short evaluation
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        env.close()
        eval_env.close()
        
        return np.mean(episode_rewards)
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print("Best hyperparameters:")
    print(study.best_params)


if __name__ == "__main__":
    if not STABLE_BASELINES_AVAILABLE:
        sys.exit(1)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Stable-Baselines3 TORCS training")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo",
                       help="RL algorithm to use")
    parser.add_argument("--mode", choices=["train", "eval", "tune"], default="train",
                       help="Mode: train, evaluate, or hyperparameter tuning")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Training timesteps")
    parser.add_argument("--model", help="Model path for evaluation")
    parser.add_argument("--host", default="localhost", help="TORCS server host")
    parser.add_argument("--port", type=int, default=3001, help="TORCS server port")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.mode == "train":
        if args.algorithm == "ppo":
            model = train_ppo_agent(args.timesteps, args.host, args.port)
        elif args.algorithm == "sac":
            model = train_sac_agent(args.timesteps, args.host, args.port)
    
    elif args.mode == "eval":
        if not args.model:
            args.model = f"models/{args.algorithm}_torcs_final"
        evaluate_model(args.model, args.algorithm, args.host, args.port)
    
    elif args.mode == "tune":
        hyperparameter_tuning()
    
    print("\nTo view training progress:")
    print("tensorboard --logdir=./tensorboard_logs/")