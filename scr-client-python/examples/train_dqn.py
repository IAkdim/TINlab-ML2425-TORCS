#!/usr/bin/env python3
"""
Example script for training a DQN agent.
"""
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scr_client import SCRClient, Stage
from drivers.ml_driver import MLDriver


def train_dqn_agent(host="localhost", port=3001, num_episodes=50):
    """Train a DQN agent for the specified number of episodes."""
    
    print(f"Training DQN agent for {num_episodes} episodes...")
    print(f"Connecting to TORCS server at {host}:{port}")
    
    # Create ML driver with DQN
    driver = MLDriver(model_type="dqn", model_path="models/dqn_driver.pth")
    driver.set_training_mode(True)
    
    # Training parameters
    max_steps_per_episode = 2000
    track_name = "aalborg"
    stage = Stage.WARMUP
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Training loop - run all episodes in one client session
    start_time = time.time()
    episode_rewards = []
    
    print(f"Starting training for {num_episodes} episodes...")
    
    # Create client with longer timeout
    client = SCRClient(host, port, "DQN_TRAINER", timeout=10.0)
    
    try:
        # Set up episode tracking
        driver.episode_count = 0
        driver.total_episodes = num_episodes
        
        # Run all episodes at once - TORCS will call on_restart() between episodes
        client.run(driver, max_episodes=num_episodes, max_steps=max_steps_per_episode, 
                  track_name=track_name, stage=stage)
        
        print(f"Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Final model save
    driver.save_model("models/dqn_driver_final.pth")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    return episode_rewards


def evaluate_agent(model_path, host="localhost", port=3001, num_episodes=5):
    """Evaluate a trained agent."""
    
    print(f"Evaluating agent from {model_path}...")
    
    # Create ML driver without training
    driver = MLDriver(model_type="dqn", model_path=model_path)
    driver.set_training_mode(False)  # Disable exploration
    
    # Create client with longer timeout
    client = SCRClient(host, port, "DQN_EVAL", timeout=5.0)
    
    # Evaluation parameters
    max_steps_per_episode = 3000
    track_name = "aalborg"
    stage = Stage.WARMUP
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n=== Evaluation Episode {episode + 1}/{num_episodes} ===")
        
        try:
            client.run(driver, max_episodes=1, max_steps=max_steps_per_episode,
                      track_name=track_name, stage=stage)
            
            episode_rewards.append(driver.episode_reward)
            print(f"Episode {episode + 1}: Reward = {driver.episode_reward:.2f}")
        
        except Exception as e:
            print(f"Error in evaluation episode {episode + 1}: {e}")
            continue
        
        time.sleep(1)
    
    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"\nEvaluation Results:")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Best episode: {max(episode_rewards):.2f}")
        print(f"Worst episode: {min(episode_rewards):.2f}")
    
    return episode_rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or evaluate DQN agent")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                       help="Mode: train or evaluate")
    parser.add_argument("--host", default="localhost", help="TORCS server host")
    parser.add_argument("--port", type=int, default=3001, help="TORCS server port")
    parser.add_argument("--episodes", type=int, default=50, 
                       help="Number of episodes")
    parser.add_argument("--model", default="models/dqn_driver_final.pth",
                       help="Model path for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_dqn_agent(args.host, args.port, args.episodes)
    else:
        evaluate_agent(args.model, args.host, args.port, args.episodes)