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
    
    # Create client
    client = SCRClient(host, port, "DQN_TRAINER")
    
    # Training parameters
    max_steps_per_episode = 2000
    track_name = "aalborg"
    stage = Stage.PRACTICE
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Training loop
    start_time = time.time()
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        # Run single episode
        try:
            client.run(driver, max_episodes=1, max_steps=max_steps_per_episode, 
                      track_name=track_name, stage=stage)
            
            # Track episode reward (basic implementation)
            episode_rewards.append(driver.episode_reward)
            
            # Print progress
            if len(episode_rewards) >= 10:
                avg_reward = sum(episode_rewards[-10:]) / 10
                print(f"Episode {episode + 1}: Reward = {driver.episode_reward:.2f}, "
                      f"Avg (last 10) = {avg_reward:.2f}")
            else:
                print(f"Episode {episode + 1}: Reward = {driver.episode_reward:.2f}")
            
            # Save model periodically
            if (episode + 1) % 10 == 0:
                model_path = f"models/dqn_driver_episode_{episode + 1}.pth"
                driver.save_model(model_path)
                print(f"Model saved to {model_path}")
        
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            continue
        
        # Brief pause between episodes
        time.sleep(1)
    
    # Final model save
    driver.save_model("models/dqn_driver_final.pth")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    
    return episode_rewards


def evaluate_agent(model_path, host="localhost", port=3001, num_episodes=5):
    """Evaluate a trained agent."""
    
    print(f"Evaluating agent from {model_path}...")
    
    # Create ML driver without training
    driver = MLDriver(model_type="dqn", model_path=model_path)
    driver.set_training_mode(False)  # Disable exploration
    
    # Create client
    client = SCRClient(host, port, "DQN_EVAL")
    
    # Evaluation parameters
    max_steps_per_episode = 3000
    track_name = "aalborg"
    stage = Stage.RACE
    
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