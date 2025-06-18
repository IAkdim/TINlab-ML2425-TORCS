#!/usr/bin/env python3
"""
Main client script for SCR Python implementation.
Compatible with the C++ version's command line interface.
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scr_client import SCRClient, parse_args, Stage


def main():
    """Main entry point for the client."""
    args = parse_args()
    
    # Create client
    client = SCRClient(args.host, args.port, args.id)
    
    # Determine driver type from arguments or environment
    driver_type = os.environ.get('DRIVER_TYPE', 'simple')
    
    # Import and create driver
    if driver_type == 'ml':
        from drivers.ml_driver import MLDriver
        model_type = os.environ.get('ML_MODEL_TYPE', 'neural')
        model_path = os.environ.get('ML_MODEL_PATH', None)
        driver = MLDriver(model_type, model_path)
        
        # Enable training if requested
        if os.environ.get('ML_TRAINING', '').lower() == 'true':
            driver.set_training_mode(True)
            print("ML Driver training mode enabled")
    else:
        from drivers.simple_driver import SimpleDriver
        driver = SimpleDriver()
    
    # Run client
    stage = Stage(args.stage) if 0 <= args.stage <= 3 else Stage.UNKNOWN
    client.run(driver, args.max_episodes, args.max_steps, args.track, stage)


if __name__ == "__main__":
    main()