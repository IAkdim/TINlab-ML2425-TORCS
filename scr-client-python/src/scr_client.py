"""
SCR UDP client implementation in Python.
Handles communication with TORCS server via UDP protocol.
"""
import socket
import select
import time
import argparse
from typing import Optional

from .base_driver import BaseDriver, Stage
from .simple_parser import SimpleParser


class SCRClient:
    """UDP client for communicating with TORCS SCR server."""
    
    def __init__(self, host: str = "localhost", port: int = 3001, 
                 client_id: str = "SCR", timeout: float = 1.0):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.server_address = (host, port)
        
    def connect(self) -> bool:
        """
        Establish UDP connection to server.
        
        Returns:
            True if connection successful
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.timeout)
            return True
        except Exception as e:
            print(f"Failed to create socket: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close connection to server."""
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def send_message(self, message: str) -> bool:
        """
        Send message to server.
        
        Args:
            message: Message to send
            
        Returns:
            True if send successful
        """
        if not self.socket:
            return False
            
        try:
            print(f"DEBUG: Sending to {self.server_address}: {message[:50]}...")
            self.socket.sendto(message.encode(), self.server_address)
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    def receive_message(self) -> Optional[str]:
        """
        Receive message from server with timeout.
        
        Returns:
            Received message or None if timeout/error
        """
        if not self.socket:
            return None
            
        try:
            # Use select for timeout handling
            ready = select.select([self.socket], [], [], self.timeout)
            if ready[0]:
                data, addr = self.socket.recvfrom(1000)
                return data.decode().strip()
            else:
                print("** Server did not respond in time")
                return None
        except Exception as e:
            print(f"Failed to receive message: {e}")
            return None
    
    def identify_client(self, driver: BaseDriver) -> bool:
        """
        Identify client to server and initialize connection.
        
        Args:
            driver: Driver instance to get initialization parameters
            
        Returns:
            True if identification successful
        """
        while True:
            # Get sensor angles from driver
            angles = driver.init()
            
            # Create identification message
            init_msg = SimpleParser.stringify_init(self.client_id, angles)
            
            print(f"Sending id to server: {self.client_id}")
            print(f"Sending init string to the server: {init_msg}")
            
            if not self.send_message(init_msg):
                return False
            
            # Wait for identification response
            response = self.receive_message()
            if response:
                print(f"Received: {response}")
                if response == "***identified***":
                    return True
            else:
                print("No response from server during identification")
                return False
    
    def run_episode(self, driver: BaseDriver, max_steps: int = 100000) -> None:
        """
        Run one episode (race) with the given driver.
        
        Args:
            driver: Driver instance
            max_steps: Maximum steps per episode
        """
        current_step = 0
        
        while current_step < max_steps:
            # Receive sensor data
            sensors = self.receive_message()
            if not sensors:
                continue
                
            print(f"Received: {sensors}")
            
            # Check for special server messages
            if sensors == "***shutdown***":
                driver.on_shutdown()
                print("Client Shutdown")
                return
                
            if sensors == "***restart***":
                driver.on_restart()
                print("Client Restart")
                return
            
            # Get driver action
            current_step += 1
            if current_step < max_steps:
                action = driver.drive(sensors)
            else:
                action = "(meta 1)"  # End episode
            
            # Send action to server
            if not self.send_message(action):
                print("Failed to send action")
                return
                
            print(f"Sending {action}")
    
    def run(self, driver: BaseDriver, max_episodes: int = 1, 
            max_steps: int = 100000, track_name: str = "unknown", 
            stage: Stage = Stage.UNKNOWN) -> None:
        """
        Run the client with the given driver.
        
        Args:
            driver: Driver instance
            max_episodes: Maximum number of episodes
            max_steps: Maximum steps per episode
            track_name: Track name
            stage: Race stage
        """
        # Set driver parameters
        driver.track_name = track_name
        driver.stage = stage
        
        # Print configuration
        print("***********************************")
        print(f"HOST: {self.host}")
        print(f"PORT: {self.port}")
        print(f"ID: {self.client_id}")
        print(f"MAX_STEPS: {max_steps}")
        print(f"MAX_EPISODES: {max_episodes}")
        print(f"TRACKNAME: {track_name}")
        print(f"STAGE: {stage.name}")
        print("***********************************")
        
        if not self.connect():
            print("Failed to connect to server")
            return
        
        try:
            shutdown_client = False
            current_episode = 0
            
            while not shutdown_client and current_episode < max_episodes:
                # Identify client for each episode
                if not self.identify_client(driver):
                    print("Failed to identify client")
                    break
                
                # Run episode
                self.run_episode(driver, max_steps)
                current_episode += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            driver.on_shutdown()
            self.disconnect()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SCR Python Client')
    
    parser.add_argument('--host', default='localhost', 
                       help='Server hostname (default: localhost)')
    parser.add_argument('--port', type=int, default=3001,
                       help='Server port (default: 3001)')
    parser.add_argument('--id', default='SCR',
                       help='Client ID (default: SCR)')
    parser.add_argument('--max-episodes', type=int, default=1,
                       help='Maximum episodes (default: 1)')
    parser.add_argument('--max-steps', type=int, default=100000,
                       help='Maximum steps per episode (default: 100000)')
    parser.add_argument('--track', default='unknown',
                       help='Track name (default: unknown)')
    parser.add_argument('--stage', type=int, default=3,
                       help='Race stage: 0=WARMUP, 1=QUALIFYING, 2=RACE, 3=UNKNOWN (default: 3)')
    
    # Parse positional arguments in format key:value
    parser.add_argument('args', nargs='*', help='Arguments in format key:value')
    
    args = parser.parse_args()
    
    # Parse positional arguments
    for arg in args.args:
        if ':' in arg:
            key, value = arg.split(':', 1)
            if key == 'host':
                args.host = value
            elif key == 'port':
                args.port = int(value)
            elif key == 'id':
                args.id = value
            elif key == 'maxEpisodes':
                args.max_episodes = int(value)
            elif key == 'maxSteps':
                args.max_steps = int(value)
            elif key == 'track':
                args.track = value
            elif key == 'stage':
                args.stage = int(value)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Create client
    client = SCRClient(args.host, args.port, args.id)
    
    # Import and create simple driver
    from drivers.simple_driver import SimpleDriver
    driver = SimpleDriver()
    
    # Run client
    stage = Stage(args.stage) if 0 <= args.stage <= 3 else Stage.UNKNOWN
    client.run(driver, args.max_episodes, args.max_steps, args.track, stage)