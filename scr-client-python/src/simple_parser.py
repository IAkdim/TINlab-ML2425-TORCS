"""
Simple parser for SCR protocol messages.
Handles parsing of sensor data and formatting of control commands.
"""
import re
from typing import Dict, Any, List


class SimpleParser:
    """Parser for SCR protocol messages in format: (key value)(key value)..."""
    
    @staticmethod
    def parse(message: str) -> Dict[str, Any]:
        """
        Parse a sensor message string into a dictionary.
        
        Args:
            message: String in format "(angle 0.003)(speedX 45.2)(track 7.4866 7.79257 ...)..."
            
        Returns:
            Dictionary with parsed sensor values
        """
        data = {}
        
        # Find all (key value) patterns
        pattern = r'\(([^)]+)\)'
        matches = re.findall(pattern, message)
        
        for match in matches:
            parts = match.split()
            if len(parts) >= 2:
                key = parts[0]
                values = parts[1:]
                
                # Handle single values vs arrays
                if len(values) == 1:
                    try:
                        # Try to parse as float, fallback to string
                        data[key] = float(values[0])
                    except ValueError:
                        data[key] = values[0]
                else:
                    # Multiple values - convert to list of floats
                    try:
                        data[key] = [float(v) for v in values]
                    except ValueError:
                        data[key] = values
        
        return data
    
    @staticmethod
    def stringify_init(client_id: str, angles: List[float]) -> str:
        """
        Create initialization string for client identification.
        
        Args:
            client_id: Client identifier
            angles: List of track sensor angles
            
        Returns:
            Formatted initialization string
        """
        angle_str = ' '.join([str(angle) for angle in angles])
        return f"{client_id}(init {angle_str})"
    
    @staticmethod
    def stringify_control(accel: float, brake: float, gear: int, 
                         steer: float, clutch: float, focus: int = 0, 
                         meta: int = 0) -> str:
        """
        Create control command string.
        
        Args:
            accel: Acceleration [0,1]
            brake: Brake [0,1] 
            gear: Gear [-1,0,1,2,3,4,5,6]
            steer: Steering [-1,1]
            clutch: Clutch [0,1]
            focus: Focus angle [-90,90]
            meta: Meta command [0,1]
            
        Returns:
            Formatted control string
        """
        return (f"(accel {accel:.6f})(brake {brake:.6f})(gear {gear})"
                f"(steer {steer:.6f})(clutch {clutch:.6f})(focus {focus})"
                f"(meta {meta})")