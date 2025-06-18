"""
Car control representation for SCR client.
Contains control commands to send to the TORCS server.
"""
from .simple_parser import SimpleParser


class CarControl:
    """Represents control commands for the car."""
    
    def __init__(self, accel: float = 0.0, brake: float = 0.0, gear: int = 1,
                 steer: float = 0.0, clutch: float = 0.0, focus: int = 0, meta: int = 0):
        self.accel = max(0.0, min(1.0, accel))  # Clamp to [0,1]
        self.brake = max(0.0, min(1.0, brake))  # Clamp to [0,1]
        self.gear = max(-1, min(6, gear))       # Clamp to [-1,6]
        self.steer = max(-1.0, min(1.0, steer)) # Clamp to [-1,1]
        self.clutch = max(0.0, min(1.0, clutch)) # Clamp to [0,1]
        self.focus = max(-90, min(90, focus))   # Clamp to [-90,90]
        self.meta = meta
    
    # Getter methods for compatibility
    def get_accel(self) -> float:
        return self.accel
    
    def get_brake(self) -> float:
        return self.brake
    
    def get_gear(self) -> int:
        return self.gear
    
    def get_steer(self) -> float:
        return self.steer
    
    def get_clutch(self) -> float:
        return self.clutch
    
    def get_focus(self) -> int:
        return self.focus
    
    def get_meta(self) -> int:
        return self.meta
    
    # Setter methods
    def set_accel(self, accel: float) -> None:
        self.accel = max(0.0, min(1.0, accel))
    
    def set_brake(self, brake: float) -> None:
        self.brake = max(0.0, min(1.0, brake))
    
    def set_gear(self, gear: int) -> None:
        self.gear = max(-1, min(6, gear))
    
    def set_steer(self, steer: float) -> None:
        self.steer = max(-1.0, min(1.0, steer))
    
    def set_clutch(self, clutch: float) -> None:
        self.clutch = max(0.0, min(1.0, clutch))
    
    def set_focus(self, focus: int) -> None:
        self.focus = max(-90, min(90, focus))
    
    def set_meta(self, meta: int) -> None:
        self.meta = meta
    
    def to_string(self) -> str:
        """Convert to SCR protocol string format."""
        return SimpleParser.stringify_control(
            self.accel, self.brake, self.gear, 
            self.steer, self.clutch, self.focus, self.meta
        )
    
    def from_string(self, control_str: str) -> None:
        """Parse control string and update values."""
        data = SimpleParser.parse(control_str)
        
        self.accel = data.get('accel', self.accel)
        self.brake = data.get('brake', self.brake)
        self.gear = int(data.get('gear', self.gear))
        self.steer = data.get('steer', self.steer)
        self.clutch = data.get('clutch', self.clutch)
        self.focus = int(data.get('focus', self.focus))
        self.meta = int(data.get('meta', self.meta))
    
    def __str__(self) -> str:
        return (f"CarControl(accel={self.accel:.2f}, brake={self.brake:.2f}, "
                f"gear={self.gear}, steer={self.steer:.2f})")