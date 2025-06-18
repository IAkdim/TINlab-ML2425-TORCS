"""
Base driver class for SCR client implementations.
Abstract base class that defines the interface for all drivers.
"""
from abc import ABC, abstractmethod
from typing import List
from enum import Enum

from .car_state import CarState
from .car_control import CarControl


class Stage(Enum):
    """Race stages."""
    WARMUP = 0
    QUALIFYING = 1
    RACE = 2
    UNKNOWN = 3


class BaseDriver(ABC):
    """Abstract base class for all driver implementations."""
    
    def __init__(self):
        self.stage = Stage.UNKNOWN
        self.track_name = "unknown"
    
    def init(self) -> List[float]:
        """
        Initialize track sensor angles.
        
        Returns:
            List of 19 sensor angles in degrees
        """
        # Default angles: -90 to +90 in 10-degree increments
        return [-90 + i * 10 for i in range(19)]
    
    @abstractmethod
    def drive(self, sensors: str) -> str:
        """
        Main driving function.
        
        Args:
            sensors: Sensor data string from server
            
        Returns:
            Control command string
        """
        pass
    
    def on_shutdown(self) -> None:
        """Called when client is shutting down."""
        pass
    
    def on_restart(self) -> None:
        """Called when race is restarting."""
        pass
    
    def _create_control(self, accel: float = 0.0, brake: float = 0.0, 
                       gear: int = 1, steer: float = 0.0, 
                       clutch: float = 0.0) -> CarControl:
        """Helper to create CarControl object."""
        return CarControl(accel, brake, gear, steer, clutch)
    
    def _parse_state(self, sensors: str) -> CarState:
        """Helper to parse sensor string into CarState."""
        return CarState(sensors)