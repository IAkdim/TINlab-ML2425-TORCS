"""
Car state representation for SCR client.
Contains all sensor data received from the TORCS server.
"""
from typing import List, Optional
from .simple_parser import SimpleParser


class CarState:
    """Represents the current state of the car based on sensor data."""
    
    def __init__(self, sensor_data: str = ""):
        # Initialize with default values
        self.angle: float = 0.0
        self.cur_lap_time: float = 0.0
        self.damage: float = 0.0
        self.dist_from_start: float = 0.0
        self.dist_raced: float = 0.0
        self.focus: List[float] = [-1.0] * 5  # 5 focus sensors
        self.fuel: float = 0.0
        self.gear: int = 0
        self.last_lap_time: float = 0.0
        self.opponents: List[float] = [200.0] * 36  # 36 opponent sensors
        self.race_pos: int = 1
        self.rpm: float = 0.0
        self.speed_x: float = 0.0
        self.speed_y: float = 0.0
        self.speed_z: float = 0.0
        self.track: List[float] = [200.0] * 19  # 19 track sensors
        self.track_pos: float = 0.0
        self.wheel_spin_vel: List[float] = [0.0] * 4  # 4 wheels
        self.z: float = 0.0
        
        if sensor_data:
            self.parse_sensors(sensor_data)
    
    def parse_sensors(self, sensor_data: str) -> None:
        """Parse sensor data string and update state."""
        data = SimpleParser.parse(sensor_data)
        
        # Map parsed data to attributes
        self.angle = data.get('angle', self.angle)
        self.cur_lap_time = data.get('curLapTime', self.cur_lap_time)
        self.damage = data.get('damage', self.damage)
        self.dist_from_start = data.get('distFromStart', self.dist_from_start)
        self.dist_raced = data.get('distRaced', self.dist_raced)
        self.fuel = data.get('fuel', self.fuel)
        self.gear = int(data.get('gear', self.gear))
        self.last_lap_time = data.get('lastLapTime', self.last_lap_time)
        self.race_pos = int(data.get('racePos', self.race_pos))
        self.rpm = data.get('rpm', self.rpm)
        self.speed_x = data.get('speedX', self.speed_x)
        self.speed_y = data.get('speedY', self.speed_y)
        self.speed_z = data.get('speedZ', self.speed_z)
        self.track_pos = data.get('trackPos', self.track_pos)
        self.z = data.get('z', self.z)
        
        # Handle array data
        if 'focus' in data and isinstance(data['focus'], list):
            self.focus = data['focus'][:5]  # Ensure max 5 elements
        if 'opponents' in data and isinstance(data['opponents'], list):
            self.opponents = data['opponents'][:36]  # Ensure max 36 elements
        if 'track' in data and isinstance(data['track'], list):
            self.track = data['track'][:19]  # Ensure max 19 elements
        if 'wheelSpinVel' in data and isinstance(data['wheelSpinVel'], list):
            self.wheel_spin_vel = data['wheelSpinVel'][:4]  # Ensure max 4 elements
    
    # Getter methods for compatibility with C++ version
    def get_angle(self) -> float:
        return self.angle
    
    def get_cur_lap_time(self) -> float:
        return self.cur_lap_time
    
    def get_damage(self) -> float:
        return self.damage
    
    def get_dist_from_start(self) -> float:
        return self.dist_from_start
    
    def get_dist_raced(self) -> float:
        return self.dist_raced
    
    def get_focus(self, i: int) -> float:
        return self.focus[i] if 0 <= i < len(self.focus) else -1.0
    
    def get_fuel(self) -> float:
        return self.fuel
    
    def get_gear(self) -> int:
        return self.gear
    
    def get_last_lap_time(self) -> float:
        return self.last_lap_time
    
    def get_opponents(self, i: int) -> float:
        return self.opponents[i] if 0 <= i < len(self.opponents) else 200.0
    
    def get_race_pos(self) -> int:
        return self.race_pos
    
    def get_rpm(self) -> float:
        return self.rpm
    
    def get_speed_x(self) -> float:
        return self.speed_x
    
    def get_speed_y(self) -> float:
        return self.speed_y
    
    def get_speed_z(self) -> float:
        return self.speed_z
    
    def get_track(self, i: int) -> float:
        return self.track[i] if 0 <= i < len(self.track) else 200.0
    
    def get_track_pos(self) -> float:
        return self.track_pos
    
    def get_wheel_spin_vel(self, i: int) -> float:
        return self.wheel_spin_vel[i] if 0 <= i < len(self.wheel_spin_vel) else 0.0
    
    def get_z(self) -> float:
        return self.z
    
    def to_vector(self) -> List[float]:
        """Convert state to feature vector for ML models."""
        vector = [
            self.angle,
            self.speed_x,
            self.speed_y,
            self.track_pos,
            self.rpm / 10000.0,  # Normalize RPM
            float(self.gear),
            self.damage / 10000.0,  # Normalize damage
        ]
        
        # Add track sensors (normalized by dividing by 200)
        vector.extend([min(t / 200.0, 1.0) for t in self.track])
        
        # Add opponent sensors (normalized, closest 5)
        closest_opponents = sorted(self.opponents)[:5]
        vector.extend([min(o / 200.0, 1.0) for o in closest_opponents])
        
        return vector
    
    def __str__(self) -> str:
        return (f"CarState(angle={self.angle:.3f}, speed={self.speed_x:.1f}, "
                f"track_pos={self.track_pos:.3f}, gear={self.gear}, rpm={self.rpm:.0f})")