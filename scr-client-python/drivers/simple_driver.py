"""
Simple rule-based driver implementation.
Python port of the C++ SimpleDriver with the same driving logic.
"""
import math
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.base_driver import BaseDriver, Stage
from src.car_state import CarState
from src.car_control import CarControl


class SimpleDriver(BaseDriver):
    """Rule-based driver implementation with same logic as C++ version."""
    
    # Gear changing constants
    GEAR_UP = [5000, 6000, 6000, 6500, 7000, 0]
    GEAR_DOWN = [0, 2500, 3000, 3000, 3500, 3500]
    
    # Stuck constants
    STUCK_TIME = 25
    STUCK_ANGLE = math.pi / 6  # 30 degrees
    
    # Accel and brake constants
    MAX_SPEED_DIST = 70
    MAX_SPEED = 150
    SIN5 = 0.08716
    COS5 = 0.99619
    
    # Steering constants
    STEER_LOCK = 0.366519
    STEER_SENSITIVITY_OFFSET = 80.0
    WHEEL_SENSITIVITY_COEFF = 1
    
    # ABS filter constants
    WHEEL_RADIUS = [0.3306, 0.3306, 0.3276, 0.3276]
    ABS_SLIP = 2.0
    ABS_RANGE = 3.0
    ABS_MIN_SPEED = 3.0
    
    # Clutch constants
    CLUTCH_MAX = 0.5
    CLUTCH_DELTA = 0.05
    CLUTCH_RANGE = 0.82
    CLUTCH_DELTA_TIME = 0.02
    CLUTCH_DELTA_RACED = 10
    CLUTCH_DEC = 0.01
    CLUTCH_MAX_MODIFIER = 1.3
    CLUTCH_MAX_TIME = 1.5
    
    def __init__(self):
        super().__init__()
        self.stuck = 0
        self.clutch = 0.0
    
    def init(self) -> List[float]:
        """Initialize sensor angles same as C++ version."""
        angles = [0.0] * 19
        
        # Set angles as {-90,-75,-60,-45,-30,-20,-15,-10,-5,0,5,10,15,20,30,45,60,75,90}
        for i in range(5):
            angles[i] = -90 + i * 15
            angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            angles[i] = -20 + (i - 5) * 5
            angles[18 - i] = 20 - (i - 5) * 5
        
        angles[9] = 0
        return angles
    
    def get_gear(self, cs: CarState) -> int:
        """Get appropriate gear based on RPM."""
        gear = cs.get_gear()
        rpm = cs.get_rpm()
        
        # If gear is 0 (N) or -1 (R) just return 1
        if gear < 1:
            return 1
            
        # Check if should shift up
        if gear < 6 and rpm >= self.GEAR_UP[gear - 1]:
            return gear + 1
        # Check if should shift down    
        elif gear > 1 and rpm <= self.GEAR_DOWN[gear - 1]:
            return gear - 1
        else:
            return gear
    
    def get_steer(self, cs: CarState) -> float:
        """Calculate steering angle."""
        # Steering angle corrects car angle w.r.t. track axis and position
        target_angle = cs.get_angle() - cs.get_track_pos() * 0.5
        
        # At high speed reduce steering to avoid losing control
        if cs.get_speed_x() > self.STEER_SENSITIVITY_OFFSET:
            return target_angle / (self.STEER_LOCK * 
                                 (cs.get_speed_x() - self.STEER_SENSITIVITY_OFFSET) * 
                                 self.WHEEL_SENSITIVITY_COEFF)
        else:
            return target_angle / self.STEER_LOCK
    
    def get_accel(self, cs: CarState) -> float:
        """Calculate acceleration/braking."""
        # Check if car is on track
        if -1 < cs.get_track_pos() < 1:
            # Reading sensors at +5, 0, -5 degrees
            rx_sensor = cs.get_track(10)  # +5 degrees
            c_sensor = cs.get_track(9)    # 0 degrees  
            sx_sensor = cs.get_track(8)   # -5 degrees
            
            # Track is straight and far from turn
            if c_sensor > self.MAX_SPEED_DIST or (c_sensor >= rx_sensor and c_sensor >= sx_sensor):
                target_speed = self.MAX_SPEED
            else:
                # Approaching a turn
                if rx_sensor > sx_sensor:
                    # Turn on right
                    h = c_sensor * self.SIN5
                    b = rx_sensor - c_sensor * self.COS5
                    sin_angle = b * b / (h * h + b * b)
                    target_speed = self.MAX_SPEED * (c_sensor * sin_angle / self.MAX_SPEED_DIST)
                else:
                    # Turn on left
                    h = c_sensor * self.SIN5
                    b = sx_sensor - c_sensor * self.COS5
                    sin_angle = b * b / (h * h + b * b)
                    target_speed = self.MAX_SPEED * (c_sensor * sin_angle / self.MAX_SPEED_DIST)
            
            # Exponential scaling of accel/brake command
            return 2 / (1 + math.exp(cs.get_speed_x() - target_speed)) - 1
        else:
            # Out of track - moderate acceleration
            return 0.3
    
    def filter_abs(self, cs: CarState, brake: float) -> float:
        """Apply ABS filtering to brake command."""
        # Convert speed to m/s
        speed = cs.get_speed_x() / 3.6
        
        # When speed lower than min speed for ABS, do nothing
        if speed < self.ABS_MIN_SPEED:
            return brake
        
        # Compute average wheel speed
        slip = 0.0
        for i in range(4):
            slip += cs.get_wheel_spin_vel(i) * self.WHEEL_RADIUS[i]
        
        # Slip is difference between car speed and average wheel speed
        slip = speed - slip / 4.0
        
        # When slip too high, apply ABS
        if slip > self.ABS_SLIP:
            brake = brake - (slip - self.ABS_SLIP) / self.ABS_RANGE
        
        # Ensure brake is not negative
        return max(0.0, brake)
    
    def clutching(self, cs: CarState, clutch: float) -> float:
        """Calculate clutch value."""
        max_clutch = self.CLUTCH_MAX
        
        # Check if this is race start
        if (cs.get_cur_lap_time() < self.CLUTCH_DELTA_TIME and 
            self.stage == Stage.RACE and 
            cs.get_dist_raced() < self.CLUTCH_DELTA_RACED):
            clutch = max_clutch
        
        # Adjust clutch value
        if clutch > 0:
            delta = self.CLUTCH_DELTA
            if cs.get_gear() < 2:
                # Stronger clutch for gear 1 at race start
                delta /= 2
                max_clutch *= self.CLUTCH_MAX_MODIFIER
                if cs.get_cur_lap_time() < self.CLUTCH_MAX_TIME:
                    clutch = max_clutch
            
            # Clamp to max value
            clutch = min(max_clutch, clutch)
            
            # Decrease clutch
            if clutch != max_clutch:
                clutch -= delta
                clutch = max(0.0, clutch)
            else:
                clutch -= self.CLUTCH_DEC
        
        return clutch
    
    def drive(self, sensors: str) -> str:
        """Main driving function."""
        cs = self._parse_state(sensors)
        
        # Check if car is stuck
        if abs(cs.get_angle()) > self.STUCK_ANGLE:
            self.stuck += 1
        else:
            self.stuck = 0
        
        # Apply stuck recovery
        if self.stuck > self.STUCK_TIME:
            # Set gear and steering for stuck recovery
            steer = -cs.get_angle() / self.STEER_LOCK
            gear = -1  # Reverse gear
            
            # If pointing in correct direction, use forward gear
            if cs.get_angle() * cs.get_track_pos() > 0:
                gear = 1
                steer = -steer
            
            # Calculate clutch
            self.clutch = self.clutching(cs, self.clutch)
            
            # Create control command
            control = CarControl(1.0, 0.0, gear, steer, self.clutch)
            return control.to_string()
        
        else:
            # Normal driving
            # Calculate accel/brake
            accel_and_brake = self.get_accel(cs)
            
            # Calculate gear
            gear = self.get_gear(cs)
            
            # Calculate steering
            steer = self.get_steer(cs)
            
            # Normalize steering
            steer = max(-1.0, min(1.0, steer))
            
            # Set accel and brake
            if accel_and_brake > 0:
                accel = accel_and_brake
                brake = 0.0
            else:
                accel = 0.0
                brake = self.filter_abs(cs, -accel_and_brake)
            
            # Calculate clutch
            self.clutch = self.clutching(cs, self.clutch)
            
            # Create control command
            control = CarControl(accel, brake, gear, steer, self.clutch)
            return control.to_string()
    
    def on_shutdown(self) -> None:
        """Called on shutdown."""
        print("Bye bye!")
    
    def on_restart(self) -> None:
        """Called on restart."""
        print("Restarting the race!")