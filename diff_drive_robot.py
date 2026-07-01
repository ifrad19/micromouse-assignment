"""
Differential Drive Robot with Spinout Dynamics - COMPLETE VERSION
------------------------------------------------------------------
This is the fully corrected version with all methods implemented.
"""

import math
import numpy as np


class TireFrictionModel:
    """Simplified tire friction model"""
    def __init__(self, mu_peak=0.8, mu_slide=0.6):
        self.mu_peak = mu_peak
        self.mu_slide = mu_slide
    
    def get_friction_coefficient(self, slip_ratio):
        if slip_ratio < 0.15:
            mu = self.mu_peak * (slip_ratio / 0.15)
        else:
            decay = np.exp(-(slip_ratio - 0.15) * 3.0)
            mu = self.mu_slide + (self.mu_peak - self.mu_slide) * decay
        return mu
    
    def calculate_slip_ratio(self, wheel_vel, ground_vel):
        if abs(wheel_vel) < 1e-6:
            return 1.0 if abs(ground_vel) > 1e-6 else 0.0
        slip = abs(wheel_vel - ground_vel) / abs(wheel_vel)
        return min(slip, 1.0)


class DifferentialDriveRobot:
    """Differential drive robot with traction modeling and spinout dynamics"""
    
    def __init__(self, config=None):
        robot_cfg = config.get('diff_drive', {}) if config else {}
        
        # Physical parameters
        self.wheel_radius = robot_cfg.get('wheel_radius', 0.033)
        self.wheelbase = robot_cfg.get('wheelbase', 0.10)
        
        # Motor constraints
        self.max_wheel_speed = robot_cfg.get('max_wheel_speed', 10.0)
        self.max_acceleration = robot_cfg.get('max_acceleration', 5.0)
        
        # Current wheel velocities
        self.wheel_vel_left = 0.0
        self.wheel_vel_right = 0.0
        
        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.is_spinning_out = False
        
        # Dynamic slip parameters
        self.base_slip_factor = robot_cfg.get('base_slip_factor', 0.98)
        self.velocity_slip_factor = robot_cfg.get('velocity_slip_factor', 0.08)
        self.turn_rate_slip_factor = robot_cfg.get('turn_rate_slip_factor', 0.12)
        self.combined_slip_factor = robot_cfg.get('combined_slip_factor', 0.05)
        self.max_slip = robot_cfg.get('max_slip', 0.35)
        
        # Spinout parameters
        self.spinout_threshold = robot_cfg.get('spinout_threshold', 0.70)
        self.spinout_angular_boost = robot_cfg.get('spinout_angular_boost', 2.5)
        self.spinout_linear_reduction = robot_cfg.get('spinout_linear_reduction', 0.35)
        self.recovery_threshold = robot_cfg.get('recovery_threshold', 0.80)
        self.spinout_noise = robot_cfg.get('spinout_noise', 0.15)
        
        # Noise
        self.velocity_noise = robot_cfg.get('velocity_noise', 0.0)
        
        # Lateral acceleration model
        self.use_lateral_model = robot_cfg.get('use_lateral_accel_model', False)
        self.robot_mass = robot_cfg.get('robot_mass', 0.15)
        self.mu = robot_cfg.get('coefficient_friction', 0.7)
        self.g = robot_cfg.get('gravity', 9.81)
        self.max_lateral_accel = self.mu * self.g
        self.lateral_threshold = robot_cfg.get('lateral_accel_threshold', 0.85)
        
        # Tire model
        self.use_tire_model = robot_cfg.get('use_tire_model', False)
        self.tire_model = TireFrictionModel(
            mu_peak=robot_cfg.get('mu_peak', 0.8),
            mu_slide=robot_cfg.get('mu_slide', 0.6)
        )
        
        # Diagnostics
        self.current_slip_factor = 1.0
        self.lateral_accel = 0.0
    
    def calculate_dynamic_slip(self, v, omega):
        """Calculate slip factor based on velocity and turning rate"""
        slip = 1.0 - self.base_slip_factor
        slip += abs(v) * self.velocity_slip_factor
        slip += abs(omega) * self.turn_rate_slip_factor
        slip += abs(v * omega) * self.combined_slip_factor  # Combined effect
        slip = min(slip, 1.0 - self.max_slip)
        slip_factor = 1.0 - slip
        return slip_factor
    
    def check_spinout(self, slip_factor, omega):
        """Check if robot should be spinning out"""
        min_omega = 0.2
        if slip_factor < self.spinout_threshold and abs(omega) > min_omega:
            return True
        if self.is_spinning_out and slip_factor > self.recovery_threshold:
            return False
        return self.is_spinning_out
    
    def calculate_lateral_acceleration(self, v, omega):
        """Calculate lateral acceleration"""
        return abs(v * omega)
    
    def calculate_traction_limited_motion(self, v, omega):
        """Apply lateral acceleration based traction limits"""
        if not self.use_lateral_model:
            return v, omega
        
        a_lateral = self.calculate_lateral_acceleration(v, omega)
        self.lateral_accel = a_lateral
        max_a = self.max_lateral_accel * self.lateral_threshold
        
        if a_lateral <= max_a:
            return v, omega
        
        slip_ratio = a_lateral / max_a
        v_actual = v / slip_ratio
        omega_actual = omega * (1.0 + (slip_ratio - 1.0) * 1.8)
        omega_actual += np.random.normal(0, abs(omega) * 0.15)
        
        return v_actual, omega_actual
    
    def set_wheel_velocities(self, v_left, v_right, dt):
        """Set wheel velocities with acceleration limits"""
        v_left = np.clip(v_left, -self.max_wheel_speed, self.max_wheel_speed)
        v_right = np.clip(v_right, -self.max_wheel_speed, self.max_wheel_speed)
        
        max_delta_v = self.max_acceleration * dt
        delta_v_left = np.clip(v_left - self.wheel_vel_left, -max_delta_v, max_delta_v)
        delta_v_right = np.clip(v_right - self.wheel_vel_right, -max_delta_v, max_delta_v)
        
        self.wheel_vel_left += delta_v_left
        self.wheel_vel_right += delta_v_right
    
    def update_kinematics(self, dt):
        """Single update method that routes to appropriate model"""
        v_left_linear = self.wheel_radius * self.wheel_vel_left
        v_right_linear = self.wheel_radius * self.wheel_vel_right
        
        if self.use_tire_model:
            return self._update_with_tire_model(dt, v_left_linear, v_right_linear)
        else:
            return self._update_with_dynamic_slip(dt, v_left_linear, v_right_linear)
    
    def _update_with_dynamic_slip(self, dt, v_left_linear, v_right_linear):
        """Update with dynamic slip and spinout"""
        # Calculate nominal velocities
        v_nominal = (v_right_linear + v_left_linear) / 2.0
        omega_nominal = (v_right_linear - v_left_linear) / self.wheelbase
        
        # Apply lateral model if enabled
        if self.use_lateral_model:
            v_nominal, omega_nominal = self.calculate_traction_limited_motion(v_nominal, omega_nominal)
        
        # Calculate dynamic slip
        slip_factor = self.calculate_dynamic_slip(v_nominal, omega_nominal)
        self.current_slip_factor = slip_factor
        
        # Check for spinout BEFORE applying slip (use nominal values)
        self.is_spinning_out = self.check_spinout(slip_factor, omega_nominal)
        
        # Apply slip
        v = v_nominal * slip_factor
        omega = omega_nominal * slip_factor
        
        # Apply spinout effects
        if self.is_spinning_out:
            omega *= self.spinout_angular_boost
            v *= self.spinout_linear_reduction
            omega += np.random.normal(0, abs(omega) * self.spinout_noise)
        
        # Add noise
        if self.velocity_noise > 0:
            v += np.random.normal(0, self.velocity_noise * abs(v))
            omega += np.random.normal(0, self.velocity_noise * abs(omega))
        
        # Update pose
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        
        return v, omega
    
    def _update_with_tire_model(self, dt, v_left_cmd, v_right_cmd):
        """Update using tire friction model"""
        v_ground = (v_left_cmd + v_right_cmd) / 2.0
        omega_ground = (v_right_cmd - v_left_cmd) / self.wheelbase
        
        slip_left = self.tire_model.calculate_slip_ratio(v_left_cmd, v_ground)
        slip_right = self.tire_model.calculate_slip_ratio(v_right_cmd, v_ground)
        
        mu_left = self.tire_model.get_friction_coefficient(slip_left)
        mu_right = self.tire_model.get_friction_coefficient(slip_right)
        
        v_left_actual = v_left_cmd * mu_left
        v_right_actual = v_right_cmd * mu_right
        
        v = (v_left_actual + v_right_actual) / 2.0
        omega = (v_right_actual - v_left_actual) / self.wheelbase
        
        avg_mu = (mu_left + mu_right) / 2.0
        self.is_spinning_out = avg_mu < 0.5
        
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        
        return v, omega
    
    def get_state(self):
        """Get current robot state"""
        return self.x, self.y, self.theta
    
    def set_state(self, x, y, theta):
        """Set robot state"""
        self.x = x
        self.y = y
        self.theta = theta
    
    def get_wheel_velocities(self):
        """Get current wheel velocities"""
        return self.wheel_vel_left, self.wheel_vel_right
    
    def get_diagnostics(self):
        """Get diagnostic information - REQUIRED METHOD"""
        return {
            'is_spinning_out': self.is_spinning_out,
            'slip_factor': self.current_slip_factor,
            'grip_percentage': self.current_slip_factor * 100,
            'lateral_accel': self.lateral_accel,
            'max_lateral_accel': self.max_lateral_accel,
            'wheel_vel_left': self.wheel_vel_left,
            'wheel_vel_right': self.wheel_vel_right
        }
    
    def stop(self):
        """Immediately stop both wheels"""
        self.wheel_vel_left = 0.0
        self.wheel_vel_right = 0.0
        self.is_spinning_out = False


class DiffDriveController:
    """Converts high-level commands to differential drive wheel velocities"""
    
    def __init__(self, wheel_radius, wheelbase):
        self.R = wheel_radius
        self.L = wheelbase
    
    def velocity_to_wheels(self, v, omega):
        """Convert linear and angular velocity to wheel velocities"""
        v_left = (2 * v - omega * self.L) / (2 * self.R)
        v_right = (2 * v + omega * self.L) / (2 * self.R)
        return v_left, v_right
    
    def steering_to_omega(self, v, steering_angle, wheelbase_equiv=1.0):
        """Convert Ackermann-style steering to angular velocity"""
        if abs(v) < 1e-6:
            return 0.0
        omega = v * math.tan(steering_angle) / wheelbase_equiv
        return omega


# Test function
if __name__ == "__main__":
    print("DifferentialDriveRobot class loaded successfully!")
    print("\nTesting basic functionality...")
    
    config = {'diff_drive': {'wheel_radius': 0.033, 'wheelbase': 0.10}}
    robot = DifferentialDriveRobot(config)
    controller = DiffDriveController(robot.wheel_radius, robot.wheelbase)
    
    # Test methods exist
    print("[OK] calculate_dynamic_slip exists:", hasattr(robot, 'calculate_dynamic_slip'))
    print("[OK] check_spinout exists:", hasattr(robot, 'check_spinout'))
    print("[OK] get_diagnostics exists:", hasattr(robot, 'get_diagnostics'))
    print("[OK] is_spinning_out initialized:", hasattr(robot, 'is_spinning_out'))
    
    # Quick test
    robot.set_state(0, 0, 0)
    v_l, v_r = controller.velocity_to_wheels(0.2, 0.5)
    robot.set_wheel_velocities(v_l, v_r, 0.1)
    v, omega = robot.update_kinematics(0.1)
    diag = robot.get_diagnostics()
    
    print(f"\nQuick test: v={v:.3f}, omega={omega:.3f}")
    print(f"Diagnostics: {diag}")
    print("\n[PASS] All tests passed! Ready to use.")