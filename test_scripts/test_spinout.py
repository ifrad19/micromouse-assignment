import math
from diff_drive_robot import DifferentialDriveRobot, DiffDriveController
import numpy as np


def test_spinout():
    """Test spin-out behavior at different speeds and turn rates"""
    
    config = {
        'diff_drive': {
            'wheel_radius': 0.033,
            'wheelbase': 0.10,
            'max_wheel_speed': 20.0,        # High to allow fast speeds
            'max_acceleration': 10.0,       # Fast response
            
            # Spinout-prone settings
            'base_slip_factor': 0.98,
            'velocity_slip_factor': 0.12,
            'turn_rate_slip_factor': 0.18,
            'combined_slip_factor': 0.10,   # Key parameter!
            'spinout_threshold': 0.70,
            'spinout_angular_boost': 3.0,
            'spinout_linear_reduction': 0.3,
            'recovery_threshold': 0.80,
            'spinout_noise': 0.2,
            
            # Physics model
            'use_lateral_accel_model': True,
            'coefficient_friction': 0.55
        }
    }
    
    robot = DifferentialDriveRobot(config)
    controller = DiffDriveController(robot.wheel_radius, robot.wheelbase)
    
    print("=" * 70)
    print("DIFFERENTIAL DRIVE SPIN-OUT TEST")
    print("=" * 70)
    
    # Test 1: Safe low-speed turn
    print("\nTest 1: Low speed, moderate turn (SAFE)")
    print("-" * 70)
    robot.set_state(0, 0, 0)
    robot.is_spinning_out = False
    v_left, v_right = controller.velocity_to_wheels(v=0.15, omega=0.8)
    
    for i in range(10):
        robot.set_wheel_velocities(v_left, v_right, 0.1)
        v, omega = robot.update_kinematics(0.1)
        x, y, theta = robot.get_state()
        diag = robot.get_diagnostics()
        status = "[SPINOUT!]" if diag['is_spinning_out'] else "[Stable]"
        print(f"  t={i*0.1:.1f}s: v={v:.3f} m/s, ω={omega:.2f} rad/s, "
              f"grip={diag['slip_factor']*100:.1f}% - {status}")
    
    # Test 2: Moderate speed, sharp turn
    print("\nTest 2: Medium speed, sharp turn (RISKY)")
    print("-" * 70)
    robot.set_state(0, 0, 0)
    robot.is_spinning_out = False
    v_left, v_right = controller.velocity_to_wheels(v=0.3, omega=1.5)
    
    for i in range(10):
        robot.set_wheel_velocities(v_left, v_right, 0.1)
        v, omega = robot.update_kinematics(0.1)
        x, y, theta = robot.get_state()
        diag = robot.get_diagnostics()
        status = "[SPINOUT!]" if diag['is_spinning_out'] else "[Stable]"
        print(f"  t={i*0.1:.1f}s: v={v:.3f} m/s, ω={omega:.2f} rad/s, "
              f"grip={diag['slip_factor']*100:.1f}% - {status}")
    
    # Test 3: High-speed sharp turn (DANGER)
    print("\nTest 3: HIGH SPEED + SHARP TURN (DANGER!)")
    print("-" * 70)
    robot.set_state(0, 0, 0)
    robot.is_spinning_out = False
    v_left, v_right = controller.velocity_to_wheels(v=0.5, omega=2.5)
    
    for i in range(10):
        robot.set_wheel_velocities(v_left, v_right, 0.1)
        v, omega = robot.update_kinematics(0.1)
        x, y, theta = robot.get_state()
        diag = robot.get_diagnostics()
        status = "[SPINOUT!]" if diag['is_spinning_out'] else "[Stable]"
        print(f"  t={i*0.1:.1f}s: v={v:.3f} m/s, ω={omega:.2f} rad/s, "
              f"grip={diag['slip_factor']*100:.1f}% - {status}")
    
    print("\n" + "=" * 70)
    print("Key: Low grip % = more slip. Spinout occurs when grip < 70%")
    print("=" * 70)


if __name__ == "__main__":
    test_spinout()