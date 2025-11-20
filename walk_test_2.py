import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
XML_PATH = "scene.xml"
SIM_DURATION = 600.0

# --- Enhanced Control Parameters ---
KP_BASE = 60.0
KD_BASE = 3.0

# Joint-specific tuning
HIP_ABDUCT_KP, HIP_ABDUCT_KD = 0.8, 0.8
HIP_FLEX_KP, HIP_FLEX_KD = 1.0, 1.0
KNEE_KP, KNEE_KD = 1.4, 1.4

# --- Gait Parameters ---
STAND_BASE = np.array([0.0, 0.9, -1.8])
WALK_FREQ = 2.0
SWING_AMP_THIGH = 0.3
SWING_AMP_CALF = 0.4

# --- Gravity Compensation & Feedforward ---
GRAVITY_COMP = {'abduction': 0.0, 'thigh': 1.8, 'calf': 4.5}
STANCE_FF = {'abduction': 0.0, 'thigh': 2.5, 'calf': 5.5}

# --- STEERING PARAMETERS ---
TARGET_YAW_RATE = 0.6  # rad/s, positive for left turn
STEERING_GAIN = 0.0
LATERAL_SHIFT_GAIN = 0.05
TURN_RATE_KP = 0.0
MAX_TURN_RATE = 0.0

# --- Robot Dimensions ---
n_qpos_per_robot = 19
n_qvel_per_robot = 18
n_ctrl_per_robot = 12

TORQUE_LIMITS = {
    'abduction': (-23.7, 23.7),
    'hip_flexion': (-23.7, 23.7),
    'knee': (-45.43, 45.43)
}

def get_joint_type(actuator_idx):
    type_idx = actuator_idx % 3
    return ['abduction', 'thigh', 'calf'][type_idx]

def get_leg_side(actuator_idx):
    return 'left' if actuator_idx in [3, 4, 5, 9, 10, 11] else 'right'

def get_gait_phase(sim_time, actuator_idx, robot_index):
    phase_offset = robot_index * 0.5
    base_phase = (sim_time * WALK_FREQ * 2 * np.pi + phase_offset) % (2 * np.pi)
    
    is_pair_A = (actuator_idx < 3) or (actuator_idx >= 9)
    leg_phase = base_phase if is_pair_A else base_phase + np.pi
    
    is_swing = np.sin(leg_phase) > 0
    return leg_phase, is_swing

def get_steering_bias(actuator_idx, target_yaw_rate, body_yaw_rate):
    """Calculate steering bias for amplitude scaling and abduction offset"""
    target_yaw_rate = np.clip(target_yaw_rate, -MAX_TURN_RATE, MAX_TURN_RATE)
    
    yaw_rate_error = target_yaw_rate - body_yaw_rate
    turn_command = target_yaw_rate + TURN_RATE_KP * yaw_rate_error
    
    amplitude_scale = 1.0 + (get_leg_side(actuator_idx) == 'left' and -1.0 or 1.0) * turn_command * STEERING_GAIN
    amplitude_scale = np.clip(amplitude_scale, 0.5, 1.5)
    
    abduction_offset = 0.0
    if actuator_idx % 3 == 0:  # Abduction joints
        side_sign = 1.0 if get_leg_side(actuator_idx) == 'left' else -1.0
        abduction_offset = side_sign * LATERAL_SHIFT_GAIN * turn_command
    
    return amplitude_scale, abduction_offset

def get_gait_target(sim_time, actuator_idx, robot_index, target_yaw_rate, body_yaw_rate):
    """Generates target angles with realistic steering"""
    leg_phase, _ = get_gait_phase(sim_time, actuator_idx, robot_index)
    amplitude_scale, abduction_offset = get_steering_bias(actuator_idx, target_yaw_rate, body_yaw_rate)
    
    joint_type = actuator_idx % 3
    base_angle = STAND_BASE[joint_type]
    
    # Apply CoM shift to abduction joints
    if joint_type == 0:
        base_angle += abduction_offset
    
    # Apply differential swing amplitude
    adjustment = 0.0
    if joint_type == 1:  # Thigh
        adjustment = -SWING_AMP_THIGH * np.sin(leg_phase) * amplitude_scale
    elif joint_type == 2:  # Calf
        swing_lift = np.cos(leg_phase)
        if swing_lift > 0:
            adjustment = -SWING_AMP_CALF * swing_lift * amplitude_scale
    
    return base_angle + adjustment

def enhanced_pd_controller(target_q, current_q, current_v, kp, kd, 
                          actuator_idx, leg_phase, robot_vel_x):
    """Enhanced PD controller based on XML physics"""
    joint_type = get_joint_type(actuator_idx)
    
    kp_tuned = kp * (HIP_ABDUCT_KP if joint_type == 'abduction' else 
                    HIP_FLEX_KP if joint_type == 'thigh' else KNEE_KP)
    kd_tuned = kd * (HIP_ABDUCT_KD if joint_type == 'abduction' else 
                    HIP_FLEX_KD if joint_type == 'thigh' else KNEE_KD)
    
    torque = kp_tuned * (target_q - current_q) - kd_tuned * current_v
    torque += GRAVITY_COMP[joint_type]
    
    # Gait phase modulation
    is_swing = np.sin(leg_phase) > 0
    if is_swing:
        torque *= 0.5
        if np.sin(leg_phase) > 0.5 and joint_type == 'thigh':
            torque += 2.0
    else:
        torque += STANCE_FF[joint_type]
        if np.sin(leg_phase) < -0.5 and joint_type == 'calf':
            torque += 3.0
    
    # Velocity damping
    damping_factor = min(abs(robot_vel_x) * 0.4, 2.0)
    torque -= damping_factor * current_v
    
    # Torque limiting
    limit_key = 'knee' if joint_type == 'calf' else 'abduction'
    torque = np.clip(torque, *TORQUE_LIMITS[limit_key])
    
    return torque

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}. Run generate_scene.py first.")
        return

    num_robots = model.nu // n_ctrl_per_robot
    print(f"Loaded Scene with {num_robots} robots. INITIATING STEERING GAIT...")

    joint_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_time = time.time()

        while viewer.is_running() and time.time() - start_time < SIM_DURATION:
            step_start = time.time()
            sim_time = data.time

            walking_active = sim_time > 1.0

            for r in range(num_robots):
                qpos_start_idx = r * n_qpos_per_robot
                qvel_start_idx = r * n_qvel_per_robot
                ctrl_start_idx = r * n_ctrl_per_robot

                robot_joints_q = data.qpos[qpos_start_idx + 7: qpos_start_idx + 19]
                robot_joints_v = data.qvel[qvel_start_idx + 6: qvel_start_idx + 18]
                robot_vel_x = data.qvel[qvel_start_idx + 0]
                body_yaw_rate = data.qvel[qvel_start_idx + 5]

                for i in range(12):
                    leg_phase, _ = get_gait_phase(sim_time - 1.0, i, r)
                    
                    target_angle = get_gait_target(
                        sim_time - 1.0, i, r, TARGET_YAW_RATE, body_yaw_rate
                    ) if walking_active else STAND_BASE[i % 3]

                    joint_idx = joint_map[i]
                    curr_q = robot_joints_q[joint_idx]
                    curr_v = robot_joints_v[joint_idx]

                    torque = enhanced_pd_controller(
                        target_angle, curr_q, curr_v, KP_BASE, KD_BASE,
                        i, leg_phase, robot_vel_x
                    )
                    
                    data.ctrl[ctrl_start_idx + i] = torque

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # FIXED: Corrected time.sleep() call
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()