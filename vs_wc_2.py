import mujoco
import mujoco.viewer
import numpy as np
import time
import sys

# =========================================================
# 1. CONFIGURATION (MERGED)
# =========================================================
XML_PATH = "scene.xml"
SIM_DURATION = 300.0

# --- FORMATION SETTINGS (From formation_control_1) ---
FORMATION_TYPE = "TRIANGLE"  # Options: LINE, TRIANGLE, CIRCLE
LINE_DIST = 1.0       
TRIANGLE_SPACING = 1.5 
CIRCLE_RADIUS = 3.0 
VS_GAIN_P = 8.0      
VS_GAIN_PSI = 2.0 
PATH_RADIUS = 8.0 
PATH_DIRECTION = 1.0 # 1.0 = Left, -1.0 = Right

# --- PHYSICS & GAIT SETTINGS (From walk_test_2) ---
KP_BASE = 60.0
KD_BASE = 3.0

# Joint-specific tuning
HIP_ABDUCT_KP, HIP_ABDUCT_KD = 0.8, 0.8
HIP_FLEX_KP, HIP_FLEX_KD = 1.0, 1.0
KNEE_KP, KNEE_KD = 1.4, 1.4

STAND_BASE = np.array([0.0, 0.9, -1.8])
CALIB_VELOCITY = 0.5      # Velocity at which gait frequency is nominal
CALIB_FREQ = 2.0          # Nominal frequency
SWING_AMP_THIGH = 0.3
SWING_AMP_CALF = 0.4

# Gravity Compensation & Feedforward
GRAVITY_COMP = {'abduction': 0.0, 'thigh': 1.8, 'calf': 4.5}
STANCE_FF = {'abduction': 0.0, 'thigh': 2.5, 'calf': 5.5}

# Steering Constants
STEERING_GAIN = -0.1
LATERAL_SHIFT_GAIN = 0.02
TURN_RATE_KP = 2.0
MAX_TURN_RATE = 2.0

# Torque Limits
TORQUE_LIMITS = {
    'abduction': (-23.7, 23.7),
    'hip_flexion': (-23.7, 23.7),
    'knee': (-45.43, 45.43)
}

# Robot Dimensions
n_qpos_per_robot = 19
n_qvel_per_robot = 18
n_ctrl_per_robot = 12

# =========================================================
# 2. PHYSICS HELPERS (From walk_test_2)
# =========================================================

def get_joint_type(actuator_idx):
    type_idx = actuator_idx % 3
    return ['abduction', 'thigh', 'calf'][type_idx]

def get_leg_side(actuator_idx):
    # Based on mapping: Right legs are 0-2, 6-8. Left are 3-5, 9-11
    # But wait, let's check the joint_map usage in main.
    # Default map used is [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    # Actuator 0,1,2 -> mapped to 3,4,5 (Left Front)
    return 'left' if actuator_idx in [0, 1, 2, 6, 7, 8] else 'right' 

def get_steering_bias(actuator_idx, target_yaw_rate, body_yaw_rate):
    """Calculate steering bias for amplitude scaling and abduction offset"""
    # Note: target_yaw_rate comes from formation steer_cmd
    target_yaw_rate = np.clip(target_yaw_rate, -MAX_TURN_RATE, MAX_TURN_RATE)
    
    yaw_rate_error = target_yaw_rate - body_yaw_rate
    turn_command = target_yaw_rate + TURN_RATE_KP * yaw_rate_error
    
    # Determine side for logic (0-2 are Left in the actuator list order of the loop?)
    # Let's rely on the loop index logic:
    # In the loop below, i=0..2 is Left Front, i=3..5 is Right Front, etc.
    is_left = (actuator_idx < 3) or (actuator_idx >= 6 and actuator_idx < 9)
    
    amplitude_scale = 1.0 + (is_left and -1.0 or 1.0) * turn_command * STEERING_GAIN
    amplitude_scale = np.clip(amplitude_scale, 0.5, 1.5)
    
    abduction_offset = 0.0
    if actuator_idx % 3 == 0:  # Abduction joints
        side_sign = 1.0 if is_left else -1.0
        abduction_offset = side_sign * LATERAL_SHIFT_GAIN * turn_command
    
    return amplitude_scale, abduction_offset

def enhanced_pd_controller(target_q, current_q, current_v, kp, kd, 
                          actuator_idx, leg_phase, robot_vel_x):
    """Enhanced PD controller based on walk_test_2 physics"""
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

def get_gait_target_enhanced(accumulated_phase, actuator_idx, target_yaw_rate, body_yaw_rate):
    """
    Generates target angles using enhanced steering logic but 
    driven by accumulated phase (variable velocity)
    """
    # Determine leg pair phase shift
    # Pair A: Left Front (0-2) and Right Hind (9-11 in joint_map / 6-8 in ctrl loop)
    # Let's standardize: 
    # The controller loop iterates 0..11. 
    # Indices 0-2 (Left Front), 3-5 (Right Front), 6-8 (Left Hind), 9-11 (Right Hind)
    
    is_pair_A = (actuator_idx < 3) or (actuator_idx >= 9) 
    leg_phase = accumulated_phase if is_pair_A else accumulated_phase + np.pi

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
    
    return base_angle + adjustment, leg_phase

# =========================================================
# 3. MATH & FORMATION HELPERS (From formation_control_1)
# =========================================================

def quat_to_yaw(quat):
    w, x, y, z = quat
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_formation_nodes(formation_type, num_robots):
    nodes = []
    if formation_type == "LINE":
        length = (num_robots - 1) * LINE_DIST
        start_x = length / 2.0
        for i in range(num_robots):
            nodes.append([start_x - i * LINE_DIST, 0.0])
            
    elif formation_type == "TRIANGLE":
        row_height = (np.sqrt(3)/2) * TRIANGLE_SPACING
        count = 0
        row = 0
        temp_nodes = []
        while count < num_robots:
            slots_in_row = row + 1
            row_width = (slots_in_row - 1) * TRIANGLE_SPACING
            x_pos = -row * row_height 
            start_y = -row_width / 2.0
            for i in range(slots_in_row):
                if count >= num_robots: break
                y_pos = start_y + i * TRIANGLE_SPACING
                temp_nodes.append([x_pos, y_pos])
                count += 1
            row += 1
        temp_nodes = np.array(temp_nodes)
        centroid = np.mean(temp_nodes, axis=0)
        nodes = temp_nodes - centroid
        
    elif formation_type == "CIRCLE":
            # Leader (robot n-1) at front (angle=0), others follow CCW
            # After [::-1] reversal in main(), robot n-1 gets the first node
            for i in range(num_robots):
                angle = i * 2 * np.pi / num_robots
                x = CIRCLE_RADIUS * np.cos(angle)
                y = CIRCLE_RADIUS * np.sin(angle)
                nodes.append([x, y])
            nodes = np.array(nodes)
            
    else:
        raise ValueError(f"Unknown formation type: {formation_type}")
        
    return np.array(nodes)

def fit_virtual_structure(robot_positions, local_nodes):
    centroid_robots = np.mean(robot_positions, axis=0)
    centroid_nodes = np.mean(local_nodes, axis=0)
    robots_centered = robot_positions - centroid_robots
    nodes_centered = local_nodes - centroid_nodes
    
    num = 0.0
    den = 0.0
    for i in range(len(local_nodes)):
        xn, yn = nodes_centered[i]
        xr, yr = robots_centered[i]
        num += (xn * yr - yn * xr)
        den += (xn * xr + yn * yr)
        
    best_yaw = np.arctan2(num, den)
    return centroid_robots, best_yaw

# =========================================================
# 4. DASHBOARD
# =========================================================
def print_vs_dashboard(sim_time, formation_type, robot_data, ref_yaw, fit_yaw, omega):
    sys.stdout.write("\033[H\033[J")
    print(f"============================================================")
    print(f"   ENHANCED FORMATION CONTROLLER | Type: {formation_type}")
    print(f"============================================================")
    print(f" Time: {sim_time:.2f} s  |  Robots: {len(robot_data)}")
    print(f" Path: CIRCLE (Radius: {PATH_RADIUS}m)")
    print(f" Reference Yaw: {ref_yaw:.2f} rad  |  Actual Yaw: {fit_yaw:.2f} rad")
    print(f" Turn Rate: {omega:.3f} rad/s")
    print(f"------------------------------------------------------------")
    print(f" {'ID':<4} | {'CMD_VEL':<8} | {'CMD_STR':<8} | {'ERR_LONG':<8} | {'ERR_LAT'}")
    print(f"------------------------------------------------------------")
    
    ids = sorted(robot_data.keys(), reverse=True)
    for r in ids:
        d = robot_data[r]
        role = "(L)" if r == ids[0] else ""
        print(f" R{r:<3}{role:<2}| {d['vel']:<8.3f} | {d['steer']:<8.3f} | {d['elong']:<8.3f} | {d['elat']:.3f}")
    print(f"------------------------------------------------------------")

# =========================================================
# 5. MAIN LOOP
# =========================================================

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}.")
        return

    num_robots = model.nu // n_ctrl_per_robot
    print(f"Detected {num_robots} robots. Starting Enhanced Formation Control...")

    # --- VS SETUP ---
    vs_nodes_raw = get_formation_nodes(FORMATION_TYPE, num_robots)
    vs_nodes_target = vs_nodes_raw[::-1] # Leader at front
    
    # Map from Controller Index (0-11) to MuJoCo qpos/qvel Index
    joint_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    
    robot_phases = np.array([r * 0.5 for r in range(num_robots)]) 
    robot_desired_velocities = np.zeros(num_robots)
    robot_steering_cmds = np.zeros(num_robots)
    dash_data = {r: {'vel':0, 'steer':0, 'elong':0, 'elat':0} for r in range(num_robots)}

    # --- TRAJECTORY STATE ---
    path_initialized = False
    ref_vs_pos = np.zeros(2)
    ref_vs_yaw = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_time = time.time()
        dt = model.opt.timestep
        step_counter = 0

        while viewer.is_running() and time.time() - start_time < SIM_DURATION:
            step_start = time.time()
            sim_time = data.time
            walking_active = sim_time > 1.0
            step_counter += 1

            # -------------------------------------------
            # A. SENSE & STATE ESTIMATION
            # -------------------------------------------
            robot_states = []
            robot_positions = []
            
            for r in range(num_robots):
                q_start = r * n_qpos_per_robot
                pos = data.qpos[q_start : q_start + 3]
                quat = data.qpos[q_start + 3 : q_start + 7]
                yaw = quat_to_yaw(quat)
                robot_states.append({'pos': pos[:2], 'yaw': yaw})
                robot_positions.append(pos[:2])
            robot_positions = np.array(robot_positions)

            fit_pos, fit_yaw = fit_virtual_structure(robot_positions, vs_nodes_target)

            # -------------------------------------------
            # B. HIGH LEVEL: VIRTUAL STRUCTURE UPDATE
            # -------------------------------------------
            if walking_active:
                if not path_initialized:
                    ref_vs_pos = fit_pos.copy()
                    ref_vs_yaw = fit_yaw
                    path_initialized = True
                
                # Path Kinematics
                mission_vel = CALIB_VELOCITY
                mission_omega = (mission_vel / PATH_RADIUS) * PATH_DIRECTION
                
                ref_vs_yaw += mission_omega * dt
                ref_vs_pos += np.array([
                    np.cos(ref_vs_yaw) * mission_vel * dt,
                    np.sin(ref_vs_yaw) * mission_vel * dt
                ])

                # Calculate Errors for each robot
                for r in range(num_robots):
                    node = vs_nodes_target[r]
                    
                    c_ref, s_ref = np.cos(ref_vs_yaw), np.sin(ref_vs_yaw)
                    target_x = (node[0] * c_ref - node[1] * s_ref) + ref_vs_pos[0]
                    target_y = (node[0] * s_ref + node[1] * c_ref) + ref_vs_pos[1]
                    
                    rx, ry = robot_states[r]['pos']
                    ryaw = robot_states[r]['yaw']
                    
                    ex_global = target_x - rx
                    ey_global = target_y - ry
                    
                    e_long = ex_global * np.cos(ryaw) + ey_global * np.sin(ryaw)
                    e_lat  = -ex_global * np.sin(ryaw) + ey_global * np.cos(ryaw)
                    e_head = wrap_angle(ref_vs_yaw - ryaw)
                    
                    # Compute Command Velocities
                    v_cmd = mission_vel + VS_GAIN_P * e_long
                    steer_cmd = VS_GAIN_PSI * e_head + VS_GAIN_P * e_lat
                    
                    v_cmd = np.clip(v_cmd, 0.0, 1.5)
                    steer_cmd = np.clip(steer_cmd, -1.0, 1.0)
                    
                    robot_desired_velocities[r] = v_cmd
                    robot_steering_cmds[r] = steer_cmd
                    dash_data[r] = {'vel': v_cmd, 'steer': steer_cmd, 'elong': e_long, 'elat': e_lat}
            else:
                mission_omega = 0.0

            # -------------------------------------------
            # C. LOW LEVEL: ENHANCED PHYSICS CONTROLLER
            # -------------------------------------------
            for r in range(num_robots):
                v_cmd = robot_desired_velocities[r]
                steer_cmd = robot_steering_cmds[r]
                
                # Variable Phase Integration (Formation Logic)
                freq_cmd = max(0.0, CALIB_FREQ * (v_cmd / CALIB_VELOCITY))
                
                if walking_active and v_cmd > 0.01:
                    robot_phases[r] += freq_cmd * 2 * np.pi * dt
                    robot_phases[r] %= (2 * np.pi)

                qpos_idx = r * n_qpos_per_robot
                qvel_idx = r * n_qvel_per_robot
                ctrl_idx = r * n_ctrl_per_robot

                # Get robot physics state for Enhanced Controller
                robot_joints_q = data.qpos[qpos_idx + 7 : qpos_idx + 19]
                robot_joints_v = data.qvel[qvel_idx + 6 : qvel_idx + 18]
                robot_vel_x = data.qvel[qvel_idx + 0]   # Body Fwd Velocity
                body_yaw_rate = data.qvel[qvel_idx + 5] # Body Yaw Rate

                for i in range(12):
                    # Generate Target (Formation Logic meets Walk Logic)
                    if walking_active:
                        target_angle, leg_phase = get_gait_target_enhanced(
                            robot_phases[r], i, steer_cmd, body_yaw_rate
                        )
                    else:
                        target_angle = STAND_BASE[i % 3]
                        leg_phase = 0.0

                    joint_idx = joint_map[i]
                    curr_q = robot_joints_q[joint_idx]
                    curr_v = robot_joints_v[joint_idx]

                    # Compute Torque (Enhanced Physics)
                    torque = enhanced_pd_controller(
                        target_angle, curr_q, curr_v, KP_BASE, KD_BASE,
                        i, leg_phase, robot_vel_x
                    )
                    
                    data.ctrl[ctrl_idx + i] = torque

            if step_counter % 50 == 0 and walking_active:
                print_vs_dashboard(sim_time, FORMATION_TYPE, dash_data, ref_vs_yaw, fit_yaw, mission_omega)

            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next = dt - (time.time() - step_start)
            if time_until_next > 0: time.sleep(time_until_next)

if __name__ == "__main__":
    main()