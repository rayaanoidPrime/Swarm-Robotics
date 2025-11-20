import mujoco
import mujoco.viewer
import numpy as np
import time
import sys

# =========================================================
# 1. CONFIGURATION (Merged from walk_test_2 & formation)
# =========================================================
XML_PATH = "scene.xml"
SIM_DURATION = 300.0

# --- FORMATION SELECTOR ---
FORMATION_TYPE = "TRIANGLE"  # Options: "LINE", "CIRCLE", "TRIANGLE"
TRIANGLE_SPACING = 1.5
CIRCLE_RADIUS = 2.0
LINE_DIST = 1.0

# --- FORMATION GAINS ---
GAIN_L1 = 2.0   # Neighbor Spring
GAIN_L2 = 2.0   # Second Neighbor Spring
GAIN_PSI = 1.5  # Bearing Control

# --- ENHANCED PHYSICS CONSTANTS (from walk_test_2) ---
KP_BASE = 60.0
KD_BASE = 3.0

# Joint-specific tuning
HIP_ABDUCT_KP, HIP_ABDUCT_KD = 0.8, 0.8
HIP_FLEX_KP, HIP_FLEX_KD = 1.0, 1.0
KNEE_KP, KNEE_KD = 1.4, 1.4

# Gravity Comp & Feedforward
GRAVITY_COMP = {'abduction': 0.0, 'thigh': 1.8, 'calf': 4.5}
STANCE_FF = {'abduction': 0.0, 'thigh': 2.5, 'calf': 5.5}
TORQUE_LIMITS = {'abduction': (-23.7, 23.7), 'hip_flexion': (-23.7, 23.7), 'knee': (-45.43, 45.43)}

# --- GAIT PARAMETERS ---
STAND_BASE = np.array([0.0, 0.9, -1.8])
CALIB_VELOCITY = 0.5 # Velocity at which frequency is 2.0Hz
BASE_WALK_FREQ = 2.0
SWING_AMP_THIGH = 0.3
SWING_AMP_CALF = 0.4

# Steering Tuning
STEERING_GAIN = 0.0
LATERAL_SHIFT_GAIN = 0.05
TURN_RATE_KP = 0.0
MAX_TURN_RATE = 0.0 # Limits internal to steering bias

# --- ROBOT DIMENSIONS ---
n_qpos_per_robot = 19
n_qvel_per_robot = 18
n_ctrl_per_robot = 12


# =========================================================
# 2. PHYSICS & GAIT HELPERS (The "Enhanced" Logic)
# =========================================================

def get_joint_type(actuator_idx):
    type_idx = actuator_idx % 3
    return ['abduction', 'thigh', 'calf'][type_idx]

def get_leg_side(actuator_idx):
    # 0-2: FR, 3-5: FL, 6-8: RR, 9-11: RL
    return 'left' if actuator_idx in [3, 4, 5, 9, 10, 11] else 'right'

def get_steering_bias(actuator_idx, target_yaw_rate, body_yaw_rate):
    """
    From walk_test_2: Calculates offsets to assist turning.
    """
    # Note: In formation control, target_yaw_rate is the steering command
    yaw_rate_error = target_yaw_rate - body_yaw_rate
    turn_command = target_yaw_rate + TURN_RATE_KP * yaw_rate_error
    
    amplitude_scale = 1.0 + (get_leg_side(actuator_idx) == 'left' and -1.0 or 1.0) * turn_command * STEERING_GAIN
    amplitude_scale = np.clip(amplitude_scale, 0.5, 1.5)
    
    abduction_offset = 0.0
    if actuator_idx % 3 == 0:  # Abduction joints
        side_sign = 1.0 if get_leg_side(actuator_idx) == 'left' else -1.0
        abduction_offset = side_sign * LATERAL_SHIFT_GAIN * turn_command
    
    return amplitude_scale, abduction_offset

def get_gait_target_enhanced(actuator_idx, current_phase, target_yaw_rate, body_yaw_rate):
    """
    Adapted from walk_test_2 to use externally accumulated phase (for variable speed).
    """
    # Determine leg phase offset
    # Pairs: (FR, RL) vs (FL, RR)
    is_pair_A = (actuator_idx < 3) or (actuator_idx >= 9)
    leg_phase = current_phase if is_pair_A else current_phase + np.pi
    
    # Get Steering adjustments
    amplitude_scale, abduction_offset = get_steering_bias(actuator_idx, target_yaw_rate, body_yaw_rate)
    
    joint_type = actuator_idx % 3
    base_angle = STAND_BASE[joint_type]
    
    # Apply CoM shift to abduction joints
    if joint_type == 0:
        base_angle += abduction_offset
    
    adjustment = 0.0
    if joint_type == 1:  # Thigh
        adjustment = -SWING_AMP_THIGH * np.sin(leg_phase) * amplitude_scale
    elif joint_type == 2:  # Calf
        swing_lift = np.cos(leg_phase)
        if swing_lift > 0:
            adjustment = -SWING_AMP_CALF * swing_lift * amplitude_scale
            
    return base_angle + adjustment, leg_phase

def enhanced_pd_controller(target_q, current_q, current_v, kp, kd, 
                          actuator_idx, leg_phase, robot_vel_x):
    """
    Direct copy from walk_test_2: Advanced PD with feedforward and damping.
    """
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

# =========================================================
# 3. FORMATION MATH (Unchanged logic)
# =========================================================

def quat_to_yaw(quat):
    w, x, y, z = quat
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)

def calculate_ll_control(my_pos, my_yaw, l1_pos, l2_pos, d1_target, d2_target):
    v1 = l1_pos - my_pos
    v2 = l2_pos - my_pos
    d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
    u1, u2 = v1 / (d1 + 1e-6), v2 / (d2 + 1e-6)
    e1, e2 = d1 - d1_target, d2 - d2_target
    
    vel_global = (GAIN_L1 * e1 * u1) + (GAIN_L2 * e2 * u2)
    c, s = np.cos(my_yaw), np.sin(my_yaw)
    target_speed = vel_global[0] * c + vel_global[1] * s
    target_lat   = -vel_global[0] * s + vel_global[1] * c
    steer_cmd = 2.0 * np.arctan2(target_lat, 1.0) 
    return target_speed, steer_cmd, d1, d2

def calculate_lpsi_control(my_pos, my_yaw, leader_pos, leader_yaw, dist_target, bearing_target):
    global_bearing = leader_yaw + bearing_target
    target_global_x = leader_pos[0] + dist_target * np.cos(global_bearing)
    target_global_y = leader_pos[1] + dist_target * np.sin(global_bearing)
    
    ex, ey = target_global_x - my_pos[0], target_global_y - my_pos[1]
    dist_error = np.sqrt(ex**2 + ey**2)
    
    c, s = np.cos(my_yaw), np.sin(my_yaw)
    v_surge, v_sway = ex * c + ey * s, -ex * s + ey * c
    
    cmd_speed = GAIN_PSI * v_surge 
    cmd_steer = 2.0 * np.arctan2(v_sway, 1.0)
    return cmd_speed, cmd_steer, dist_error

def build_triangle_topology(num_robots, spacing):
    topology = {}
    current_idx = num_robots - 1
    rows = [] 
    r_idx = 0
    
    while current_idx >= 0:
        count_in_row = r_idx + 1
        current_row = []
        for k in range(count_in_row):
            if current_idx < 0: break
            current_row.append(current_idx)
            current_idx -= 1
        rows.append(current_row)
        r_idx += 1
        
    leader_id = rows[0][0]
    topology[leader_id] = {'type': 'LEADER'}
    
    for r_i in range(1, len(rows)):
        curr_row, prev_row = rows[r_i], rows[r_i - 1]
        for k, bot_id in enumerate(curr_row):
            if k == 0: # Left Edge
                topology[bot_id] = {'type': 'LPSI', 'leader': prev_row[0], 
                                  'dist': np.sqrt(spacing**2 + (spacing/2.0)**2), 
                                  'bearing': np.arctan2(-spacing/2.0, -spacing)}
            elif k == len(curr_row) - 1: # Right Edge
                topology[bot_id] = {'type': 'LPSI', 'leader': prev_row[-1], 
                                  'dist': np.sqrt(spacing**2 + (spacing/2.0)**2), 
                                  'bearing': np.arctan2(spacing/2.0, -spacing)}
            else: # Internal
                topology[bot_id] = {'type': 'LL', 'l1': prev_row[k-1], 'l2': prev_row[k], 
                                  'd1': np.sqrt(spacing**2 + (spacing/2.0)**2), 
                                  'd2': np.sqrt(spacing**2 + (spacing/2.0)**2)}
    return topology, leader_id

# =========================================================
# 4. DASHBOARD & MAIN
# =========================================================

def print_dashboard(sim_time, formation_type, robot_data, leader_id):
    sys.stdout.write("\033[H\033[J")
    print(f"=== ENHANCED FORMATION DASHBOARD | Type: {formation_type} ===")
    print(f" Time: {sim_time:.2f} s")
    print(f"------------------------------------------------------------")
    print(f" {'ID':<4} | {'ROLE':<18} | {'VEL':<6} | {'STR':<6} | {'STATUS'}")
    print(f"------------------------------------------------------------")
    for r in sorted(robot_data.keys(), reverse=True):
        d = robot_data[r]
        color = "\033[92m" if r == leader_id else "\033[0m"
        print(f"{color} R{r:<3} | {d['role']:<18} | {d['vel']:.2f}   | {d['steer']:>5.2f}  | {d['error']} \033[0m")

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}.")
        return

    num_robots = model.nu // n_ctrl_per_robot
    print(f"Initializing {num_robots} robots in {FORMATION_TYPE} formation...")

    # --- SETUP TOPOLOGY ---
    formation_map = {}
    global_leader_id = num_robots - 1
    circle_d1, circle_d2, circle_bearing = 0, 0, 0
    
    if FORMATION_TYPE == "TRIANGLE":
        formation_map, global_leader_id = build_triangle_topology(num_robots, TRIANGLE_SPACING)
    elif FORMATION_TYPE == "CIRCLE" and num_robots >= 3:
        theta_step = 2 * np.pi / num_robots
        circle_d1 = 2 * CIRCLE_RADIUS * np.sin(theta_step / 2.0)
        circle_d2 = 2 * CIRCLE_RADIUS * np.sin(theta_step)
        vec_dx = CIRCLE_RADIUS * np.cos(-theta_step) - CIRCLE_RADIUS
        vec_dy = CIRCLE_RADIUS * np.sin(-theta_step)
        circle_bearing = np.arctan2(vec_dy, vec_dx)

    # Joint map (same as walk_test_2: FR, FL, RR, RL)
    joint_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    
    # State trackers
    robot_phases = np.array([r * 0.5 for r in range(num_robots)])
    robot_cmd_vel = np.zeros(num_robots)
    robot_cmd_steer = np.zeros(num_robots)
    dash_data = {r: {'role':'-', 'vel':0, 'steer':0, 'error':'-'} for r in range(num_robots)}

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

            # 1. SENSOR UPDATE
            robot_states = []
            for r in range(num_robots):
                q_start = r * n_qpos_per_robot
                qv_start = r * n_qvel_per_robot
                robot_states.append({
                    'pos': data.qpos[q_start:q_start+3],
                    'yaw': quat_to_yaw(data.qpos[q_start+3:q_start+7]),
                    'vx': data.qvel[qv_start], # Body forward velocity
                    'yaw_rate': data.qvel[qv_start+5] # Body yaw rate
                })

            # 2. FORMATION CONTROLLER (High Level)
            if walking_active:
                # Global Leader Logic
                robot_cmd_vel[global_leader_id] = 0.6
                robot_cmd_steer[global_leader_id] = 0.2 * np.sin(sim_time * 0.5) # Gentle weave
                dash_data[global_leader_id] = {'role': "LEADER", 'vel': robot_cmd_vel[global_leader_id], 'steer': robot_cmd_steer[global_leader_id], 'error': "N/A"}

                # Followers
                for r in range(num_robots - 1, -1, -1):
                    if r == global_leader_id: continue
                    
                    curr = robot_states[r]
                    tv, sc = 0.0, 0.0
                    
                    # -- Apply Topology --
                    if FORMATION_TYPE == "TRIANGLE":
                        spec = formation_map.get(r, {})
                        if spec.get('type') == 'LPSI':
                            lid = spec['leader']
                            tv, sc, err = calculate_lpsi_control(curr['pos'], curr['yaw'], robot_states[lid]['pos'], robot_states[lid]['yaw'], spec['dist'], spec['bearing'])
                            tv += robot_cmd_vel[lid]
                            dash_data[r] = {'role': f"L-Psi(R{lid})", 'vel': tv, 'steer': sc, 'error': f"D:{err:.2f}"}
                        elif spec.get('type') == 'LL':
                            l1, l2 = spec['l1'], spec['l2']
                            tv, sc, d1, d2 = calculate_ll_control(curr['pos'], curr['yaw'], robot_states[l1]['pos'], robot_states[l2]['pos'], spec['d1'], spec['d2'])
                            tv += (robot_cmd_vel[l1] + robot_cmd_vel[l2]) / 2.0
                            dash_data[r] = {'role': f"L-L(R{l1},R{l2})", 'vel': tv, 'steer': sc, 'error': f"E:{d1-spec['d1']:.1f}"}
                    
                    # Same CIRCLE/LINE logic as before (omitted for brevity, defaults to TRIANGLE if not changed)
                    elif FORMATION_TYPE == "LINE":
                         lid = r + 1
                         tv, sc, err = calculate_lpsi_control(curr['pos'], curr['yaw'], robot_states[lid]['pos'], robot_states[lid]['yaw'], LINE_DIST, np.pi)
                         tv += robot_cmd_vel[lid]
                         dash_data[r] = {'role': f"Follow(R{lid})", 'vel': tv, 'steer': sc, 'error': f"E:{err:.2f}"}

                    # Smooth & Clamp
                    ACCEL = 0.01
                    robot_cmd_vel[r] = np.clip(tv, robot_cmd_vel[r] - ACCEL, robot_cmd_vel[r] + ACCEL)
                    robot_cmd_vel[r] = np.clip(robot_cmd_vel[r], 0.0, 1.2)
                    robot_cmd_steer[r] = np.clip(sc, -0.6, 0.6)

            # 3. LOCOMOTION CONTROLLER (Low Level - Walk Test 2 Logic)
            for r in range(num_robots):
                v_cmd = robot_cmd_vel[r]
                yaw_cmd = robot_cmd_steer[r]
                
                # Variable frequency based on speed
                freq_cmd = BASE_WALK_FREQ * (v_cmd / CALIB_VELOCITY) if v_cmd > 0.1 else 0.0
                freq_cmd = np.clip(freq_cmd, 0.0, 3.0)
                
                if walking_active and v_cmd > 0.05:
                    robot_phases[r] += freq_cmd * 2 * np.pi * dt
                    robot_phases[r] %= (2 * np.pi)

                qpos_idx = r * n_qpos_per_robot
                qvel_idx = r * n_qvel_per_robot
                ctrl_idx = r * n_ctrl_per_robot

                # Robot specific data
                robot_q = data.qpos[qpos_idx + 7 : qpos_idx + 19]
                robot_v = data.qvel[qvel_idx + 6 : qvel_idx + 18]
                robot_vx = data.qvel[qvel_idx]
                robot_yaw_rate = data.qvel[qvel_idx + 5]

                for i in range(12):
                    # Calculate Target
                    target_angle, leg_phase = get_gait_target_enhanced(
                        i, robot_phases[r], yaw_cmd, robot_yaw_rate
                    )
                    
                    if not walking_active: 
                        target_angle = STAND_BASE[i%3]

                    joint_idx = joint_map[i]
                    
                    # Enhanced PD with Gravity Comp & Damping
                    torque = enhanced_pd_controller(
                        target_angle, robot_q[joint_idx], robot_v[joint_idx],
                        KP_BASE, KD_BASE,
                        i, leg_phase, robot_vx
                    )
                    
                    data.ctrl[ctrl_idx + i] = torque

            # 4. STEP & DISPLAY
            if step_counter % 50 == 0 and walking_active:
                print_dashboard(sim_time, FORMATION_TYPE, dash_data, global_leader_id)

            mujoco.mj_step(model, data)
            viewer.sync()
            
            remain = dt - (time.time() - step_start)
            if remain > 0: time.sleep(remain)

if __name__ == "__main__":
    main()