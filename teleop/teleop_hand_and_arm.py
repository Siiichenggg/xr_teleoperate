import time
import argparse
import csv
import numpy as np
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.payload_estimator import PayloadEstimatorG1_29
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


# ==================== XR_TELEOPERATE PATCH START ====================
# Relative to upstream: the helper block below was added for the G1_29 payload
# estimator validation chain and the conservative closed-loop mass writeback.
# ===================== XR_TELEOPERATE PATCH END =====================
PAYLOAD_EST_CSV_HEADER = [
    "timestamp",
    "loop_idx",
    "arm",
    "payload_side",
    "mass_hat",
    "mode",
    "reason",
    "dq_active_max",
    "tau_nominal_max",
    "tau_est_max",
    "tau_residual_max",
    "tau_unit_max",
    "mass_raw",
]


def _append_payload_est_csv_row(csv_path: str, row: dict) -> None:
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAYLOAD_EST_CSV_HEADER)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def _format_payload_est_main_log(loop_idx: int, snapshot: dict, closed_loop: dict = None) -> str:
    msg = (
        "[payload_est_main] "
        f"loop={loop_idx} "
        f"side={snapshot['payload_side']} "
        f"mass_hat={float(snapshot['mass_hat']):.5f} "
        f"mode={snapshot['mode']} "
        f"reason={snapshot['reason']} "
        f"dq_max={float(snapshot['dq_active_max']):.5f} "
        f"tau_nom_max={float(snapshot['tau_nominal_max']):.5f} "
        f"tau_est_max={float(snapshot['tau_est_max']):.5f} "
        f"tau_res_max={float(snapshot['tau_residual_max']):.5f} "
        f"tau_unit_max={float(snapshot['tau_unit_max']):.5f} "
        f"mass_raw={float(snapshot['mass_raw']):.5f}"
    )
    if closed_loop is not None:
        msg += (
            f" mass_cmd={float(closed_loop['mass_cmd']):.5f} "
            f"mass_target={float(closed_loop['target_mass']):.5f} "
            f"cl_active={bool(closed_loop['tracking_active'])} "
            f"cl_reason={closed_loop['reason']} "
            f"streak={int(closed_loop['valid_streak'])}"
        )
    return msg


def _make_payload_closed_loop_state(initial_mass: float, alpha: float, max_step: float, min_valid_updates: int) -> dict:
    return {
        "enabled": True,
        "mass_cmd": float(initial_mass),
        "target_mass": float(initial_mass),
        "alpha": float(np.clip(alpha, 0.0, 1.0)),
        "max_step": max(0.0, float(max_step)),
        "min_valid_updates": max(1, int(min_valid_updates)),
        "valid_streak": 0,
        "tracking_active": False,
        "reason": "init",
    }


def _update_payload_closed_loop_state(state: dict, snapshot: dict, mass_min: float, mass_max: float) -> dict:
    if snapshot is None:
        state["tracking_active"] = False
        state["reason"] = "freeze_no_snapshot"
        return state

    mass_hat = float(snapshot["mass_hat"])
    is_ok_update = (
        snapshot["mode"] == "update"
        and snapshot["reason"] == "ok"
        and bool(snapshot["is_valid"])
        and np.isfinite(mass_hat)
    )

    if is_ok_update:
        state["valid_streak"] += 1
    else:
        state["valid_streak"] = 0
        state["tracking_active"] = False
        state["reason"] = f"freeze_{snapshot['reason']}"
        return state

    if state["valid_streak"] < state["min_valid_updates"]:
        state["tracking_active"] = False
        state["reason"] = f"warmup_{state['valid_streak']}/{state['min_valid_updates']}"
        return state

    target_mass = float(np.clip(mass_hat, mass_min, mass_max))
    filtered_target = (1.0 - state["alpha"]) * state["mass_cmd"] + state["alpha"] * target_mass
    delta = float(np.clip(filtered_target - state["mass_cmd"], -state["max_step"], state["max_step"]))

    state["target_mass"] = target_mass
    state["mass_cmd"] = float(np.clip(state["mass_cmd"] + delta, mass_min, mass_max))
    state["tracking_active"] = True
    state["reason"] = "tracking"
    return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    parser.add_argument('--payload-enable', action='store_true', help='Enable static payload gravity compensation for G1_29 arm')
    parser.add_argument('--payload-side', type=str, choices=['left', 'right'], default='right', help='Active payload side for G1_29 payload compensation')
    parser.add_argument('--payload-mass', type=float, default=0.0, help='Payload mass in kg for G1_29 payload compensation')
    parser.add_argument('--payload-com', type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=('X', 'Y', 'Z'),
                        help='Payload CoM offset in active ee frame [m] for G1_29 payload compensation')
    parser.add_argument('--payload-scale', type=float, default=1.0, help='Scale applied to payload compensation torque for G1_29')
    parser.add_argument('--payload-debug', action='store_true', help='Enable payload compensation debug logs for G1_29')
    parser.add_argument('--payload-log-every', type=int, default=30, help='Log payload debug every N IK cycles for G1_29')
    parser.add_argument('--arm-tau-limit', type=float, default=None, help='Hard clip total arm feedforward torque before DDS send for G1_29')
    # ===== XR_TELEOPERATE PATCH: estimator / closed-loop CLI =====
    parser.add_argument('--payload-est-enable', action='store_true', help='Enable G1_29 payload mass estimator logging')
    parser.add_argument('--payload-est-debug', action='store_true', help='Enable G1_29 payload estimator debug logs')
    parser.add_argument('--payload-est-log-every', type=int, default=30, help='Log payload estimator output every N control cycles')
    parser.add_argument('--payload-est-csv', action='store_true', help='Write G1_29 payload estimator validation rows to CSV')
    parser.add_argument('--payload-est-csv-path', type=str, default='outputs/payload_est_log.csv', help='CSV path for G1_29 payload estimator validation logs')
    parser.add_argument('--payload-est-closed-loop', action='store_true', help='Enable G1_29 payload estimator closed-loop mass writeback')
    parser.add_argument('--payload-est-closed-loop-alpha', type=float, default=0.15, help='Low-pass alpha for G1_29 payload estimator closed-loop mass writeback')
    parser.add_argument('--payload-est-closed-loop-max-step', type=float, default=0.02, help='Maximum payload mass change [kg] per control cycle for G1_29 closed-loop mass writeback')
    parser.add_argument('--payload-est-closed-loop-min-valid-updates', type=int, default=10, help='Number of consecutive valid estimator updates before enabling G1_29 closed-loop mass writeback')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # ===== XR_TELEOPERATE PATCH: estimator runtime / closed-loop state =====
        payload_estimator = None
        payload_est_log_every = max(1, int(args.payload_est_log_every))
        payload_est_debug_enabled = bool(args.arm == "G1_29" and args.payload_est_enable and args.payload_est_debug)
        payload_est_csv_enabled = bool(args.arm == "G1_29" and args.payload_est_enable and args.payload_est_csv)
        payload_est_closed_loop_enabled = bool(args.arm == "G1_29" and args.payload_est_enable and args.payload_est_closed_loop)
        payload_est_closed_loop_state = None
        if args.payload_est_closed_loop and args.arm != "G1_29":
            logger_mp.warning("[teleop_hand_and_arm] payload estimator closed-loop is only supported on G1_29; disabling it.")
        if args.payload_est_closed_loop and not args.payload_est_enable:
            logger_mp.warning("[teleop_hand_and_arm] payload estimator closed-loop requires --payload-est-enable; disabling closed-loop.")
        # arm
        if args.arm == "G1_29":
            payload_enabled = bool(args.payload_enable or payload_est_closed_loop_enabled)
            if payload_est_closed_loop_enabled and not args.payload_enable:
                logger_mp.warning(
                    "[teleop_hand_and_arm] enabling payload compensation because payload estimator closed-loop is requested."
                )
            payload_cfg = {
                "enabled": payload_enabled,
                "side": args.payload_side,
                "mass": args.payload_mass,
                "com_ee": np.array(args.payload_com, dtype=float),
                "scale": args.payload_scale,
                "debug": args.payload_debug,
                "log_every": args.payload_log_every,
            }
            logger_mp.info(f"G1_29 payload_cfg: {payload_cfg}, arm_tau_limit={args.arm_tau_limit}")
            arm_ik = G1_29_ArmIK(payload_cfg=payload_cfg)
            arm_ctrl = G1_29_ArmController(
                motion_mode=args.motion,
                simulation_mode=args.sim,
                arm_tau_limit=args.arm_tau_limit,
            )
            if args.payload_est_enable:
                # ===== XR_TELEOPERATE PATCH: optional G1_29 payload estimator =====
                payload_estimator = PayloadEstimatorG1_29(
                    payload_side=args.payload_side,
                    payload_com_ee=np.array(args.payload_com, dtype=float),
                    debug=False,
                    log_every=payload_est_log_every,
                    mass_max=max(3.0, float(args.payload_mass)),
                )
                logger_mp.info("[teleop_hand_and_arm] G1_29 payload estimator enabled.")
                if payload_est_closed_loop_enabled:
                    # ===== XR_TELEOPERATE PATCH: closed-loop mass writeback state =====
                    payload_est_closed_loop_state = _make_payload_closed_loop_state(
                        initial_mass=float(payload_cfg["mass"]),
                        alpha=args.payload_est_closed_loop_alpha,
                        max_step=args.payload_est_closed_loop_max_step,
                        min_valid_updates=args.payload_est_closed_loop_min_valid_updates,
                    )
                    logger_mp.info(
                        "[teleop_hand_and_arm] G1_29 payload estimator closed-loop enabled: "
                        f"alpha={payload_est_closed_loop_state['alpha']}, "
                        f"max_step={payload_est_closed_loop_state['max_step']}, "
                        f"min_valid_updates={payload_est_closed_loop_state['min_valid_updates']}"
                    )
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        loop_idx = 0
        # main loop. robot start to follow VR user's motion
        while not STOP:
            loop_idx += 1
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            # ===== XR_TELEOPERATE PATCH: estimator observation assembly =====
            current_lr_arm_tau_est = arm_ctrl.get_current_dual_arm_tau_est() if args.arm == "G1_29" else None
            arm_obs = {
                "q": current_lr_arm_q,
                "dq": current_lr_arm_dq,
                "tau_est": current_lr_arm_tau_est,
            }
            mass_hat = None
            payload_est_snapshot = None
            payload_est_closed_loop_snapshot = None
            if payload_estimator is not None:
                mass_hat = payload_estimator.update(arm_obs)
                arm_obs["mass_hat"] = mass_hat
                payload_est_snapshot = payload_estimator.get_debug_snapshot()
                if payload_est_closed_loop_state is not None:
                    # ===== XR_TELEOPERATE PATCH: gated closed-loop mass writeback =====
                    payload_est_closed_loop_snapshot = _update_payload_closed_loop_state(
                        payload_est_closed_loop_state,
                        payload_est_snapshot,
                        payload_estimator.mass_min,
                        payload_estimator.mass_max,
                    )
                    arm_ik.payload_cfg["mass"] = float(payload_est_closed_loop_snapshot["mass_cmd"])

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # ===== XR_TELEOPERATE PATCH: estimator periodic log / CSV =====
            if payload_est_snapshot is not None and loop_idx % payload_est_log_every == 0:
                if payload_est_debug_enabled:
                    logger_mp.info(
                        _format_payload_est_main_log(
                            loop_idx,
                            payload_est_snapshot,
                            closed_loop=payload_est_closed_loop_snapshot,
                        )
                    )
                if payload_est_csv_enabled:
                    csv_row = {
                        "timestamp": f"{time.time():.6f}",
                        "loop_idx": loop_idx,
                        "arm": args.arm,
                        "payload_side": payload_est_snapshot["payload_side"],
                        "mass_hat": f"{float(payload_est_snapshot['mass_hat']):.6f}",
                        "mode": payload_est_snapshot["mode"],
                        "reason": payload_est_snapshot["reason"],
                        "dq_active_max": f"{float(payload_est_snapshot['dq_active_max']):.6f}",
                        "tau_nominal_max": f"{float(payload_est_snapshot['tau_nominal_max']):.6f}",
                        "tau_est_max": f"{float(payload_est_snapshot['tau_est_max']):.6f}",
                        "tau_residual_max": f"{float(payload_est_snapshot['tau_residual_max']):.6f}",
                        "tau_unit_max": f"{float(payload_est_snapshot['tau_unit_max']):.6f}",
                        "mass_raw": f"{float(payload_est_snapshot['mass_raw']):.6f}",
                    }
                    try:
                        _append_payload_est_csv_row(args.payload_est_csv_path, csv_row)
                    except Exception as e:
                        logger_mp.warning(f"[payload_est_main] failed to append csv row to {args.payload_est_csv_path}: {e}")

            payload_metrics = None
            if args.arm == "G1_29":
                payload_metrics = {
                    "payload": arm_ik.get_last_payload_debug(),
                }
                if mass_hat is not None:
                    payload_metrics["payload"]["mass_hat"] = float(mass_hat)
                if payload_est_snapshot is not None:
                    payload_metrics["payload"]["payload_estimator"] = payload_est_snapshot
                if payload_est_closed_loop_snapshot is not None:
                    # ===== XR_TELEOPERATE PATCH: closed-loop metrics recording =====
                    payload_metrics["payload"]["payload_estimator_closed_loop"] = {
                        "enabled": True,
                        "tracking_active": bool(payload_est_closed_loop_snapshot["tracking_active"]),
                        "reason": payload_est_closed_loop_snapshot["reason"],
                        "valid_streak": int(payload_est_closed_loop_snapshot["valid_streak"]),
                        "mass_cmd": float(payload_est_closed_loop_snapshot["mass_cmd"]),
                        "target_mass": float(payload_est_closed_loop_snapshot["target_mass"]),
                    }

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   current_lr_arm_dq[:7].tolist(),
                            "torque": [] if current_lr_arm_tau_est is None else current_lr_arm_tau_est[:7].tolist(),
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   current_lr_arm_dq[-7:].tolist(),
                            "torque": [] if current_lr_arm_tau_est is None else current_lr_arm_tau_est[-7:].tolist(),
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": sol_tauff[:7].tolist(),
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": sol_tauff[-7:].tolist(),
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state, metrics=payload_metrics)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, metrics=payload_metrics)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
