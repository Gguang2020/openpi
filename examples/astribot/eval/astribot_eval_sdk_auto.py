"""AstriBot evaluation script with SDK auto control.

This script performs robot evaluation using policy inference from a remote server.
It reads initial poses from an HDF5 dataset and executes actions predicted by
a policy model via websocket communication.
"""

import dataclasses
import logging
import os
import re
import time

import cv2
import h5py
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

from core.astribot_api.astribot_client import Astribot
from openpi_client import websocket_client_policy as _websocket_client_policy
from robotics_library_py.robotics_library_py import KinematicsTrans as tf


@dataclasses.dataclass
class Args:
    """Configuration arguments for AstriBot evaluation."""

    # Dataset configuration
    dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/gripper_pnp/to_Cart/731_left_1toy/"
    dataset_repo_id: str = "lerobot_so3_data_30hz"
    data_idx: int = 0

    # Network configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Robot control configuration
    infer_step: int = 18
    run_skip_frame: int = 3
    image_from_s1_topic: bool = True

    # Camera configuration
    camera_timeout: float = 0.3


def parse_args() -> Args:
    """Parse command line arguments.

    Returns:
        Args: Configuration arguments instance.
    """
    return Args()


def load_h5py(file_path: str) -> dict:
    """Load HDF5 file recursively.

    Args:
        file_path: Path to the HDF5 file.

    Returns:
        Dictionary containing the loaded data, or None if error occurs.
    """
    def read_group(group):
        """Recursively read HDF5 group."""
        content = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                content[key] = read_group(item)
            elif isinstance(item, h5py.Dataset):
                content[key] = np.array(item)
        return content

    try:
        with h5py.File(file_path, 'r') as f:
            data_dict = {}
            for key, item in f.items():
                if isinstance(item, h5py.Group):
                    data_dict[key] = read_group(item)
                elif isinstance(item, h5py.Dataset):
                    data_dict[key] = np.array(item)
            return data_dict
    except RuntimeError as e:
        print(f"Error reading HDF5 file {file_path}: {e}")
        return None


def get_hdf5_files_with_data_id(folder_path: str, data_id: int) -> str:
    """Find HDF5 file with specific data ID.

    Args:
        folder_path: Directory to search in.
        data_id: Data ID to match.

    Returns:
        Filename if found, None otherwise.
    """
    pattern = f"_({data_id})\.hdf5$"
    for f in os.listdir(folder_path):
        if re.search(pattern, f):
            return f
    return None


def load_data_dict(folder_path: str, data_id: int) -> dict:
    """Load data dictionary from HDF5 file with specific data ID.

    Args:
        folder_path: Directory containing HDF5 files.
        data_id: Data ID to load.

    Returns:
        Loaded data dictionary.
    """
    hdf5_file = get_hdf5_files_with_data_id(folder_path, data_id)
    file_path = os.path.join(folder_path, hdf5_file)
    data_dict = load_h5py(file_path)
    return data_dict


def xyzquat_2_xyzso3(xyzquat: np.ndarray) -> list:
    """Convert XYZ quaternion to XYZ SO3 representation.

    SO3 representation uses the first two rows of the rotation matrix instead of
    quaternions for better numerical stability in neural network training.

    Args:
        xyzquat: Array of [x, y, z, qw, qx, qy, qz].

    Returns:
        List of [x, y, z, r11, r12, r13, r21, r22, r23] (SO3 representation).
    """
    matrix = tf.TransformQuaternionPose2RTMatrix(np.array(xyzquat))
    # Extract position (xyz) and first two rows of rotation matrix
    xyzso3 = list(matrix[:3, 3]) + list(matrix[0, :3]) + list(matrix[1, :3])
    return xyzso3


def xyzquat_2_xyzso3_list(xyzquat_list: list) -> np.ndarray:
    """Convert list of XYZ quaternions to concatenated XYZ SO3 array.

    Args:
        xyzquat_list: List of quaternion poses (7-element arrays) or SO3 poses (9-element arrays).

    Returns:
        Concatenated array of SO3 representations.
    """
    xyzso3_list = []
    for xyzquat in xyzquat_list:
        if len(xyzquat) == 7:
            xyzso3 = xyzquat_2_xyzso3(xyzquat)
        else:
            xyzso3 = xyzquat
        xyzso3_list.append(np.array(xyzso3))

    return np.concatenate(xyzso3_list)


def xyzso3_2_xyzquat(xyzso3: np.ndarray) -> np.ndarray:
    """Convert XYZ SO3 to XYZ quaternion representation.

    Reconstructs the full rotation matrix from SO3 representation by computing
    the third row via cross product and orthonormalizing.

    Args:
        xyzso3: Array of [x, y, z, r11, r12, r13, r21, r22, r23] (SO3 representation).

    Returns:
        Array of [x, y, z, qw, qx, qy, qz].
    """
    matrix = np.eye(4)
    matrix[:3, 3] = xyzso3[:3]  # Position
    # Normalize first two rows of rotation matrix
    matrix[0, :3] = xyzso3[3:6] / np.linalg.norm(xyzso3[3:6])
    matrix[1, :3] = xyzso3[6:9] / np.linalg.norm(xyzso3[6:9])
    # Compute third row via cross product to ensure orthogonality
    matrix[2, :3] = np.cross(matrix[0, :3], matrix[1, :3])
    # Re-orthogonalize second row
    matrix[1, :3] = np.cross(matrix[2, :3], matrix[0, :3])
    quat = tf.TransformRTMatrix2Quaternion(matrix)

    xyzquat = np.concatenate((xyzso3[:3], np.concatenate(quat)))
    return xyzquat


def arm_grip_xyzso3_list_2_xyzquat_list(
    xyzso3_list: np.ndarray, dim_list: list = [9, 1, 9, 1]
) -> list:
    """Convert arm and gripper SO3 list to quaternion list.

    Args:
        xyzso3_list: Concatenated array of SO3 representations.
        dim_list: Dimensions to split the array (default: [9, 1, 9, 1]).

    Returns:
        List of quaternion poses.
    """
    xyzquat_list = []
    xyzso3_list = np.split(xyzso3_list, np.cumsum(dim_list))[:-1]
    for xyzso3 in xyzso3_list:
        if len(xyzso3) == 9:
            xyzquat = xyzso3_2_xyzquat(xyzso3).tolist()
        else:
            xyzquat = xyzso3.tolist()
        xyzquat_list.append(xyzquat)

    return xyzquat_list


class RGBDRead:
    """RGBD camera reader that supports both ROS topics and direct camera access."""

    def __init__(self, astribot: Astribot, use_topic: bool = False, cameras_info: dict = None, timeout: float = 0.3):
        """Initialize RGBD reader.

        Args:
            astribot: AstriBot instance.
            use_topic: If True, use ROS topics; otherwise use direct camera access.
            cameras_info: Camera configuration dictionary.
            timeout: Timeout in seconds for image reception.
        """
        self.use_topic = use_topic
        self.astribot = astribot
        self.timeout = timeout

        if use_topic:
            self.head_image = None
            self.left_image = None
            self.right_image = None

            self.last_head_time = 0
            self.last_left_time = 0
            self.last_right_time = 0

            rospy.Subscriber(
                '/astribot_camera/head_rgbd/color_compress/compressed',
                CompressedImage,
                self.head_callback
            )
            rospy.Subscriber(
                '/astribot_camera/left_wrist_rgbd/color_compress/compressed',
                CompressedImage,
                self.left_callback
            )
            rospy.Subscriber(
                '/astribot_camera/right_wrist_rgbd/color_compress/compressed',
                CompressedImage,
                self.right_callback
            )
        else:
            astribot.activate_camera(cameras_info)

    def head_callback(self, msg: CompressedImage) -> None:
        """Callback for head camera images."""
        self.head_image = self.convert_compressed_image(msg)
        self.last_head_time = time.time()

    def left_callback(self, msg: CompressedImage) -> None:
        """Callback for left wrist camera images."""
        self.left_image = self.convert_compressed_image(msg)
        self.last_left_time = time.time()

    def right_callback(self, msg: CompressedImage) -> None:
        """Callback for right wrist camera images."""
        self.right_image = self.convert_compressed_image(msg)
        self.last_right_time = time.time()

    @staticmethod
    def convert_compressed_image(msg: CompressedImage) -> np.ndarray:
        """Convert ROS CompressedImage to numpy array.

        Args:
            msg: ROS CompressedImage message.

        Returns:
            Image as numpy array in BGR format.
        """
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image

    def get_rgb_images_dict(self) -> dict:
        """Get RGB images from all cameras.

        Returns:
            Dictionary with camera names as keys and images as values,
            or None if timeout occurs.
        """
        if self.use_topic:
            now = time.time()
            time_out_flag = False

            if now - self.last_head_time > self.timeout:
                self.head_image = None
                time_out_flag = True
                print("\033[91mRGBDRead: Timeout, head image is None\033[0m")

            if now - self.last_right_time > self.timeout:
                self.right_image = None
                time_out_flag = True
                print("\033[91mRGBDRead: Timeout, right_image is None\033[0m")

            if now - self.last_left_time > self.timeout:
                self.left_image = None
                time_out_flag = True
                print("\033[91mRGBDRead: Timeout, left_image is None\033[0m")

            if time_out_flag:
                print("RGBDRead: Timeout, some images are None")
                return None
            else:
                return {
                    "Bolt": self.head_image,
                    "left_D405": self.left_image,
                    "right_D405": self.right_image,
                }
        else:
            rgb_dict, depth_dict, ir_dict, _ = self.astribot.get_images_dict()
            return rgb_dict


def main(args: Args) -> None:
    """Main evaluation loop.

    Args:
        args: Configuration arguments.
    """
    # Constants
    INITIAL_DURATION = 4.0
    FIRST_MOVE_DURATION = 5.0
    SUBSEQUENT_MOVE_DURATION = 0.5
    MAX_DATA_LEN = 100000

    # Initialize robot
    astribot = Astribot(high_control_rights=True)
    astribot.set_head_follow_effector(False)

    # Configure cameras
    cameras_info = {
        'left_D405': {'flag_getdepth': False, 'flag_getIR': False},
        'right_D405': {'flag_getdepth': False, 'flag_getIR': False},
        'Bolt': {'flag_getdepth': False},
        "Stereo": {}
    }
    rgbd_get = RGBDRead(astribot, args.image_from_s1_topic, cameras_info, args.camera_timeout)

    # Load dataset
    folder_path = args.dataset_root_path + '_'.join(args.dataset_repo_id.split('_')[1:])
    data_dict = load_data_dict(folder_path, args.data_idx)

    # Initialize policy
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # Move to initial position
    joint_action_init = data_dict['joints_dict']['joints_position_command'][0][-22:]
    print("joint_action_init:", joint_action_init)
    joint_names = astribot.whole_body_names[1:]
    joint_action = [
        joint_action_init[:4].tolist(),
        joint_action_init[4:11].tolist(),
        joint_action_init[11:12].tolist(),
        joint_action_init[12:19].tolist(),
        joint_action_init[19:20].tolist(),
        joint_action_init[20:22].tolist()
    ]
    astribot.move_joints_position(joint_names, joint_action, duration=INITIAL_DURATION, use_wbc=False)
    astribot.set_head_follow_effector(True)

    # Main evaluation loop
    idx = 0
    duration_time = FIRST_MOVE_DURATION
    data_len = MAX_DATA_LEN

    while idx < data_len:
        # Get RGB images
        rgb_dict = rgbd_get.get_rgb_images_dict()
        if rgb_dict is None:
            break

        # Get robot state
        real_state = astribot.get_current_cartesian_pose()
        real_joint_state = astribot.get_current_joints_position()
        real_state = xyzquat_2_xyzso3_list(real_state[1:6] + [real_joint_state[-1]])

        # Prepare image dictionary
        image_dict = {
            'cam_high': np.transpose(rgb_dict['Bolt'], (2, 0, 1))[[2, 1, 0], :, :] / 255.0,
            'cam_left_wrist': np.transpose(rgb_dict['left_D405'], (2, 0, 1))[[2, 1, 0], :, :] / 255.0,
            'cam_right_wrist': np.transpose(rgb_dict['right_D405'], (2, 0, 1))[[2, 1, 0], :, :] / 255.0
        }

        # Prepare observation dictionary
        obs_dict = {
            "images": image_dict,
            "state": real_state,
            "prompt": "S1 pick the obj and place it into the box."
        }

        # Inference
        time_start = time.time()
        infer_action_list = policy.infer(obs_dict)['actions']
        print(f"Inference time: {time.time() - time_start:.3f}s")

        # Execute actions
        for loop_idx in range(0, args.infer_step, args.run_skip_frame):
            infer_action = infer_action_list[loop_idx + 1]
            pose_action = arm_grip_xyzso3_list_2_xyzquat_list(infer_action, dim_list=[9, 9, 1, 9, 1, 2])[:-1]
            pose_names = astribot.whole_body_names[1:6]
            astribot.move_cartesian_pose(pose_names, pose_action, duration=duration_time, use_wbc=True)
            duration_time = SUBSEQUENT_MOVE_DURATION

        idx += args.infer_step
        print(f"Frame {idx}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    main(args)
