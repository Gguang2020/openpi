import copy
import os
import cv2
import sys
import h5py
import torch
import rospy
from sensor_msgs.msg import CompressedImage
import pickle
import threading
import numpy as np
from core.astribot_api.astribot_client import Astribot
from robotics_library_py.robotics_library_py import KinematicsTrans as tf
import argparse
import re

import dataclasses
import pathlib
import logging
import numpy as np

import matplotlib

matplotlib.use('Agg')  # 设置后端为 Agg，避免图形界面冲突
import h5py
import time

import tyro

# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi_client import websocket_client_policy as _websocket_client_policy

# from meta.whole_body_control.script.whole_body_control import *

# from utils import set_seed

sys.path.append('/home/xc/work/astribot_il/astribot_control')
from trajectory_fusion import TrajectoryFusion


@dataclasses.dataclass
class Args:
    out_path: pathlib.Path = pathlib.Path("out.mp4")
    # dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/hand_right_test/722/"
    # dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/hand_right_test/724/"
    # dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/hand_right_test/725/"
    # dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/hand_right_test/728/"
    # dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/hand_right_test/729/"
    # dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/gripper_pnp/729_left_1toy/"
    dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/gripper_pnp/to_Cart/731_left_1toy/"


    dataset_repo_id: str = "lerobot_so3_data_30hz"

    # config: str = "hand_right_pnp"
    # config: str = "hand_right_pnp_hight"
    # config: str = "hand_right_pnp_hight_trans_hand_cam"
    # config: str = "hand_right_pnp_hight_trans_hand_cam_list"
    config: str = "gripper_pnp_list"

    so3_data_len: int = 36

    seed: int = 0

    action_horizon: int = 5

    host: str = "0.0.0.0"
    # host: str = "10.11.12.105"
    # host: str = "10.11.5.2"
    port: int = 8000

    # host: str = "10.11.12.134"
    # port: int = 8002


    display: bool = False
    env: str = "ALOHA_SIM"
    image_from_data: bool = False
    infer_step: int = 18
    run_skip_frame: int = 3
    data_idx: int = 0

    use_mask: bool = True if "mask" in config else False
    control_right_arm_only: bool = False
    torso_control: bool = True
    use_trajectory_fusion: bool = False

    image_from_s1_topic: bool = True

    traj_frequency = 30.0
    update_frequency = traj_frequency / infer_step
    traj_length = 50
    only_use_high_cam = True if "_onlyheadimage" in config else False
    high_cam_name = "Bolt"  # "Stereo_left"  "Bolt" or "Stereo"
    cam_coordinate = None if "_delta_head" not in config else "head_so3_poses"  # 'self' 'world' 'head_so3_poses' 'head_quat_poses'


def parse_args() -> Args:
    return Args()


def load_h5py(file_path):
    def read_group(group):
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


def get_hdf5_files_with_data_id(folder_path, data_id):
    pattern = f"_({data_id})\.hdf5$"
    for f in os.listdir(folder_path):
        if re.search(pattern, f):
            return f
    return None


def load_data_dict(folder_path, data_id):
    hdf5_file = get_hdf5_files_with_data_id(folder_path, data_id)
    file_path = os.path.join(folder_path, hdf5_file)
    data_dict = load_h5py(file_path)
    return data_dict


def xyzquat_2_xyzso3(xyzquat):
    matrix = tf.TransformQuaternionPose2RTMatrix(np.array(xyzquat))
    xyzso3 = list(matrix[:3, 3]) + list(matrix[0, :3]) + list(matrix[1, :3])
    return xyzso3


def xyzquat_2_xyzso3_list(xyzquat_list):
    xyzso3_list = []
    for xyzquat in xyzquat_list:
        if len(xyzquat) == 7:
            xyzso3 = xyzquat_2_xyzso3(xyzquat)
        else:
            xyzso3 = xyzquat
        xyzso3_list.append(np.array(xyzso3))

    return np.concatenate(xyzso3_list)


def xyzso3_2_xyzquat(xyzso3):
    matrix = np.eye(4)
    matrix[:3, 3] = xyzso3[:3]
    matrix[0, :3] = xyzso3[3:6] / np.linalg.norm(xyzso3[3:6])
    matrix[1, :3] = xyzso3[6:9] / np.linalg.norm(xyzso3[6:9])
    matrix[2, :3] = np.cross(matrix[0, :3], matrix[1, :3])
    matrix[1, :3] = np.cross(matrix[2, :3], matrix[0, :3])
    quat = tf.TransformRTMatrix2Quaternion(matrix)

    # 拼接 xyz 和四元数，并转为一维
    xyzquat = np.concatenate((xyzso3[:3], np.concatenate(quat)))
    return xyzquat


def arm_grip_xyzso3_list_2_xyzquat_list(xyzso3_list, dim_list=[9, 1, 9, 1]):
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
    def __init__(self, astribot, use_topic=False, cameras_info=None):
        self.use_topic = use_topic
        self.astribot = astribot
        self.timeout = 0.3  # 秒

        if use_topic:
            self.head_image = None
            self.left_image = None
            self.right_image = None

            self.last_head_time = 0
            self.last_left_time = 0
            self.last_right_time = 0
            rospy.Subscriber('/astribot_camera/head_rgbd/color_compress/compressed',
                             CompressedImage, self.head_callback)
            rospy.Subscriber('/astribot_camera/left_wrist_rgbd/color_compress/compressed',
                             CompressedImage, self.left_callback)
            rospy.Subscriber('/astribot_camera/right_wrist_rgbd/color_compress/compressed',
                             CompressedImage, self.right_callback)
        else:

            astribot.activate_camera(cameras_info)

    def head_callback(self, msg):
        self.head_image = self.convert_compressed_image(msg)
        self.last_head_time = time.time()

    def left_callback(self, msg):
        self.left_image = self.convert_compressed_image(msg)
        self.last_left_time = time.time()

    def right_callback(self, msg):
        self.right_image = self.convert_compressed_image(msg)
        self.last_right_time = time.time()

    def convert_compressed_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image

    def get_rgb_images_dict(self):
        if self.use_topic:

            now = time.time()
            time_out_flag = False
            # 检查超时
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
                # self.left_image = self.right_image.copy()
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


def main(args) -> None:
    infer_step = args.infer_step
    astribot = Astribot(high_control_rights=True)

    control_frequency = 250.0
    eval_frequency = 30.0
    if args.use_trajectory_fusion:
        fusion = TrajectoryFusion(args.traj_frequency, args.update_frequency, args.traj_length, mode="act")

    if args.use_mask:
        sys.path.append('/home/xc/work/astribot_il/dataset/tools/gripper_render/')
        from gripper_render_sdk import RenderGripper
        #
        render_gripper = RenderGripper(pose_type='joints')

    # astribot.set_head_follow_effector(True)

    if "so3" in args.config:
        astribot.set_head_follow_effector(False)
    cameras_info = {
        'left_D405': {'flag_getdepth': False, 'flag_getIR': False},
        'right_D405': {'flag_getdepth': False, 'flag_getIR': False},
        'Bolt': {'flag_getdepth': False},
        "Stereo": {}
    }
    rgbd_get = RGBDRead(astribot, args.image_from_s1_topic, cameras_info)

    # for i in range(10):
    #     rgb =  rgbd_get.get_rgb_images_dict()
    #     time.sleep(0.1)

    # astribot.move_to_home()

    folder_path = args.dataset_root_path + '_'.join(args.dataset_repo_id.split('_')[1:])
    data_idx = args.data_idx
    data_dict = load_data_dict(folder_path, data_idx)

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    # episode_id = 0
    # dataset = LeRobotDataset(root=Args.dataset_root_path + Args.dataset_repo_id,
    #                          repo_id=Args.dataset_repo_id,
    #                          episodes=[episode_id],
    # )

    if args.control_right_arm_only:
        astribot.set_head_follow_effector(False)
    joint_action_init = data_dict['joints_dict']['joints_position_command'][0][-22:]
    print("joint_action_init:", joint_action_init)
    joint_action_init[-2] -= 0.0
    joint_action_init[-1] -= 0.00
    joint_action_init[0] += 0.00
    joint_action_init[2] += 0.00
    joint_action_init[12] += 0.0
    joint_names = astribot.whole_body_names[1:]
    joint_action = [
        joint_action_init[:4].tolist(),
        joint_action_init[4:11].tolist(),
        joint_action_init[11:12].tolist(),
        joint_action_init[12:19].tolist(),
        joint_action_init[19:20].tolist(),
        joint_action_init[20:22].tolist()
    ]
    astribot.move_joints_position(joint_names, joint_action, duration=4.0, use_wbc=False)

    # pi0_eval.eval_init()
    action_save_npy = []
    current_joints_position = astribot.get_current_joints_position()

    desired_cartesian_pose_init = astribot.get_current_cartesian_pose()
    astribot.set_head_follow_effector(True)

    idx = 0
    duration_time = 5.0

    if not args.image_from_data:
        data_len = 100000
    else:
        data_len = len(data_dict['time'])
    while idx < data_len:

        rgb_dict = rgbd_get.get_rgb_images_dict()
        if rgb_dict is None:
            exit()
        # time.sleep(0.09)

        if args.image_from_data:
            data_state = data_dict['cartesian_so3_dict']['cartesian_pose_state'][idx]
            data_action = data_dict['cartesian_so3_dict']['cartesian_pose_command'][idx]
        real_state = astribot.get_current_cartesian_pose()
        if args.so3_data_len == 36:
            real_joint_state = astribot.get_current_joints_position()
            real_state = xyzquat_2_xyzso3_list(real_state[1:6] + [real_joint_state[-1]])
        elif args.so3_data_len == 24:
            real_state = xyzquat_2_xyzso3_list(real_state[2:6])
        else:
            print("wrong so3_data_len")
        image_dict = {}
        if args.image_from_data:

            if "Stereo" in args.high_cam_name.split("_"):
                high_cam_name = "stereo"
            else:
                high_cam_name = args.high_cam_name

            if args.use_mask:
                image_dict['cam_high'] = np.transpose(data_dict['images_dict']['head_gripper_mask_2']['rgb'][idx],
                                                      (2, 0, 1))[[2, 1, 0], :, :] / 255.0
            else:
                head_image = data_dict['images_dict'][high_cam_name]['rgb'][idx]
                if "Stereo_left" == args.high_cam_name:
                    head_image = head_image[:, :640, :]
                image_dict['cam_high'] = np.transpose(head_image, (2, 0, 1))[[2, 1, 0], :, :] / 255.0
            if not args.only_use_high_cam:
                image_dict['cam_left_wrist'] = np.transpose(data_dict['images_dict']['left']['rgb'][idx], (2, 0, 1))[
                                               [2, 1, 0], :, :] / 255.0
                image_dict['cam_right_wrist'] = np.transpose(data_dict['images_dict']['right']['rgb'][idx], (2, 0, 1))[
                                                [2, 1, 0], :, :] / 255.0

            # Step 1: 转换格式为 (384, 384, 3) -> OpenCV 格式
            image_hwc = np.transpose(image_dict['cam_high'], (1, 2, 0))  # (H, W, C)

            # 假设 image_hwc 的值在 [0, 1] 范围内
            image_hwc_uint8 = (image_hwc * 255).astype(np.uint8)

            # 显示图像
            cv2.imwrite("output_image_1.png", image_hwc_uint8)  # 保存图片
        else:

            # rgb_dict, depth_dict, ir_dict ,_= astribot.get_images_dict()

            if "Stereo" in args.high_cam_name.split("_"):
                high_cam_name = "Stereo"
            else:
                high_cam_name = args.high_cam_name

            head_image = rgb_dict[high_cam_name]
            if "Stereo_left" == args.high_cam_name:
                head_image = head_image[:, :640, :]

            if args.use_mask:
                if args.image_from_data:
                    joint_state = copy.deepcopy(data_dict['joints_dict']['joints_position_state'][idx][:])
                else:
                    joint_state = astribot.get_current_joints_position()

                # cv2.imwrite("head_image_1.png", head_image)  # 保存图片
                # head_image = cv2.imread("head_image_1.png")
                # delay_idx = 3

                chassis_pose = joint_state[0]
                torso_pose = joint_state[1]
                left_arm_pose = joint_state[2]
                left_gripper_pose = joint_state[3]
                right_arm_pose = joint_state[4]
                right_gripper_pose = joint_state[5]
                head_pose = joint_state[6]
                new_head_image, dict_pose, mask_image = render_gripper.render_image(
                    head_pose, torso_pose, left_arm_pose,
                    right_arm_pose, left_gripper_pose, right_gripper_pose,
                    head_image,
                    kind='head',
                    mode=2
                )
                image_dict['cam_high'] = np.transpose(new_head_image, (2, 0, 1))[[2, 1, 0], :, :] / 255.0
            else:
                image_dict['cam_high'] = np.transpose(head_image, (2, 0, 1))[[2, 1, 0], :, :] / 255.0
            if not args.only_use_high_cam:
                image_dict['cam_left_wrist'] = np.transpose(rgb_dict['left_D405'], (2, 0, 1))[[2, 1, 0], :, :] / 255.0
                image_dict['cam_right_wrist'] = np.transpose(rgb_dict['right_D405'], (2, 0, 1))[[2, 1, 0], :, :] / 255.0

        # Step 1: 转换格式为 (384, 384, 3) -> OpenCV 格式
        image_hwc = np.transpose(image_dict['cam_right_wrist'], (1, 2, 0))  # (H, W, C)

        # 假设 image_hwc 的值在 [0, 1] 范围内
        image_hwc_uint8 = (image_hwc * 255).astype(np.uint8)

        # 显示图像
        cv2.imwrite("/home/xc/work/openpi/examples/s1/astribot_eval_sdk_head_image.png", image_hwc_uint8)  # 保存图片

        obs_dict = {}
        obs_dict["images"] = image_dict
        obs_dict["state"] = real_state
        # obj_name = "eggplant"
        # obj_name = "banana"
        obj_name = "corn"
        # obj_name = "eggplant"
        # obj_name = "Steamed bun"

        # text = 'pick ' + obj_name + ' into the box' + '\0'
        # text_list = [ord(char) for char in text]
        # obs_dict["prompt"] = text_list
        obs_dict["prompt"] = "S1 pick the obj and place it into the box."

        if args.cam_coordinate == "head_so3_poses":
            real_cartesian_pose = astribot.get_current_cartesian_pose()
            head_pose = xyzquat_2_xyzso3(real_cartesian_pose[-1])
            obs_dict["head_so3_poses"] = head_pose

        time_1 = time.time()
        # if np.isnan(obs_dict["state"][18]):
        #     obs_dict["state"][18] = 0.0
        # if np.isnan(obs_dict["state"][28]):
        #     obs_dict["state"][28] = 0
        infer_action_list = policy.infer(obs_dict)['actions']
        print("infer time:", time.time() - time_1)

        if args.use_trajectory_fusion:
            fusion.add_trajectory(infer_action_list)
            infer_delay_time = 0.05
        for loop_idx in range(0, infer_step, args.run_skip_frame):

            if args.use_trajectory_fusion:
                fuse_time = (idx + loop_idx) / args.traj_frequency + infer_delay_time
                infer_action = fusion.fuse(fuse_time=fuse_time)

            else:
                infer_action = infer_action_list[loop_idx + 1]


            if infer_action.shape[0] == 31:
                if not args.torso_control:

                    so3_action_init = xyzquat_2_xyzso3_list(desired_cartesian_pose_init[1:6])
                    # so3_action_init = data_dict['cartesian_so3_dict']['cartesian_pose_command'][0][:]
                    infer_action = np.concatenate([so3_action_init[:9], infer_action[9:]])

                if args.control_right_arm_only:
                    so3_action_init = xyzquat_2_xyzso3_list(desired_cartesian_pose_init[1:6])
                    # so3_action_init = data_dict['cartesian_so3_dict']['cartesian_pose_command'][0][:]
                    infer_action = np.concatenate([so3_action_init[:19], infer_action[19:]])
                # print(infer_action)
                pose_action = arm_grip_xyzso3_list_2_xyzquat_list(infer_action, dim_list=[9, 9, 1, 9, 1, 2])[:-1]
                pose_names = astribot.whole_body_names[1:6]
                action_save_npy.append(pose_action)
                astribot.move_cartesian_pose(pose_names, pose_action, duration=duration_time, use_wbc=True)
            #
            # elif args.control_right_arm_only:
            #     infer_action = np.concatenate([desired_cartesian_pose_init[1], infer_action[6:]])
            #     so3_action_init = data_dict['cartesian_so3_dict']['cartesian_pose_command'][0][:]
            #     infer_action = np.concatenate([so3_action_init[:10], infer_action[10:]])
            #     pose_action = arm_grip_xyzso3_list_2_xyzquat_list(infer_action)
            #     pose_names = astribot.whole_body_names[1:6]
            #     pose_action = [desired_cartesian_pose_init[1]]+ pose_action
            #     astribot.move_cartesian_pose(pose_names, pose_action,duration=duration_time, use_wbc=True)

            elif infer_action.shape[0] == 20:
                so3_action_init = data_dict['cartesian_so3_dict']['cartesian_pose_command'][0][:]
                if args.control_right_arm_only:
                    infer_action = np.concatenate([so3_action_init[9:19], infer_action[10:]])
                infer_action = np.concatenate([so3_action_init[:9], infer_action[:]])
                pose_action = arm_grip_xyzso3_list_2_xyzquat_list(infer_action, dim_list=[9, 9, 1, 9, 1, 2])[:-1]
                pose_names = astribot.whole_body_names[1:6]
                astribot.move_cartesian_pose(pose_names, pose_action, duration=duration_time, use_wbc=True)
            pass

            duration_time = 0.5
        idx += infer_step

        # if idx >= 150:
        #     np.save("action_save_npy.npy",  np.array(action_save_npy, dtype=object))

        print(f"Frame {idx}:")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()

    main(args)
