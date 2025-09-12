# ruff: noqa
"""
Script courtesy of Raziel90 https://github.com/huggingface/lerobot/pull/586/files

Example usage
python scripts/aloha_hd5.py --raw-path ~/data/ --dataset-repo-id <hf-username>/<dataset-name> --robot-type <aloha-stationary|aloha-mobile> --fps 50 --video-encoding=false --push=false

The data will be saved locally the value of the LEROBOT_HOME environment variable. By default this is set to ~/.cache/huggingface/lerobot
If you wish to submit the dataset to the hub, you can do so by setting up the hf cli https://huggingface.co/docs/huggingface_hub/en/guides/cli and setting --push=true
"""

import argparse
import copy
import logging
import os
from pathlib import Path
import shutil
import traceback
import numpy as np
import json
import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_stats

from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames_torchvision,
    encode_video_frames,
    get_video_info,
)
from collections import OrderedDict
import s1_hd5_2_lerobot
from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    EPISODES_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    append_jsonlines,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_branch,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    get_delta_indices,
    get_episode_data_index,
    get_features_from_robot,
    get_hf_features_from_features,
    get_hub_safe_version,
    hf_transform_to_torch,
    load_episodes,
    load_info,
    load_stats,
    load_tasks,
    serialize_dict,
    write_json,
    write_parquet,
)
import torch
import s1_hd5_2_lerobot
import pandas as pd
import pyarrow as pa
import sys
sys.path.append('/home/xc/work/astribot_il/dataset/tools/gripper_render/')
# from gripper_render_sdk import RenderGripper


def encode_video_frames_from_img(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 30,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: str | None = "error",
    overwrite: bool = False,
    image_format: str = ".png",
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_args = OrderedDict(
        [
            ("-f", "image2"),
            ("-r", str(fps)),
            ("-i", str(imgs_dir / f"frame_%06d{image_format}")),
            ("-vcodec", vcodec),
            ("-pix_fmt", pix_fmt),
        ]
    )

    if g is not None:
        ffmpeg_args["-g"] = str(g)

    if crf is not None:
        ffmpeg_args["-crf"] = str(crf)

    if fast_decode:
        key = "-svtav1-params" if vcodec == "libsvtav1" else "-tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        ffmpeg_args[key] = value

    if log_level is not None:
        ffmpeg_args["-loglevel"] = str(log_level)

    ffmpeg_args = [item for pair in ffmpeg_args.items() for item in pair]
    if overwrite:
        ffmpeg_args.append("-y")

    ffmpeg_cmd = ["ffmpeg"] + ffmpeg_args + [str(video_path)]
    # redirect stdin to subprocess.DEVNULL to prevent reading random keyboard inputs from terminal
    subprocess.run(ffmpeg_cmd, check=True, stdin=subprocess.DEVNULL)

    if not video_path.exists():
        raise OSError(
            f"Video encoding did not work. File not found: {video_path}. "
            f"Try running the command manually to debug: `{''.join(ffmpeg_cmd)}`"
        )

class CustomLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        **kwargs,  # 保持可扩展性
    ):
        super().__init__(repo_id, root, **kwargs)  # 调用父类的初始化
        # print(f"CustomLeRobotDataset initialized with new_param: {self.input_video_flag}")

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)

            if self.features[key]["dtype"] not in ["image", "video"]:
                item = frame[key].numpy() if isinstance(frame[key], torch.Tensor) else frame[key]
                self.episode_buffer[key].append(item)
            elif self.features[key]["dtype"] in ["image", "video"]:

                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                if frame[key].ndim == 1:
                    img_path = img_path.with_suffix(".jpg")
                    with open(img_path, "wb") as f:
                        f.write(frame[key].cpu().numpy().tobytes())
                        self.episode_buffer[key].append(str(img_path))
                else:
                    self._save_image(frame[key], img_path)
                    self.episode_buffer[key].append(str(img_path))

        self.episode_buffer["size"] += 1


    def save_episode(self, task: str, encode_videos: bool = True, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer. Note that since it affects files on
        disk, it sets self.consolidated to False to ensure proper consolidation later on before uploading to
        the hub.

        Use 'encode_videos' if you want to encode videos during the saving of this episode. Otherwise,
        you can do it later with dataset.consolidate(). This is to give more flexibility on when to spend
        time for video encoding.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        episode_length = episode_buffer.pop("size")
        episode_index = episode_buffer["episode_index"]
        if episode_index != self.meta.total_episodes:
            # TODO(aliberts): Add option to use existing episode_index
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        if episode_length == 0:
            raise ValueError(
                "You must add one or several frames with `add_frame` before calling `add_episode`."
            )

        task_index = self.meta.get_task_index(task)

        if not set(episode_buffer.keys()) == set(self.features):
            raise ValueError()

        for key, ft in self.features.items():
            if key == "index":
                episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                episode_buffer[key] = np.full((episode_length,), episode_index)
            elif key == "task_index":
                episode_buffer[key] = np.full((episode_length,), task_index)
            elif ft["dtype"] in ["image", "video"]:
                continue
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                episode_buffer[key] = np.stack(episode_buffer[key])
            else:
                raise ValueError(key)

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        self.meta.save_episode(episode_index, episode_length, task, task_index)
        if encode_videos and len(self.meta.video_keys) > 0:

            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

        self.consolidated = False
    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue

            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent

            file_suffix = None
            for file in img_dir.iterdir():
                if file.is_file():
                    if file.suffix == ".jpg":
                        file_suffix = ".jpg"
                    elif file.suffix == ".png":
                        file_suffix = ".png"

                if file_suffix is not None:
                    break
            if file_suffix == ".png":
                encode_video_frames_from_img(img_dir, video_path, self.fps, vcodec="h264", overwrite=True, image_format=".png")
            elif file_suffix == ".jpg":
                encode_video_frames_from_img(img_dir, video_path, self.fps, vcodec="h264", overwrite=True, image_format=".jpg")
            else:
                raise ValueError(f"No image files found in {img_dir}. Cannot encode video.")

        return video_paths


    def consolidate(self, run_compute_stats: bool = True, keep_image_files: bool = False) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        check_timestamps_sync(self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s)

        if len(self.meta.video_keys) > 0:
            self.encode_videos()
            self.meta.write_video_info()

        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        if run_compute_stats:
            self.stop_image_writer()
            # TODO(aliberts): refactor stats in save_episodes
            self.meta.stats = compute_stats(self, num_workers=0,max_num_samples=1000)
            serialized_stats = serialize_dict(self.meta.stats)
            write_json(serialized_stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )
            
class S1Transformer:

    joints_names = ["astribot_chassis", "astribot_torso", "astribot_arm_left", "astribot_gripper_left", "astribot_arm_right", "astribot_gripper_right", "astribot_head"]
    joints_dim = [3, 4, 7, 1, 7, 1, 2]
    joints_name_dim_dict = dict(zip(joints_names, joints_dim))

    def get_joint_data_form_name(self,joint_data, name):
        if name not in self.joints_name_dim_dict:
            return None  # 名称不存在，返回 None

        index = self.joints_names.index(name)  # 找到关节索引
        start = sum(self.joints_dim[:index])  # 计算起始索引
        end = start + self.joints_dim[index]  # 计算结束索引

        data = joint_data[start:end]  # 获取对应的 joint 数据
        if np.isnan(data[0]):
            data = np.zeros(self.joints_dim[index])  # 如果数据为 NaN，则返回全零数组

        return data  # 返回对应的 joint 数据

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to rotation matrix.
        Parameters
        ----------
        q : np.ndarray
            Quaternion (x, y, z, w).
        Returns
        -------
        np.ndarray
            Rotation matrix.
        """
        x, y, z, w = q
        x2, y2, z2, w2 = x * x, y * y, z * z, w * w
        xy, xz, xw, yz, yw, zw = x * y, x * z, x * w, y * z, y * w, z * w
        return np.array(
            [
                [w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw)],
                [2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw)],
                [2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2],
            ]
        )
    def xyz_quat_to_so3(self,xyz_quat):
        x, y, z, qx, qy, qz, qw = xyz_quat
        R = self.quaternion_to_rotation_matrix([qx, qy, qz, qw])


        return np.hstack([xyz_quat[:3], R[:2].flatten()])  # 返回 [x, y, z, R(3x3) 展平为 9 个数]


class S1HD5Extractor:
    TAGS = ["aloha", "robotics", "hdf5"]
    aloha_stationary = "aloha-stationary"
    aloha_mobile = "aloha-mobile"
    @staticmethod
    def get_cameras(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        list of str
            A list of keys corresponding to RGB cameras in the dataset.
        """

        rgb_cameras = []
        for key in hdf5_data["/images_dict/"]:
            if "rgb" in hdf5_data[f"/images_dict/{key}"]:
                rgb_cameras.append(key)
        # rgb_cameras = [key for key in hdf5_data["/images_dict/"] if "/rgb" not in key
        return rgb_cameras

    @staticmethod
    def check_format(episode_list: list[str] | list[Path],  image_compressed: bool = True):
        """
        Check the format of the given list of HDF5 files.
        Parameters
        ----------
        episode_list : list of str or list of Path
            List of paths to the HDF5 files to be checked.
        image_compressed : bool, optional
            Flag indicating whether the images are compressed (default is True).
        Raises
        ------
        ValueError
            If the episode_list is empty.
            If any HDF5 file is missing required keys '/joints_dict/joints_position_command' or '/joints_dict/joints_position_state'.
            If the '/joints_dict/joints_position_command' or '/joints_dict/joints_position_state' keys do not have 2 dimensions.
            If the number of frames in '/joints_dict/joints_position_command' and '/joints_dict/joints_position_state' keys do not match.
            If the number of frames in '/images_dict/{camera}/rgb' does not match the number of frames in '/joints_dict/joints_position_command' and '/joints_dict/joints_position_state'.
            If the dimensions of images do not match the expected dimensions based on the image_compressed flag.
            If uncompressed images do not have the expected (h, w, c) format.
        """

        if not episode_list:
            raise ValueError("No hdf5 files found in the raw directory. Make sure they are named 'episode_*.hdf5'")
        for episode_path in episode_list:
            print('check episode_path:',episode_path)
            with h5py.File(episode_path, "r") as data:
                if not all(key in data for key in ["/joints_dict/joints_position_command", "/joints_dict/joints_position_state"]):
                    raise ValueError(
                        episode_path + "Missing required keys in the hdf5 file. Make sure the keys '/joints_dict/joints_position_command' and '/joints_dict/joints_position_state' are present."
                    )

                if not data["/joints_dict/joints_position_command"].ndim == data["/joints_dict/joints_position_state"].ndim == 2:
                    raise ValueError("The '/joints_dict/joints_position_command' and '/joints_dict/joints_position_state' keys should have both 2 dimensions.")

                if (num_frames := data["/joints_dict/joints_position_command"].shape[0]) != data["/joints_dict/joints_position_state"].shape[0]:
                    raise ValueError(
                        "The '/joints_dict/joints_position_command' and '/joints_dict/joints_position_state' keys should have the same number of frames."
                    )

                for camera in S1HD5Extractor.get_cameras(data):
                    if data[f"/images_dict/{camera}/rgb"].ndim > 1 and num_frames != data[f"/images_dict/{camera}/rgb"].shape[0]:
                        raise ValueError(
                            f"The number of frames in '/images_dict/{camera}/rgb' should be the same as in '/joints_dict/joints_position_command' and '/joints_dict/joints_position_state' keys."
                        )

                    expected_dims = 1 if image_compressed else 4
                    if data[f"/images_dict/{camera}/rgb"].ndim != expected_dims:
                        raise ValueError(
                            f"Expect {expected_dims} dimensions for {'compressed' if image_compressed else 'uncompressed'} images but {data[f'/images_dict/{camera}/rgb'].ndim} provided."
                        )
                    if not image_compressed:
                        b, h, w, c = data[f"/images_dict/{camera}/rgb"].shape
                        if not c < h and c < w:
                            raise ValueError(f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided.")

            print('check episode_path done')

        print('check episode_path done')
    @staticmethod
    def extract_episode_frames(
        episode_path: str | Path, features: dict[str, dict],image_compressed: bool,state_is_last_action:bool = True,label_dicts:dict = None
    ) -> list[dict[str, torch.Tensor]]:
        """
        Extract frames from an episode stored in an HDF5 file.
        Parameters
        ----------
        episode_path : str or Path
            Path to the HDF5 file containing the episode data.
        features : dict of str to dict
            Dictionary where keys are feature identifiers and values are dictionaries with feature details.
        image_compressed : bool
            Flag indicating whether the images are stored in a compressed format.
        Returns
        -------
        list of dict of str to torch.Tensor
            List of frames, where each frame is a dictionary mapping feature identifiers to tensors.
        """
        s1_trans = S1Transformer()
        #
        # render_gripper = RenderGripper(
        #          pose_type = 'cartesian')
        image_delay_frame = 3
        frames = []
        with h5py.File(episode_path, "r") as file:
            episode_idx = episode_path.stem.split("_")[-1]
            if label_dicts is not None:
                cur_label_dict = label_dicts.get(episode_idx, None)
                frame_range = None
                for temp_key in cur_label_dict.keys():
                    if temp_key == "invalid_data":
                        continue
                    elif temp_key == "other":
                        frame_range = range(cur_label_dict[temp_key]["frame_range"][0],cur_label_dict[temp_key]["frame_range"][1])
            else:
                frame_range = range(file["/joints_dict/joints_position_command"].shape[0])

            if frame_range is None:
                assert(f"Episode {episode_idx} has no label, using all frames.")

            for frame_idx in frame_range:
                frame = {}
                for feature_id in features:
                    if feature_id.replace(".","/") not in file.keys():
                        if "xyz_quat_state" in feature_id.split("."):
                            if state_is_last_action and "gripper" not in feature_id.split(".")[-1].split("_"):
                                feature_name_hd5 = "command_poses_dict/" + feature_id.split(".")[1]
                                state_frame_idx = max(0,frame_idx - 1 )
                                frame[feature_id] = torch.from_numpy(file[feature_name_hd5][state_frame_idx])
                            else:
                                # 计算xyz_quat_state
                                feature_name_hd5 = "poses_dict/" + feature_id.split(".")[1]
                                frame[feature_id] = torch.from_numpy(file[feature_name_hd5][frame_idx])
                        elif "xyz_quat_cmd" in feature_id.split("."):
                            # 计算xyz_quat_cmd
                            feature_name_hd5 = "command_poses_dict/" + feature_id.split(".")[1]
                            frame[feature_id] = torch.from_numpy(file[feature_name_hd5][frame_idx])
                        elif "xyz_so3_state" in feature_id.split("."):
                            if state_is_last_action and "gripper" not in feature_id.split(".")[-1].split("_"):
                                feature_name_hd5 = "command_poses_dict/" + feature_id.split(".")[1]
                                state_frame_idx = max(0,frame_idx - 1 )
                                xyz_quat = file[feature_name_hd5][state_frame_idx]
                            else:
                                # 计算xyz_so3_state
                                feature_name_hd5 = "poses_dict/" + feature_id.split(".")[1]
                                xyz_quat = file[feature_name_hd5][frame_idx]
                            xyz_so3 = s1_trans.xyz_quat_to_so3(xyz_quat)

                            frame[feature_id] = torch.from_numpy(xyz_so3)
                        elif "xyz_so3_cmd" in feature_id.split("."):
                            # 计算xyz_so3_cmd
                            feature_name_hd5 = "command_poses_dict/" + feature_id.split(".")[1]
                            xyz_quat = file[feature_name_hd5][frame_idx]
                            xyz_so3 = s1_trans.xyz_quat_to_so3(xyz_quat)
                            frame[feature_id] = torch.from_numpy(xyz_so3)
                        elif "joint_pose_state" in feature_id.split("."):
                            if state_is_last_action and "gripper" not in feature_id.split(".")[-1].split("_"):
                                feature_name_hd5 = "joints_dict/joints_position_command"
                                state_frame_idx = max(0,frame_idx - 1 )
                                joints_pose = file[feature_name_hd5][state_frame_idx]
                            else:
                                # 计算joint_pose_state
                                feature_name_hd5 = "joints_dict/joints_position_state"
                                joints_pose = file[feature_name_hd5][frame_idx]
                            joint_pose = s1_trans.get_joint_data_form_name(joints_pose, feature_id.split(".")[1])
                            frame[feature_id] = torch.from_numpy(joint_pose)
                        elif "joint_vel_state" in feature_id.split("."):
                            feature_name_hd5 = "joints_dict/joints_velocity_state"
                            joints_vel = file[feature_name_hd5][frame_idx]
                            joint_vel = s1_trans.get_joint_data_form_name(joints_vel, feature_id.split(".")[1])
                            frame[feature_id] = torch.from_numpy(joint_vel)
                        elif "joint_pose_cmd" in feature_id.split("."):
                            feature_name_hd5 = "joints_dict/joints_position_command"
                            joints_pose = file[feature_name_hd5][frame_idx]
                            joint_pose = s1_trans.get_joint_data_form_name(joints_pose, feature_id.split(".")[1])
                            frame[feature_id] = torch.from_numpy(joint_pose)

                        continue
                    feature_name_hd5 = feature_id.replace(".", "/")
                    if "rgb" in feature_id.split("."):
                        if image_compressed:
                            img_size_list = file[feature_name_hd5.replace("rgb", "rgb_size")][:]
                            start_frame = int(np.sum(img_size_list[:frame_idx]))
                            end_frame = int(start_frame + img_size_list[frame_idx])
                        image = (
                            (file[feature_name_hd5][frame_idx])
                            if not image_compressed
                                else file[feature_name_hd5][start_frame:end_frame]
                                    )
                        if image_compressed:
                            frame[feature_id] = torch.from_numpy(image)
                        else:
                            frame[feature_id] = torch.from_numpy(image.transpose(2, 0, 1)[[2, 1, 0], :, :])


                    else:
                        frame[feature_id] = torch.from_numpy(file[feature_name_hd5][frame_idx])

                if "cartesian_so3_dict.cartesian_pose_command" in features and "cartesian_so3_dict/cartesian_pose_command" not in file.keys():
                    so3_torso = frame["xyz_so3_cmd.astribot_torso"]
                    so3_arm_left = frame["xyz_so3_cmd.astribot_arm_left"]
                    gripper_left = frame["joint_pose_cmd.astribot_gripper_left"]
                    so3_arm_right = frame["xyz_so3_cmd.astribot_arm_right"]
                    gripper_right = frame["joint_pose_cmd.astribot_gripper_right"]
                    head = frame["joint_pose_cmd.astribot_head"]
                    frame["cartesian_so3_dict.cartesian_pose_command"] = torch.cat([so3_torso, so3_arm_left, gripper_left, so3_arm_right, gripper_right, head])


                if "cartesian_so3_dict.cartesian_pose_state" in features and "cartesian_so3_dict/cartesian_pose_state" not in file.keys():
                    so3_torso = frame["xyz_so3_state.astribot_torso"]
                    so3_arm_left = frame["xyz_so3_state.astribot_arm_left"]
                    gripper_left = frame["joint_pose_state.astribot_gripper_left"]
                    so3_arm_right = frame["xyz_so3_state.astribot_arm_right"]
                    gripper_right = frame["joint_pose_state.astribot_gripper_right"]
                    head = frame["joint_pose_state.astribot_head"]
                    frame["cartesian_so3_dict.cartesian_pose_state"] = torch.cat([so3_torso, so3_arm_left, gripper_left, so3_arm_right, gripper_right, head])

                frames.append(frame)
        return frames

    @staticmethod
    def define_features(
        hdf5_file_path: Path, image_compressed: bool = True, encode_as_video: bool = True

    ) -> dict[str, dict]:
        """
        Define features from an HDF5 file.
        Parameters
        ----------
        hdf5_file_path : Path
            The path to the HDF5 file.
        image_compressed : bool, optional
            Whether the images are compressed, by default True.
        encode_as_video : bool, optional
            Whether to encode images as video or as images, by default True.
        Returns
        -------
        dict[str, dict]
            A dictionary where keys are topic names and values are dictionaries
            containing feature information such as dtype, shape, and names.
        """

        # Initialize lists to store topics and features
        s1_trans = S1Transformer()
        topics = []
        features = {}

        # Open the HDF5 file
        with (h5py.File(hdf5_file_path, "r") as hdf5_file):
            # Collect all dataset names in the HDF5 file
            hdf5_file.visititems(lambda name, obj: topics.append(name) if isinstance(obj, h5py.Dataset) else None)

            # Iterate over each topic to define its features
            for topic in topics:
                # If the topic is an image, define it as a video feature
                if "rgb" in topic.split("/"):
                    sample = hdf5_file[topic][0]
                    if image_compressed:
                        img_size = int(hdf5_file[topic.replace("rgb","rgb_size")][0])
                    features[topic.replace("/", ".")] = {
                        "dtype": "video" if encode_as_video else "image",
                        "shape": cv2.imdecode(hdf5_file[topic][0:img_size], 1).transpose(2, 0, 1).shape
                        if image_compressed
                        else sample.transpose(2, 0, 1).shape,
                        "names": [
                            "channel",
                            "height",
                            "width",
                        ],
                    }
                # Skip compressed length topics
                elif "compress_len" in topic.split("/"):
                    continue
                # Otherwise, define it as a regular feature
                elif ("joints_dict" in topic.split("/") or "command_poses_dict" in topic.split("/") or "poses_dict" in topic.split("/")) and ("timestamp" not in topic.split("/")[-1].split("_")) :
                    features[topic.replace("/", ".")] = {
                        "dtype": str(hdf5_file[topic][0].dtype),
                        "shape": (topic_shape := hdf5_file[topic][0].shape),
                        "names": [f"{topic.split('/')[-1]}_{k}" for k in range(topic_shape[0])],
                    }
                else:
                    continue

            def add_xyz_quat_fature(name:str,features:dict):
                if name not in features:
                    features[name] = {
                        "dtype": "float64",
                        "shape": [7],
                        "names": ["x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w"],
                    }
            add_xyz_quat_fature("xyz_quat_state.astribot_chassis",features)
            add_xyz_quat_fature("xyz_quat_state.astribot_torso",features)
            add_xyz_quat_fature("xyz_quat_state.astribot_arm_left",features)
            add_xyz_quat_fature("xyz_quat_state.astribot_arm_right",features)
            add_xyz_quat_fature("xyz_quat_state.astribot_head",features)

            add_xyz_quat_fature("xyz_quat_cmd.astribot_chassis",features)
            add_xyz_quat_fature("xyz_quat_cmd.astribot_torso",features)
            add_xyz_quat_fature("xyz_quat_cmd.astribot_arm_left",features)
            add_xyz_quat_fature("xyz_quat_cmd.astribot_arm_right",features)
            add_xyz_quat_fature("xyz_quat_cmd.astribot_head",features)


            def add_xyz_so3_fature(name:str,features:dict):
                if name not in features:
                    features[name] = {
                        "dtype": "float64",
                        "shape": [9],
                        "names": ["x", "y", "z", "so3_0", "so3_1", "so3_2", "so3_3", "so3_4", "so3_5"],
                    }
            add_xyz_so3_fature("xyz_so3_state.astribot_chassis",features)
            add_xyz_so3_fature("xyz_so3_state.astribot_torso",features)
            add_xyz_so3_fature("xyz_so3_state.astribot_arm_left",features)
            add_xyz_so3_fature("xyz_so3_state.astribot_arm_right",features)
            add_xyz_so3_fature("xyz_so3_state.astribot_head",features)

            add_xyz_so3_fature("xyz_so3_cmd.astribot_chassis",features)
            add_xyz_so3_fature("xyz_so3_cmd.astribot_torso",features)
            add_xyz_so3_fature("xyz_so3_cmd.astribot_arm_left",features)
            add_xyz_so3_fature("xyz_so3_cmd.astribot_arm_right",features)
            add_xyz_so3_fature("xyz_so3_cmd.astribot_head",features)

            def add_joint_fature(name:str,features:dict,shape:int):
                if name not in features:
                    features[name] = {
                        "dtype": "float64",
                        "shape": [shape],
                        "names":  [f"{name}_{k}" for k in range(shape)],
                    }

            add_joint_fature("joint_pose_state.astribot_chassis",features,s1_trans.joints_name_dim_dict["astribot_chassis"])
            add_joint_fature("joint_pose_state.astribot_torso",features,s1_trans.joints_name_dim_dict["astribot_torso"])
            add_joint_fature("joint_pose_state.astribot_arm_left",features,s1_trans.joints_name_dim_dict["astribot_arm_left"])
            add_joint_fature("joint_pose_state.astribot_gripper_left",features,s1_trans.joints_name_dim_dict["astribot_gripper_left"])
            add_joint_fature("joint_pose_state.astribot_arm_right",features,s1_trans.joints_name_dim_dict["astribot_arm_right"])
            add_joint_fature("joint_pose_state.astribot_gripper_right",features,s1_trans.joints_name_dim_dict["astribot_gripper_right"])
            add_joint_fature("joint_pose_state.astribot_head",features,s1_trans.joints_name_dim_dict["astribot_head"])

            add_joint_fature("joint_vel_state.astribot_chassis",features,s1_trans.joints_name_dim_dict["astribot_chassis"])
            add_joint_fature("joint_vel_state.astribot_torso",features,s1_trans.joints_name_dim_dict["astribot_torso"])
            add_joint_fature("joint_vel_state.astribot_arm_left",features,s1_trans.joints_name_dim_dict["astribot_arm_left"])
            add_joint_fature("joint_vel_state.astribot_gripper_left",features,s1_trans.joints_name_dim_dict["astribot_gripper_left"])
            add_joint_fature("joint_vel_state.astribot_arm_right",features,s1_trans.joints_name_dim_dict["astribot_arm_right"])
            add_joint_fature("joint_vel_state.astribot_gripper_right",features,s1_trans.joints_name_dim_dict["astribot_gripper_right"])
            add_joint_fature("joint_vel_state.astribot_head",features,s1_trans.joints_name_dim_dict["astribot_head"])

            add_joint_fature("joint_pose_cmd.astribot_chassis",features,s1_trans.joints_name_dim_dict["astribot_chassis"])
            add_joint_fature("joint_pose_cmd.astribot_torso",features,s1_trans.joints_name_dim_dict["astribot_torso"])
            add_joint_fature("joint_pose_cmd.astribot_arm_left",features,s1_trans.joints_name_dim_dict["astribot_arm_left"])
            add_joint_fature("joint_pose_cmd.astribot_gripper_left",features,s1_trans.joints_name_dim_dict["astribot_gripper_left"])
            add_joint_fature("joint_pose_cmd.astribot_arm_right",features,s1_trans.joints_name_dim_dict["astribot_arm_right"])
            add_joint_fature("joint_pose_cmd.astribot_gripper_right",features,s1_trans.joints_name_dim_dict["astribot_gripper_right"])
            add_joint_fature("joint_pose_cmd.astribot_head",features,s1_trans.joints_name_dim_dict["astribot_head"])

            add_joint_fature("cartesian_so3_dict.cartesian_pose_command",features,31)
            add_joint_fature("cartesian_so3_dict.cartesian_pose_state",features,31)


        # Return the defined features
        return features

def get_label_dict(label_raw_path: list[Path]) -> list[dict]:
    label_dir_list = list((label_raw_path).glob("*.json"))
    label_dict = {}
    fps = 30

    def recursively_parse_json(obj):
        if isinstance(obj, dict):
            return {k: recursively_parse_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursively_parse_json(item) for item in obj]
        elif isinstance(obj, str):
            try:
                parsed = json.loads(obj)
                return recursively_parse_json(parsed)
            except (json.JSONDecodeError, TypeError):
                return obj  # 返回原始字符串
        else:
            return obj  # 原始值（如 int、float、None）

    def analyze_label_dict(obj):
        """
        Analyze the label dictionary to extract relevant information.
        This function is a placeholder and should be implemented based on specific requirements.
        """
        # Example implementation: print the keys of the label dictionary
        ori_label_dict = recursively_parse_json(obj)
        output_dict = {}
        for item in ori_label_dict:
            idx = int(item["fileName"].split(".")[-2].split("_")[-1])
            key_label = {}
            start_frame = 0
            if len(item["result"]) == 0:
                print(f"\033[91mEpisode {idx} has no label, using all frames.\033[0m")
                input_key = input(f"\033[93mEpisode {idx} has no label, 是否使用全部frame y/n:\033[0m ")
                if input_key == "y":
                    output_dict[str(idx)] = None
                else:
                    exit()

                continue
            end_frame = int(item["result"]["duration"] * fps)
            for result in item["result"]["annotations"][0]["result"]:
                key_label[result["attributes"]["Data_Validity"][0]] = result
                key_label[result["attributes"]["Data_Validity"][0]]["frame_range"] = [start_frame,int(result["time"] * fps)]
                start_frame = int(result["time"] * fps)

            if end_frame > start_frame:
                key_label["other"]={"frame_range": [start_frame, end_frame]}

            output_dict[str(idx)] = key_label
        return output_dict



    for label_dir in label_dir_list:
        label_dict.update(analyze_label_dict(json.load(open(label_dir, "r"))))
    return label_dict


class DatasetConverter:
    """
    A class to convert datasets to Lerobot format.
    Parameters
    ----------
    raw_path : Path or str
        The path to the raw dataset.
    dataset_repo_id : str
        The repository ID where the dataset will be stored.
    fps : int
        Frames per second for the dataset.
    robot_type : str, optional
        The type of robot, by default "".
    encode_as_videos : bool, optional
        Whether to encode images as videos, by default True.
    image_compressed : bool, optional
        Whether the images are compressed, by default True.
    image_writer_processes : int, optional
        Number of processes for writing images, by default 0.
    image_writer_threads : int, optional
        Number of threads for writing images, by default 0.
    Methods
    -------
    extract_episode(episode_path, task_description='')
        Extracts frames from a single episode and saves it with a description.
    extract_episodes(episode_description='')
        Extracts frames from all episodes and saves them with a description.
    push_dataset_to_hub(dataset_tags=None, private=False, push_videos=True, license="apache-2.0")
        Pushes the dataset to the Hugging Face Hub.
    init_lerobot_dataset()
        Initializes the Lerobot dataset.
    """

    def __init__(
        self,
        raw_path: Path | str,
        root_path: Path | str,
        dataset_repo_id: str,
        fps: int,
        epoch_num: int = -1,
        robot_type: str = "",
        encode_as_videos: bool = True,
        image_compressed: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        state_is_last_action: bool = True,
        use_label: bool = False,
    ):
        self.raw_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        self.root_path = root_path if isinstance(root_path, Path) else Path(root_path)
        self.dataset_repo_id = dataset_repo_id
        self.fps = fps
        self.robot_type = robot_type
        self.image_compressed = image_compressed
        self.image_writer_threads = image_writer_threads
        self.image_writer_processes = image_writer_processes
        self.encode_as_videos = encode_as_videos
        self.state_is_last_action = state_is_last_action
        self.use_label = use_label

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"{'-'*10} Aloha HD5 -> Lerobot Converter {'-'*10}")
        self.logger.info(f"Processing Aloha HD5 dataset from {self.raw_path}")
        self.logger.info(f"Dataset will be stored in {self.dataset_repo_id}")
        self.logger.info(f"FPS: {self.fps}")
        self.logger.info(f"Robot type: {self.robot_type}")
        self.logger.info(f"Image compressed: {self.image_compressed}")
        self.logger.info(f"Encoding images as videos: {self.encode_as_videos}")
        self.logger.info(f"#writer processes: {self.image_writer_processes}")
        self.logger.info(f"#writer threads: {self.image_writer_threads}")

        self.episode_list = list(self.raw_path.glob("*.hdf5"))
        if epoch_num <= 0:
            epoch_num = len(self.episode_list)
        else:
            epoch_num = min(epoch_num, len(self.episode_list))
            self.episode_list = self.episode_list[:epoch_num]
        self.epoch_num = epoch_num
        if use_label:
            self.label_dict = get_label_dict(self.raw_path/"Labels")
        else:
            self.label_dict = None





        S1HD5Extractor.check_format(self.episode_list, image_compressed=self.image_compressed)

        self.features = S1HD5Extractor.define_features(
            self.episode_list[0],
            image_compressed=self.image_compressed,
            encode_as_video=self.encode_as_videos,

        )

    def extract_episode(self, episode_path, task_description: str = ""):
        """
        Extracts frames from an episode and saves them to the dataset.
        Parameters
        ----------
        episode_path : str
            The path to the episode file.
        task_description : str, optional
            A description of the task associated with the episode (default is an empty string).
        Returns
        -------
        None
        """

        for frame in S1HD5Extractor.extract_episode_frames(episode_path, self.features,self.image_compressed,self.state_is_last_action,self.label_dict):
            self.dataset.add_frame(frame)
        episode_idx = episode_path.stem.split("_")[-1]
        self.logger.info(f"Saving Episode:{episode_idx} with Description: {task_description} ...")

        self.save_images_path()
        self.dataset.save_episode(task=task_description)
    def rm_images(self,keep_image_files:bool = False):
        """
        及时删除图片  释放空间
        """
        if not keep_image_files:
            if self.cur_image_path_list :
                for img_path in self.cur_image_path_list:
                    shutil.rmtree(img_path)
        self.cur_image_path_list = []
        pass
    def save_images_path(self):
        """
        保存图片的保存文件夹
        """
        self.cur_image_path_list = []
        for key in self.features:
            if self.dataset.features[key]["dtype"] in ["image", "video"]:
                img_path = Path(self.dataset.episode_buffer[key][0]).parent
                self.cur_image_path_list.append(img_path)

        pass

    def extract_episodes(self, episode_description: str = "",keep_image_files:bool = False):
        """
        Extracts episodes from the episode list and processes them.
        Parameters
        ----------
        episode_description : str, optional
            A description of the task to be passed to the extract_episode method (default is '').
        Raises
        ------
        Exception
            If an error occurs during the processing of an episode, it will be caught and printed.
        Notes
        -----
        After processing all episodes, the dataset is consolidated.
        """

        self.episode_list.sort(key=lambda x: int(str(x).rsplit(".", 1)[0].rsplit("_", 1)[-1]))

        for episode_idx , episode_path in enumerate( self.episode_list):
            try:
                episode_idx_name = episode_path.stem.split(".")[-1]
                self.logger.info(f"Processing episode: {episode_idx} / {self.epoch_num}  , ...." + episode_idx_name)
                self.extract_episode(episode_path, task_description=episode_description)
                self.rm_images(keep_image_files)

            except Exception as e:
                print(f"Error processing episode {episode_path}", f"{e}")
                traceback.print_exc()
                continue
        self.dataset.consolidate(run_compute_stats=True,keep_image_files=keep_image_files)

    def push_dataset_to_hub(
        self,
        dataset_tags: list[str] | None = None,
        private: bool = False,
        push_videos: bool = True,
        license: str | None = "apache-2.0",
    ):
        """
        Pushes the dataset to the Hugging Face Hub.
        Parameters
        ----------
        dataset_tags : list of str, optional
            A list of tags to associate with the dataset on the Hub. Default is None.
        private : bool, optional
            If True, the dataset will be private. Default is False.
        push_videos : bool, optional
            If True, videos will be pushed along with the dataset. Default is True.
        license : str, optional
            The license under which the dataset is released. Default is "apache-2.0".
        Returns
        -------
        None
        """

        self.logger.info(f"Pushing dataset to Hugging Face Hub. ID: {self.dataset_repo_id} ...")
        self.dataset.push_to_hub(
            tags=dataset_tags,
            license=license,
            push_videos=push_videos,
            private=private,
        )

    def init_lerobot_dataset(self):
        """
        Initializes the LeRobot dataset.
        This method cleans the cache if the dataset already exists and then creates a new LeRobot dataset.
        Returns
        -------
        LeRobotDataset
            The initialized LeRobot dataset.
        """

        # Clean the cache if the dataset already exists

        # Clean the cache if the dataset already exists
        if os.path.exists(self.root_path / self.dataset_repo_id):
            user_input = input("删除(y)  ,计算stats mean std (s), 其他退出：" + str(self.root_path / self.dataset_repo_id))  # 等待用户输入
            if user_input == "y":
                shutil.rmtree(self.root_path / self.dataset_repo_id)
            else:
                if user_input == "s":

                    self.dataset = CustomLeRobotDataset(
                        root=self.root_path/ self.dataset_repo_id,
                        repo_id=self.dataset_repo_id,
                    )
                    self.dataset.meta.stats = compute_stats(self.dataset)

                    serialized_stats = serialize_dict(self.dataset.meta.stats)
                    write_json(serialized_stats, self.dataset.root / STATS_PATH)
                exit()
        self.dataset = CustomLeRobotDataset.create(
            root=self.root_path/ self.dataset_repo_id,
            repo_id=self.dataset_repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            features=self.features,
            image_writer_threads=self.image_writer_threads,
            image_writer_processes=self.image_writer_processes,
        )

        return self.dataset


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    """
    Convert Aloha HD5 dataset and push to Hugging Face hub.
    This script processes raw HDF5 files from the Aloha dataset, converts them into a specified format,
    and optionally uploads the dataset to the Hugging Face hub.
    Parameters
    ----------
    --raw-path : Path
        Directory containing the raw HDF5 files.
    --dataset-repo-id : str
        Repository ID where the dataset will be stored.
    --fps : int
        Frames per second for the dataset.
    --robot-type : str, optional
        Type of robot, either "aloha-stationary" or "aloha-mobile". Default is "aloha-stationary".
    --private : bool, optional
        Set to True to make the dataset private. Default is False.
    --push-videos : bool, optional
        Set to True to push videos to the hub. Default is True.
    --license : str, optional
        License for the dataset. Default is "apache-2.0".
    --image-compressed : bool, optional
        Set to True if the images are compressed. Default is True.
    --video-encoding : bool, optional
        Set to True to encode images as videos. Default is True.
    --nproc : int, optional
        Number of image writer processes. Default is 10.
    --nthreads : int, optional
        Number of image writer threads. Default is 5.
    """

    parser = argparse.ArgumentParser(description="Convert Aloha HD5 dataset and push to Hugging Face hub.")
    parser.add_argument("--raw-path", type=Path, required=False,
                        default=Path("/media/xc/data/data/s1_pick_and_place/right_and_left_arm_pick_and_place/Object_sorting_using_right_hands/so3_data_30hz/group_1"), help="Directory containing the raw hdf5 filyes.")
    parser.add_argument("--root-path", type=Path, required=False, default=None, help="Root path for the dataset.")
    parser.add_argument("--epoch_num", type=int, required=False, default=-1, help="Number of episodes to process.")
    parser.add_argument(
        "--dataset-repo-id", type=str, required=False, default=None,
        help="Repository ID where the dataset will be stored."
    )
    parser.add_argument("--fps", type=int, required=False, default=30,
        help="Frames per second for the dataset.")
    parser.add_argument(
        "--description", type=str, help="Description of the dataset.", default="S1 pick the obj and place it into the box."
    )

    parser.add_argument(
        "--robot-type",
        type=str,
        choices=["S1-stationary", "S1-mobile"],
        default="S1-stationary",
        help="Type of robot.",
    )
    parser.add_argument("--private", type=str2bool, default=True, help="Set to True to make the dataset private.")
    parser.add_argument("--push", type=str2bool, default=False,
                        help="Set to True to push videos to the hub.")
    parser.add_argument("--license", type=str, default="apache-2.0", help="License for the dataset.")
    parser.add_argument(
        "--image-compressed", type=str2bool, default=False, help="Set to True if the images are compressed."
    )
    parser.add_argument("--video-encoding", type=str2bool, default=True, help="Set to True to encode images as videos.")

    parser.add_argument("--nproc", type=int, default=10, help="Number of image writer processes.")
    parser.add_argument("--nthreads", type=int, default=5, help="Number of image writer threads.")
    parser.add_argument("--use_label", type=str2bool, default=True, help="use label in the dataset")


    args = parser.parse_args()
    print(
        args.video_encoding,
    )
    raw_path = args.raw_path
    if args.root_path is None:
        root_path = Path(*raw_path.parts[:-2])
    else:
        root_path = args.root_path
    if args.dataset_repo_id is None:
        dataset_repo_id = f"{raw_path.parts[-2]}/{raw_path.parts[-1]}/lerobot_astribot"
    else:
        dataset_repo_id = args.dataset_repo_id

    if args.use_label:
        print("use label in the dataset")


    converter = DatasetConverter(
        raw_path=args.raw_path,
        root_path=root_path,
        epoch_num=args.epoch_num,
        dataset_repo_id=dataset_repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        image_compressed=args.image_compressed,
        encode_as_videos=args.video_encoding,
        image_writer_processes=args.nproc,
        image_writer_threads=args.nthreads,
        use_label=args.use_label
    )
    converter.init_lerobot_dataset()
    converter.extract_episodes(episode_description=args.description)

    if args.push:
        converter.push_dataset_to_hub(
            dataset_tags=S1HD5Extractor.TAGS, private=args.private, push_videos=True, license=args.license
        )


if __name__ == "__main__":
    main()
