import dataclasses
import json
import pathlib
import logging
import numpy as np
import openpi.training.config_auto as _config
import time
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg，避免图形界面冲突
import matplotlib.pyplot as plt


from openpi_client import websocket_client_policy as _websocket_client_policy

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import tyro

from glob import glob

@dataclasses.dataclass
class Args:
    out_path: pathlib.Path = pathlib.Path("out.mp4")
    dataset_root_path: str = "/home/astribot-2gpu/openpi_test/openpi/pi0_open/"
    dataset_repo_id: str = "lerobot_so3_data_30hz"
    config: str = "popcorn"

    data_episode_id: int = 0
    frame_step: int = 10

    host: str = "0.0.0.0"
    port: int = 8000


def main(args: Args) -> None:

    # config = _config.get_config(args.config)
    # data_config = config.data.create(config.assets_dirs, config.model)


    # structure = data_config.repack_transforms.inputs[0].structure
    image_names = { "cam_high": "images_dict.head.rgb",
                                    "cam_left_wrist": "images_dict.left.rgb",
                                    "cam_right_wrist": "images_dict.right.rgb",
                                    }
    state_names = "cartesian_so3_dict.cartesian_pose_state"
    action_names = "cartesian_so3_dict.cartesian_pose_command"
    episode_id = args.data_episode_id
    dataset = LeRobotDataset(root=Args.dataset_root_path + Args.dataset_repo_id,
                             repo_id=Args.dataset_repo_id,
                             episodes=[episode_id],local_files_only=True
    )

    policy=_websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    # 用于保存每一帧的 action 和 real_action
    all_actions = []
    all_real_actions = []
    all_action_diffs = []
    all_states = []
    frame_id = 0
    frame_step = args.frame_step
    while frame_id < dataset.num_frames:
        obs_dict = {}
        data = dataset[frame_id]
        image_dict = {}
        show_image_dict = {}
        for name in image_names:
            image = data[image_names[name]].numpy()

            image_hwc = np.transpose(image, (1, 2, 0))  # (H, W, C)

            image_hwc_uint8 = (image_hwc * 255).astype(np.uint8)

            # 显示图像
            cv2.imwrite("./result/" +"show_result_head_image.png", image_hwc_uint8)  # 保存图片

            image_dict[name] = image

            image_hwc_uint8 = image_hwc_uint8[..., ::-1]
            show_image_dict[name] = image_hwc_uint8

        obs_dict["images"] = image_dict

        obs_dict["state"] = data[state_names].numpy()

        obs_dict["prompt"] = "S1 pick the obj and place it into the box."
        obs_dict["actions"] = [data[action_names].numpy()]
        time_1 = time.time()
        action_list = policy.infer(obs_dict)['actions']
        print("infer time:", time.time()-time_1)

        for run_step in range(frame_step):
            action = action_list[run_step]

            if frame_id+run_step >= dataset.num_frames:
                break
            data = dataset[frame_id+run_step]
            real_action = data[action_names].numpy()

            all_actions.append(action)
            all_real_actions.append(real_action)

            action_diff = action - real_action
            all_action_diffs.append(action_diff)


            real_state = data[state_names].numpy()
            all_states.append(real_state)

            print(f"Frame {frame_id}:")
        frame_id += frame_step

    num_dimensions = len(all_actions[0])  # 假设每个 action 的维度数目相同

    # 创建子图，行数为维度数，列数为1（所有子图在一列中）
    fig, axes = plt.subplots(num_dimensions, 1, figsize=(12, 2 * num_dimensions))

    # 如果只有一个维度，axes 会是一个数组，处理此情况
    if num_dimensions == 1:
        axes = [axes]

    # 循环绘制每个维度的 action 和 real_action
    for dim in range(num_dimensions):
        action_dim = [action[dim] for action in all_actions]  # 当前维度的所有时间帧数据
        real_action_dim = [real_action[dim] for real_action in all_real_actions]  # 当前维度的所有时间帧数据
        real_state_dim = [real_state[dim] for real_state in all_states]  # 当前维度的所有时间帧数据

        # 获取当前子图
        ax = axes[dim]

        # 绘制当前维度的数据
        ax.plot(action_dim, label=f'Predicted Action (Dim {dim})', marker='o', linestyle='--')
        ax.plot(real_action_dim, label=f'Real Action (Dim {dim})', marker='x', linestyle='-')
        ax.plot(real_state_dim, label=f'Real State (Dim {dim})', marker='*', linestyle='-')

        # 添加图表标题和标签
        ax.set_title(f'Action and Real Action Comparison for Dimension {dim}')
        ax.set_xlabel('Time Frame (Index)')
        ax.set_ylabel('Action Value')
        ax.grid(True)
        ax.legend()

    # 调整布局，避免子图重叠
    plt.tight_layout()

    # 保存 action 和 real_action 的图
    plt.savefig("./result/" + args.config + '_'  + str(args.data_episode_id)+'_action_real_comparison.png')
    plt.close()  # 关闭图表，避免内存溢出


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)


