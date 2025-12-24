"""Visualization script for policy prediction results.

This script compares predicted actions from a policy with ground truth actions
from a dataset and generates comparison plots for each action dimension.
"""

import dataclasses
import logging
import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi_client import websocket_client_policy as _websocket_client_policy

matplotlib.use('Agg')  # Set backend to Agg to avoid GUI conflicts

@dataclasses.dataclass
class Args:
    """Configuration arguments for result visualization."""

    # Dataset configuration
    dataset_root_path: str = "/home/astribot-2gpu/openpi_test/openpi/pi0_open/"
    dataset_repo_id: str = "lerobot_so3_data_30hz"
    data_episode_id: int = 0

    # Visualization configuration
    config: str = "popcorn"
    frame_step: int = 10

    # Network configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Output configuration
    output_dir: str = "./result/"


def prepare_image_dict(data: dict, image_names: dict, output_dir: str) -> dict:
    """Prepare image dictionary from dataset frame.

    Args:
        data: Dataset frame data.
        image_names: Mapping of camera names to data keys.
        output_dir: Directory to save debug images.

    Returns:
        Dictionary with camera names as keys and images as values.
    """
    image_dict = {}
    for name in image_names:
        image = data[image_names[name]].numpy()

        # Convert CHW to HWC and scale to uint8
        image_hwc = np.transpose(image, (1, 2, 0))
        image_hwc_uint8 = (image_hwc * 255).astype(np.uint8)

        # Save debug image
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "show_result_head_image.png"), image_hwc_uint8)

        image_dict[name] = image

    return image_dict


def plot_action_comparison(
    all_actions: list,
    all_real_actions: list,
    all_states: list,
    output_path: str
) -> None:
    """Plot comparison between predicted and real actions.

    Args:
        all_actions: List of predicted actions.
        all_real_actions: List of ground truth actions.
        all_states: List of robot states.
        output_path: Path to save the plot.
    """
    num_dimensions = len(all_actions[0])

    # Create subplots for each dimension
    _, axes = plt.subplots(num_dimensions, 1, figsize=(12, 2 * num_dimensions))

    # Handle single dimension case
    if num_dimensions == 1:
        axes = [axes]

    # Plot each dimension
    for dim in range(num_dimensions):
        action_dim = [action[dim] for action in all_actions]
        real_action_dim = [real_action[dim] for real_action in all_real_actions]
        real_state_dim = [real_state[dim] for real_state in all_states]

        ax = axes[dim]
        ax.plot(action_dim, label=f'Predicted Action (Dim {dim})', marker='o', linestyle='--')
        ax.plot(real_action_dim, label=f'Real Action (Dim {dim})', marker='x', linestyle='-')
        ax.plot(real_state_dim, label=f'Real State (Dim {dim})', marker='*', linestyle='-')

        ax.set_title(f'Action and Real Action Comparison for Dimension {dim}')
        ax.set_xlabel('Time Frame (Index)')
        ax.set_ylabel('Action Value')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(args: Args) -> None:
    """Main evaluation and visualization function.

    Args:
        args: Configuration arguments.
    """
    # Define data field mappings
    image_names = {
        "cam_high": "images_dict.head.rgb",
        "cam_left_wrist": "images_dict.left.rgb",
        "cam_right_wrist": "images_dict.right.rgb",
    }
    state_names = "cartesian_so3_dict.cartesian_pose_state"
    action_names = "cartesian_so3_dict.cartesian_pose_command"

    # Load dataset
    dataset = LeRobotDataset(
        root=args.dataset_root_path + args.dataset_repo_id,
        repo_id=args.dataset_repo_id,
        episodes=[args.data_episode_id],
        local_files_only=True
    )

    # Initialize policy
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # Storage for results
    all_actions = []
    all_real_actions = []
    all_states = []

    # Process dataset frames
    frame_id = 0
    while frame_id < dataset.num_frames:
        # Get current frame data
        data = dataset[frame_id]

        # Prepare observation
        image_dict = prepare_image_dict(data, image_names, args.output_dir)
        obs_dict = {
            "images": image_dict,
            "state": data[state_names].numpy(),
            "prompt": "S1 pick the obj and place it into the box.",
            "actions": [data[action_names].numpy()]
        }

        # Run inference
        time_start = time.time()
        action_list = policy.infer(obs_dict)['actions']
        print(f"Inference time: {time.time() - time_start:.3f}s")

        # Collect actions for each step
        for run_step in range(args.frame_step):
            if frame_id + run_step >= dataset.num_frames:
                break

            action = action_list[run_step]
            data = dataset[frame_id + run_step]
            real_action = data[action_names].numpy()
            real_state = data[state_names].numpy()

            all_actions.append(action)
            all_real_actions.append(real_action)
            all_states.append(real_state)

            print(f"Frame {frame_id + run_step}")

        frame_id += args.frame_step

    # Generate comparison plot
    output_path = os.path.join(
        args.output_dir,
        f"{args.config}_{args.data_episode_id}_action_real_comparison.png"
    )
    plot_action_comparison(all_actions, all_real_actions, all_states, output_path)
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)


