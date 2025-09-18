import dataclasses
import json
import pathlib
import logging
import numpy as np
import openpi.training.config_auto as _config
import time
import cv2
import os

# import scripts.serve_policy as _serve_policy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg，避免图形界面冲突
import matplotlib.pyplot as plt


# from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
# from openpi_client.runtime import runtime as _runtime
# from openpi_client.runtime.agents import policy_agent as _policy_agent

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# import saver as _saver
import tyro

from glob import glob

@dataclasses.dataclass
class Args:
    out_path: pathlib.Path = pathlib.Path("out.mp4")
    dataset_root_path: str = "/media/xc/guang4T/astribot_dataset/s1_pnp/WRC/gripper_pnp/to_Cart/731_left_1toy/"


    dataset_repo_id: str = "lerobot_so3_data_30hz"
    config: str = "wrc_pnp_test"

    so3_data_len: int = 36
    data_episode_id: int = 0
    seed: int = 0
    frame_step: int = 10
    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False
    env: str = "ALOHA_SIM"
    state_random_scale: float = 0.5
    save_attentions: bool = False


def resize_with_padding(image, target_width, target_height):
    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    pad_right = target_width - new_width - pad_left
    pad_bottom = target_height - new_height - pad_top

    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return padded_image


def build_data(data, image_names, state_names, structure, args):
    obs_dict = {}
    image_dict = {}
    all_image = []

    for name in image_names:
        image = data[image_names[name]].numpy()
        image_dict[name] = image

        image_hwc = np.transpose(image, (1, 2, 0))  # (H, W, C)   # Step 1: 转换格式为 (384, 384, 3) -> OpenCV 格式
        image_hwc_uint8 = (image_hwc * 255).astype(np.uint8)  # 假设 image_hwc 的值在 [0, 1] 范围内
        image_hwc_uint8 = image_hwc_uint8[..., ::-1]
        image_hwc_uint8 = cv2.resize(image_hwc_uint8, (224, 224))  # [..., ::-1]

        all_image.append(image_hwc_uint8)

        cv2.imwrite("output_{}.png".format(name), image_hwc_uint8)

    all_image = np.concatenate(all_image, axis=1)
    cv2.imwrite("output.png", all_image)

    obs_dict["images"] = image_dict

    obs_dict["state"] = data[state_names].numpy()

    if "prompt" in structure:
        obj_name = "carrot"
        text = 'pick ' + obj_name + ' into the box' + '\0'
        text_list = [ord(char) for char in text]
        obs_dict["prompt"] = text_list
        # obs_dict["prompt"] = data["prompt"]
    return obs_dict, all_image


def read_attention():
    # sort_files = []
    all_atts = []
    prev_0 = 0
    prev_1 = 0
    for root, _, files in os.walk('../../mask_vis'):
        files.sort()
        for subfiles in files:
            each_part = subfiles.split('.')
            temp_0 = int(each_part[0])
            temp_1 = int(each_part[1])

            assert temp_0 >= prev_0 and prev_1 >= prev_1
            prev_0 = temp_0
            prev_1 = temp_1
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = cur_dir +'/'+ root + '/' + subfiles
            # print(file_path)
            temp_att = np.load(file_path)
            all_atts.append(temp_att)
            os.remove(file_path)
    all_atts = np.concatenate(all_atts, axis=0)

    return all_atts


def all_np_files(folder_path,key = ".npy"):
    """判断文件夹内是否所有文件都是 .np 结尾"""
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("路径不存在或不是文件夹")
        return False

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    return all(f.endswith(key) for f in files) if files else False  # 确保文件夹非空


def clear_vis_att():
    folder_path = '../../mask_vis'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if all_np_files(folder_path):
            os.remove(folder_path)
            print("remove ",folder_path)
            os.makedirs(folder_path)
    folder_path = '../../attention_vis'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if all_np_files(folder_path, key=".png"):
            os.remove(folder_path)
            print("remove ",folder_path)
            os.makedirs(folder_path)

def delete_png_files(folder_path = '../../attention_vis'):
    """删除指定文件夹下所有 .png 文件"""
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("路径不存在或不是文件夹")
        return

    count = 0
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".png"):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            count += 1

    print(f"共删除 {count} 个 .png 文件")

def vis_att(all_atts, ori_image_dict, image_step=0, frame_id=0, pred_step=0):
    all_image = []
    for name in ["cam_high","cam_left_wrist","cam_right_wrist"]:
        if name in ori_image_dict:
            ori_image = ori_image_dict[name]
            padded_image = resize_with_padding(ori_image, 112, 112)

            all_image.append(padded_image)
        else:
            all_image.append(np.zeros((112, 112, 3), dtype=np.uint8))
    all_image = np.concatenate(all_image, axis=1)
    ori_image = cv2.resize(all_image, (112 * 3, 112))

    c = np.array([0, 255, 0])
    # c = np.array([255, 255, 255])


    # write to
    position = (0, 20)     # 文字的左下角位置
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
    font_scale = 0.5         # 字体大小
    color = (0, 0, 255)      # 文字颜色，格式为BGR（蓝色、绿色、红色）
    thickness = 1            # 字体粗细
    num_s = 3

    # left
    left = []
    right = []
    for n_h in range(6):
        temp_att = all_atts[frame_id, n_h, pred_step]
        s_att = temp_att[..., -1]
        temp_att = temp_att[..., :-1]

        state_score = round(float(s_att), num_s)




        new_att = []
        new_att.append(temp_att[:49].reshape(7, 7))
        new_att.append(temp_att[49:49 + 49].reshape(7, 7))
        new_att.append(temp_att[-49:].reshape(7, 7))

        new_att = np.concatenate(new_att, axis=1)
        new_att = cv2.resize(new_att, (112 * 3, 112))
        max_v = max(new_att.max(), s_att)
        min_v = min(new_att.min(), s_att)


        new_att = (new_att - min_v) / (max_v - min_v)
        new_att = new_att[..., None]

        save_img = ori_image * 0.5 + new_att * c * 0.5



        cv2.putText(save_img, str(state_score), position, font, font_scale, color, thickness)


        if n_h % 2 == 0:
            left.append(save_img)
        else:
            right.append(save_img)

    left = np.concatenate(left, axis=0)
    right = np.concatenate(right, axis=0)

    merge = np.concatenate([left, right], axis=1)

    for r_ind in range(6):
        cv2.line(merge, (0, 112 * r_ind), (448 * 2, 112 * r_ind), (0, 0, 128), 2)
    for c_ind in range(3):
        cv2.line(merge, (c_ind * 112 * 3, 0), (c_ind * 112 * 3, 112 * 5), (0, 0, 128), 2)


    merge = np.clip(merge, 0, 255).astype(np.uint8)

    # 创建空白区域用于放置文本
    text_height = 25  # 空白区域高度
    blank_area = np.ones((text_height, merge.shape[1], 3), dtype=np.uint8) * 255  # 白色背景

    # 组合空白区域和原始图片
    merge_with_text_space = np.vstack((blank_area, merge))

    # 生成文本
    text = f"Step: {image_step}, Frame: {frame_id}, Pred: {pred_step}"

    # 设置文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, text_height - 5)  # 文本放置在空白区域
    font_scale = 0.5
    font_color = (0, 0, 0)  # 文字颜色（黑色）
    thickness = 1

    # 在空白区域上绘制文本
    cv2.putText(merge_with_text_space, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    cv2.imwrite('../../attention_vis/image_{}_s_{}_p_{}.png'.format(image_step, frame_id, pred_step), merge_with_text_space)


def images_to_video(image_folder='../../attention_vis', output_video='../../attention_vis/att_video.mp4', fps=6):
    """将指定文件夹下的所有图片合成一个视频"""

    # 获取所有图片文件，支持 jpg、png、jpeg
    image_files = sorted(glob(os.path.join(image_folder, '*.*')), key=os.path.getmtime)

    # 过滤出图片文件
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("未找到图片文件")
        return

    # 读取第一张图片以获取尺寸
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' 适用于 .mp4 格式
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 逐帧写入视频
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"无法读取图片: {image_file}")
            continue
        video_writer.write(frame)
        print(f"添加帧: {image_file}")

    # 释放资源
    video_writer.release()
    print(f"视频已保存至: {output_video}")

def get_state_norm(data_config):
    data_root = data_config.dataset_root
    if data_config.repo_id_list is not None:
        data_root = data_root  + data_config.repo_id_list[0]
    json_path = data_root + '/meta/stats.json'
    # if data_config.norm_stats_path is not None:
    #     json_path = data_config.norm_stats_path
    with open(json_path, 'r') as data:
        stats_norm = json.load(data)
    state_norm = stats_norm[data_config.repack_transforms.inputs[0].structure["state"]]
    return state_norm


def main(args: Args) -> None:
    # clear_vis_att()
    delete_png_files()

    config = _config.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)

    # state_norm = get_state_norm(data_config)

    structure = data_config.repack_transforms.inputs[0].structure
    image_names = structure["images"]
    state_names = structure["state"]
    action_names = structure["actions"]
    episode_id = args.data_episode_id
    dataset = LeRobotDataset(root=Args.dataset_root_path + Args.dataset_repo_id,
                             repo_id=Args.dataset_repo_id,
                             episodes=[episode_id],
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
    # for frame_id in range(10):
        obs_dict = {}
        data = dataset[frame_id]
        image_dict = {}
        show_image_dict = {}
        for name in image_names:
            image = data[image_names[name]].numpy()

            # Step 1: 转换格式为 (384, 384, 3) -> OpenCV 格式
            image_hwc = np.transpose(image, (1, 2, 0))  # (H, W, C)

            # 假设 image_hwc 的值在 [0, 1] 范围内
            image_hwc_uint8 = (image_hwc * 255).astype(np.uint8)

            # 显示图像
            cv2.imwrite("./result/" +"show_result_head_image.png", image_hwc_uint8)  # 保存图片

            image_dict[name] = image

            image_hwc_uint8 = image_hwc_uint8[..., ::-1]
            show_image_dict[name] = image_hwc_uint8

        obs_dict["images"] = image_dict

        obs_dict["state"] = data[state_names].numpy()

        # state_random_values =  np.array(state_norm['std']) * args.state_random_scale
        # obs_dict["state"] += state_random_values
        if "prompt" in structure:
            obj_name = "carrot"
            # obj_name = "banana"
            text = 'pick ' + obj_name + ' into the box' + '\0'
            text_list = [ord(char) for char in text]
            obs_dict["prompt"] = text_list
            # obs_dict["prompt"] = data["prompt"]
            pass

        obs_dict["prompt"] = "S1 pick the obj and place it into the box."
        obs_dict["actions"] = [data[action_names].numpy()]
        if "head_so3_poses" in structure:
            head_so3_poses = data[structure["head_so3_poses"][0]].numpy()
            obs_dict["head_so3_poses"] = head_so3_poses
            pass
        time_1 = time.time()
        action_list = policy.infer(obs_dict)['actions']
        print("infer time:", time.time()-time_1)

        if args.save_attentions:
            all_atts = read_attention()

            for i in range(10):

                for j in range(frame_step):
                    vis_att(all_atts, show_image_dict, image_step=frame_id, frame_id=i * 12 + 11, pred_step=j)

        for run_step in range(frame_step):
            action = action_list[run_step]

            if frame_id+run_step >= dataset.num_frames:
                break
            data = dataset[frame_id+run_step]
            real_action = data[action_names].numpy()

            if args.so3_data_len == 24 and len(real_action) == 31:
                real_action = real_action[9:29]
            elif args.so3_data_len == 24 and len(real_action) == 20:
                real_action = real_action
            elif args.so3_data_len == 36 and len(real_action) == 31:
                real_action = real_action
            elif args.so3_data_len == 36 and len(real_action) == 34:
                real_action = real_action
            else:
                print("real_action error")

            # 保存每一帧的 action 和 real_action
            all_actions.append(action)
            all_real_actions.append(real_action)

        # 计算 action 的差异
            action_diff = action - real_action
            all_action_diffs.append(action_diff)


            real_state = data[state_names].numpy()
            if args.so3_data_len == 24 and len(real_state) == 31:
                real_state = real_state[9:29]
            elif args.so3_data_len == 24 and len(real_state) == 20:
                real_state = real_state
            elif args.so3_data_len == 36 and len(real_state) == 31:
                real_state = real_state
            elif args.so3_data_len == 36 and len(real_state) == 34:
                real_state = real_state
            else:
                print("real_state error")
            all_states.append(real_state)

            # 可选：打印每一帧的 action 和差异
            print(f"Frame {frame_id}:")
            # print("Predicted Action:", action)
            # print("Real Action:", real_action)
            # print("Action Difference:", action_diff)
        frame_id += frame_step

    # 假设 action 是多维数组，绘制每个维度的单独区域
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

    # 绘制单独的 action_diff 图
    fig, axes = plt.subplots(num_dimensions, 1, figsize=(12, 2 * num_dimensions))

    # 如果只有一个维度，axes 会是一个数组，处理此情况
    if num_dimensions == 1:
        axes = [axes]

    # 循环绘制每个维度的 action_diff
    for dim in range(num_dimensions):
        action_diff_dim = [action_diff[dim] for action_diff in all_action_diffs]  # 当前维度的所有时间帧数据

        # 获取当前子图
        ax = axes[dim]

        # 绘制当前维度的差异数据
        ax.plot(action_diff_dim, label=f'Difference (Dim {dim})', color='green', linestyle=':')

        # 添加图表标题和标签
        ax.set_title(f'Action Difference for Dimension {dim}')
        ax.set_xlabel('Time Frame (Index)')
        ax.set_ylabel('Difference Value')
        ax.grid(True)
        ax.legend()

    # 调整布局，避免子图重叠
    plt.tight_layout()

    # 保存差异图
    plt.savefig("./result/" +args.config + '_'+ str(args.data_episode_id)+'_action_diff.png')
    plt.close()  # 关闭图表，避免内存溢出

    print("Action and Difference plots saved.")
    images_to_video()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)


