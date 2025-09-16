"""Compute normalization statistics for configs in parallel (multi-threaded).
"""
import copy
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config_auto as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import json

from pathlib import Path
class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
):
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
):
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(dataset, num_batches=num_batches)
    return data_loader, num_batches


def create_dataset(config: _config.TrainConfig, max_frames: int | None = None):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        return create_rlds_dataloader(data_config, config.model.action_horizon, config.batch_size, max_frames)
    else:
        return create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )


def process_one(idx: int, repo_id: str, config: _config.TrainConfig, list_len: int, max_frames: int | None = None):
    """子线程任务：建 dataloader + 统计 norm stats"""
    data_loader, num_batches = create_dataset(config, max_frames)

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(
        data_loader, total=num_batches, desc=f"Computing stats for {idx+1}/{list_len} {repo_id}"
    ):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: st.get_statistics() for key, st in stats.items()}

    output_path = (
        config.data.assets.assets_dir
        + "/"
        + config.data.assets.asset_id
        + "/"
        + config.name
    )
    print(f"[{idx+1}/{list_len}] Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    return str(output_path)

def compute_global_norm_stats(norm_stats_path_list, N_list):
    """根据多个 norm_stats.json 文件和对应的数据集大小 N_list 计算全局 norm stats"""
    norm_stats_list = []
    for norm_stats_path in norm_stats_path_list:
        if norm_stats_path.exists():
            norm_stats = normalize.load(norm_stats_path.parent)
            norm_stats_list.append(norm_stats)
        else:
            print(f"Warning: Norm stats file {norm_stats_path} does not exist, skipping...")

    if not norm_stats_list:
        raise ValueError("No norm stats files found to aggregate.")

    sum_norm_stats = {}
    for key in norm_stats_list[0].keys():
        mean_list, std_list, q01_list, q99_list = [], [], [], []

        for stats in norm_stats_list:
            mean_list.append(stats[key].mean)
            std_list.append(stats[key].std)
            q01_list.append(stats[key].q01)
            q99_list.append(stats[key].q99)

        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        N_array = np.array(N_list)

        # 加权平均
        total_mean = np.sum(N_array[:, None] * mean_list, axis=0) / np.sum(N_array)
        total_std = np.sqrt(
            np.sum(N_array[:, None] * (std_list ** 2 + mean_list ** 2), axis=0) / np.sum(N_array)
            - total_mean ** 2
        )

        weights = N_array / np.sum(N_array)
        global_q01 = np.sum([w * q for w, q in zip(weights, q01_list)], axis=0)
        global_q99 = np.sum([w * q for w, q in zip(weights, q99_list)], axis=0)

        sum_norm_stats[key] = normalize.NormStats(
            mean=total_mean, std=total_std, q01=global_q01, q99=global_q99
        )

    return sum_norm_stats
def main(config_name: str, max_frames: int | None = None, workers: int = 4):
    config = _config.get_config(config_name)
    repo_id_list, config_list = [], []
    norm_stats_path_list, N_list = [], []

    if config.data.repo_id_list is not None:
        for repo_id in config.data.repo_id_list:
            config_temp = copy.deepcopy(config)
            repo_id_force = "/".join(repo_id.split("/")[:-1])
            repo_id_end = repo_id.split("/")[-1]
            asset_id_child = "/".join([config_temp.data.assets.asset_id, repo_id])
            config_temp = replace(
                config_temp,
                data=replace(
                    config_temp.data,
                    repo_id=repo_id_end,
                    repo_id_list=[repo_id_end],
                    dataset_root="/".join([config_temp.data.dataset_root, repo_id_force]),
                    assets=replace(config_temp.data.assets, asset_id=asset_id_child),
                ),
                name=config.name.replace("_list", ""),
            )

            assets_dir = Path(config_temp.data.assets.assets_dir)
            norm_stats_path = assets_dir / config_temp.data.assets.asset_id / config_temp.name / "norm_stats.json"
            norm_stats_path_list.append(norm_stats_path)
            episodes_num_dir = assets_dir / config_temp.data.assets.asset_id / "meta/info.json"
            if episodes_num_dir.exists():
                total_episodes = json.loads(episodes_num_dir.read_text(encoding="utf-8"))["total_frames"]
                N_list.append(total_episodes)
            else:
                raise ValueError(f"episodes_num for {episodes_num_dir} not found")


            if norm_stats_path.exists():
                print(f"---Norm stats already exists for {repo_id}, path:{norm_stats_path} skipping...")
                continue

            print(f"---Processing {repo_id}...")
            repo_id_list.append(repo_id)
            config_list.append(config_temp)
    else:
        raise ValueError("config.data.repo_id_list is None, cannot process list.")

    list_len = len(repo_id_list)
    print(f"Total {list_len} datasets to process.")

    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, (repo_id, cfg) in enumerate(zip(repo_id_list, config_list)):
            tasks.append(executor.submit(process_one, idx, repo_id, cfg, list_len, max_frames))

        for f in as_completed(tasks):
            print("Done:", f.result())

    sum_norm_stats = compute_global_norm_stats(norm_stats_path_list, N_list)
    assets_dir = Path(config.data.assets.assets_dir)
    sum_norm_stats_path = assets_dir / config.data.assets.asset_id / config_name / "norm_stats.json"
    sum_norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
    normalize.save(sum_norm_stats_path.parent, sum_norm_stats)
    print(f"Saved combined norm stats to {sum_norm_stats_path}")



if __name__ == "__main__":
    tyro.cli(main)
