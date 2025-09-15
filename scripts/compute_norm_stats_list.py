"""Compute normalization statistics for configs in parallel.

This script computes the mean and standard deviation of the data in datasets
and saves them to the config metadata directories. Supports multi-processing.
"""
import copy
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config_auto as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

from concurrent.futures import ThreadPoolExecutor



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
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None :
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
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
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches

def create_dataset(config: _config.TrainConfig, max_frames: int | None = None) -> tuple[str, _data_loader.Dataset]:

    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers,
            max_frames
        )


    return data_loader, num_batches


def process_one(idx: int, repo_id: str, data_loader, data_config, num_batches:int ,list_len: int, max_frames: int | None = None):

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches,
        desc=f"Computing stats for {idx+1}/{list_len} {repo_id}",):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = (data_config.data.assets.assets_dir + "/" + data_config.data.assets.asset_id
                   + "/" +  data_config.name)
    print(f"[{idx+1}/{list_len}] Writing stats to: {output_path}")
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    return str(output_path)


def main(config_name: str, max_frames: int | None = None, workers: int = 4):
    config = _config.get_config(config_name)
    repo_id_list, data_loader_list, num_batches_list,config_list = [], [], [], []

    if config.data.repo_id_list is not None:
        for repo_id in config.data.repo_id_list:
            config_temp = copy.deepcopy(config)
            repo_id_force = "/".join(repo_id.split("/")[:-1])
            repo_id_end = repo_id.split("/")[-1]
            asset_id_child = "/".join([config_temp.data.assets.asset_id,repo_id])
            config_temp = replace(
                config_temp,
                data=replace(
                    config_temp.data,
                    repo_id=repo_id_end,
                    repo_id_list=[repo_id_end],
                    dataset_root="/".join([config_temp.data.dataset_root, repo_id_force]),
                    assets=replace(config_temp.data.assets, asset_id=asset_id_child)
                ),
                # metadata_base_dir="/".join([config.metadata_base_dir, repo_id_force]),
                name=config.name.replace("_list", ""),
            )
            from pathlib import Path

            # 强制转换为 Path
            assets_dir = Path(config_temp.data.assets.assets_dir)

            norm_stats_path = (
                assets_dir /
                config_temp.data.assets.asset_id /
                config_temp.name /
                "norm_stats.json")
            if norm_stats_path.exists():
                print(f"---Norm stats already exists for {repo_id}, skipping...")
                continue

            print(f"---Processing {repo_id}...")

            data_loader, num_batches = create_dataset(config_temp,max_frames)


            repo_id_list.append(repo_id)
            data_loader_list.append(data_loader)
            config_list.append(config_temp)
            num_batches_list.append(num_batches)
    else:
        repo_id = config.data.repo_id
        data_loader, num_batches = create_dataset(config,max_frames)
        norm_stats_path = (
            config.metadata_dir.parent
            / config.data.repo_id
            / config.metadata_dir.name
            / "norm_stats.json"
        )
        if norm_stats_path.exists():
            print(f"---Norm stats already exists for {repo_id}, skipping...")
        else:
            print(f"---Processing {repo_id}...")
            repo_id_list.append(repo_id)

            repo_id_list.append(repo_id)
            data_loader_list.append(data_loader)
            num_batches_list.append(num_batches)

    list_len = len(repo_id_list)
    print(f"Total {list_len} datasets to process.")

    tasks = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx in range(list_len):
            tasks.append(
                executor.submit(
                    process_one,
                    idx,
                    repo_id_list[idx],
                    data_loader_list[idx],
                    config_list[idx],
                    num_batches_list[idx],
                    list_len,
                    max_frames,
                )
            )
        for f in as_completed(tasks):
            print("Done:", f.result())


if __name__ == "__main__":
    tyro.cli(main)
