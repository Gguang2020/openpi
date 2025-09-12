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

import openpi.shared.normalize as normalize
import openpi.training.config_auto as _config
import openpi.training.data_loader as _data_loader


def create_dataset(config: _config.TrainConfig) -> tuple[str, _data_loader.Dataset]:
    model = config.create_model()
    data_config = config.data.create(config.metadata_dir, model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.TransformedDataset(
        _data_loader.create_dataset(data_config, model),
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )
    return data_config.repo_id, dataset


def process_one(idx: int, repo_id: str, dataset, config, list_len: int, max_frames: int | None = None):
    metadata_name = config.metadata_base_dir.split("/")[-1]
    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(
        data_loader,
        total=num_frames,
        desc=f"Computing stats for {idx+1}/{list_len} {metadata_name}",
        position=idx,
        leave=False,
    ):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = config.metadata_dir.parent / repo_id / config.metadata_dir.name
    print(f"[{idx+1}/{list_len}] Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    return str(output_path)


def main(config_name: str, max_frames: int | None = None, workers: int = 4):
    config = _config.get_config(config_name)
    repo_id_list, dataset_list, config_list = [], [], []

    if config.data.repo_id_list is not None:
        for repo_id in config.data.repo_id_list:
            config_temp = copy.deepcopy(config)
            repo_id_force = "/".join(repo_id.split("/")[:-1])
            repo_id_end = repo_id.split("/")[-1]
            config_temp = replace(
                config_temp,
                data=replace(
                    config_temp.data,
                    repo_id=repo_id_end,
                    repo_id_list=None,
                    dataset_root="/".join([config_temp.data.dataset_root, repo_id]),
                ),
                metadata_base_dir="/".join([config.metadata_base_dir, repo_id_force]),
                name=config.name.replace("_list", ""),
            )

            norm_stats_path = (
                config_temp.metadata_dir.parent
                / config_temp.data.repo_id
                / config_temp.metadata_dir.name
                / "norm_stats.json"
            )
            if norm_stats_path.exists():
                print(f"---Norm stats already exists for {repo_id}, skipping...")
                continue

            print(f"---Processing {repo_id}...")
            repo_id_new, dataset = create_dataset(config_temp)
            repo_id_list.append(repo_id_new)
            dataset_list.append(dataset)
            config_list.append(config_temp)
    else:
        repo_id, dataset = create_dataset(config)
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
            dataset_list.append(dataset)
            config_list.append(config)

    list_len = len(repo_id_list)
    print(f"Total {list_len} datasets to process.")

    tasks = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx in range(list_len):
            tasks.append(
                executor.submit(
                    process_one,
                    idx,
                    repo_id_list[idx],
                    dataset_list[idx],
                    config_list[idx],
                    list_len,
                    max_frames,
                )
            )
        for f in as_completed(tasks):
            print("Done:", f.result())


if __name__ == "__main__":
    tyro.cli(main)
