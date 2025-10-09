import argparse
from pathlib import Path
import json
import os
import glob
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import subprocess


def process_repo(repo_id, dataset_root, base_dir, auto_choice="y"):
    raw_path = f"{dataset_root + base_dir}/{repo_id}"
    parent_dir = os.path.dirname(raw_path)

    hdf5_files = glob.glob(os.path.join(parent_dir, "**", "*.hdf5"), recursive=True)

    if hdf5_files:
        # 提取每个文件所在目录
        dirs = [os.path.dirname(f) for f in hdf5_files]

        # 统计每个目录中 .hdf5 文件的数量
        dir_counts = Counter(dirs)

        # 找出包含最多 .hdf5 文件的目录
        most_common_dir, count = dir_counts.most_common(1)[0]

        print(f"Found {len(hdf5_files)} .hdf5 files in total.")
        print(f"Directory with the most .hdf5 files: {most_common_dir} ({count} files)")
    else:
        raise FileNotFoundError(
            f"\033[91m⚠️ No .hdf5 files found in '{parent_dir}' or its subdirectories.\033[0m"
        )

    command = [
        "uv", "run", "s1_hd5_2_lerobot.py",
        "--raw-path", most_common_dir,
        "--image-compressed", "True",
        "--auto_input", auto_choice,
    ]

    print(f"[CMD] {' '.join(command)}")

    try:
        # stderr 合并到 stdout，避免死锁
        proc = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr 重定向到 stdout
            bufsize=1,
            universal_newlines=True,
        )

        # 实时读取输出
        for line in proc.stdout:
            print(f"[{repo_id}] {line}", end="")  # 输出前加 repo_id 标识

        proc.wait()

        if proc.returncode == 0:
            print(f"[DONE] {repo_id} finished successfully ✅")
        else:
            print(f"[FAIL] {repo_id} failed with exit code {proc.returncode} ❌")

        return repo_id, proc.returncode

    except Exception as e:
        print(f"[ERROR] {repo_id}: {e}")
        return repo_id, -1

def main():
    parser = argparse.ArgumentParser(description="Convert Aloha HD5 dataset and push to Hugging Face hub.")
    parser.add_argument("--json_dir", type=Path, required=False,
                        default=Path(
                            "configs/wrc_pnp_test.json"),
                        help="Path to JSON configuration file.")

    json_dir = parser.parse_args().json_dir
    print(f"json_path: {json_dir}")
    with open(json_dir, "r") as f:
        task_config = json.load(f)

    dataset_root = task_config["DatasetRootPath"]
    base_dir = task_config["base_dir"]
    repo_id_list = task_config["repo_id_list"]

    # 使用线程池并行处理
    max_workers = min(4, len(repo_id_list))  # 最多开 4 个线程
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_repo = {
            executor.submit(process_repo, repo, dataset_root, base_dir): repo
            for repo in repo_id_list
        }

        for future in as_completed(future_to_repo):
            repo_id = future_to_repo[future]
            try:
                _, exit_code = future.result()
            except Exception as e:
                print(f"[ERROR] {repo_id} raised exception: {e}")


if __name__ == "__main__":
    main()
