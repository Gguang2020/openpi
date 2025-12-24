import argparse
import glob
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def process_repo(repo_id: str, dataset_root: str, base_dir: str, auto_choice: str = "y") -> tuple[str, int]:
    """Process a single repository by converting HDF5 files to LeRobot format.

    Args:
        repo_id: Repository identifier.
        dataset_root: Root directory of the dataset.
        base_dir: Base directory relative to dataset_root.
        auto_choice: Automatic input choice for the conversion script (default: "y").

    Returns:
        Tuple of (repo_id, exit_code) indicating the processing result.

    Raises:
        FileNotFoundError: If no HDF5 files are found in the directory.
    """
    raw_path = f"{dataset_root + base_dir}/{repo_id}"
    parent_dir = os.path.dirname(raw_path)

    hdf5_files = glob.glob(os.path.join(parent_dir, "**", "*.hdf5"), recursive=True)

    if hdf5_files:
        # Extract the directory of each file
        dirs = [os.path.dirname(f) for f in hdf5_files]

        # Count the number of .hdf5 files in each directory
        dir_counts = Counter(dirs)

        # Find the directory with the most .hdf5 files
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
        # Merge stderr to stdout to avoid deadlock
        proc = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            bufsize=1,
            universal_newlines=True,
        )

        # Read output in real-time
        for line in proc.stdout:
            print(f"[{repo_id}] {line}", end="")  # Prefix output with repo_id

        proc.wait()

        if proc.returncode == 0:
            print(f"[DONE] {repo_id} finished successfully ✅")
        else:
            print(f"[FAIL] {repo_id} failed with exit code {proc.returncode} ❌")

        return repo_id, proc.returncode

    except Exception as e:
        print(f"[ERROR] {repo_id}: {e}")
        return repo_id, -1


def main() -> None:
    """Main function to batch convert HDF5 datasets to LeRobot format.

    Reads configuration from a JSON file and processes multiple repositories
    in parallel using a thread pool.
    """
    parser = argparse.ArgumentParser(description="Convert Aloha HDF5 dataset and push to Hugging Face hub.")
    parser.add_argument(
        "--json_dir",
        type=Path,
        required=False,
        default=Path("configs/wrc_pnp_test.json"),
        help="Path to JSON configuration file."
    )

    json_dir = parser.parse_args().json_dir
    print(f"json_path: {json_dir}")

    with open(json_dir) as f:
        task_config = json.load(f)

    dataset_root = task_config["DatasetRootPath"]
    base_dir = task_config["base_dir"]
    repo_id_list = task_config["repo_id_list"]

    # Use thread pool for parallel processing
    max_workers = min(4, len(repo_id_list))  # Maximum of 4 threads
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
