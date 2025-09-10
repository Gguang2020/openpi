"""See _CONFIGS for the list of available configs."""

import dataclasses
import difflib
import pathlib
from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

DatasetRootPath = "/media/xc/guang4T/astribot_dataset/"

def default_dataset_root() -> str:
    """Default location for the dataset cache."""
    return str(download.get_cache_dir() / "datasets")


@dataclasses.dataclass
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    repo_id_list: list[str] | None = None
    # Contains precomputed normalization stats.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    norm_stats_path: str | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Indicates where the cached dataset should be stored.
    dataset_root: str | None = dataclasses.field(default_factory=default_dataset_root)

    # If true, will disable syncing the dataset from the huggingface hub. Allows training on local-only datasets.
    local_files_only: bool = True



@runtime_checkable
class DataConfigFactory(Protocol):
    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        """Create a data config."""


class FakeDataConfig(DataConfigFactory):
    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        return DataConfig(repo_id="fake")


@dataclasses.dataclass(frozen=True)
class LeRobotS1DataConfig(DataConfigFactory):
    use_so3: list[bool] = field(default_factory=lambda: [False, False])
    so3_name_list: list[str] = field(default_factory=lambda: ["arm_left","gripper_left","arm_right","gripper_right"])
    so3_type: str = "torso_arm_gripper_head" #"torso_arm_gripper_head" "only_two_arm_gripper""move_torso_arm_gripper_head"
    image_mask: str = None
    head_img_name: str = None
    only_head_image: bool = False
    # The LeRobot repo id.
    repo_id: str = "lerobot_so3_data_30hz"
    repo_id_list: list[str] = None

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = False
    use_delta_so3_actions: str | None = None
    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # If true, will adapt the joint and gripper values to match the pi runtime. This useful when
    # fine-tuning a pretrained model.
    adapt_to_pi: bool = False
    # If true, will disable syncing the dataset from the huggingface hub.
    local_files_only: bool = True
    # Repack transforms. Default is used if not provided.
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None
    # Indicates where the cached dataset should be stored.
    dataset_root: str = DatasetRootPath + "s0_pick_and_place/vr_gripper_data/250305_1/lerobot_so3_data_30hz"
    # dataset_root: str = "/home/gguang/work/data/lerobot_so3_data_30hz/"
    norm_stats_path: str = None
    use_hand_cam_trans: bool = False


    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        norm_stats = None
        if self.repo_id_list is not None:

            # 提取每个字符串的 '/' 之前的部分
            prefix_list = [s.split('/')[-2] for s in self.repo_id_list]
            repo_ids = "-".join(prefix_list)  # 以空格拼接
            norm_stats_path = metadata_dir /repo_ids / "norm_stats.json"
            if norm_stats_path.exists():
               norm_stats = _normalize.deserialize_json(norm_stats_path.read_text())
            else:
                metadata_only_root = metadata_dir.parent
                name = metadata_dir.name
                name_temp = name.rsplit("_", 1)[0]
                norm_stats_list = []
                N_list = []
                for repo_id in self.repo_id_list:
                    repo_id_temp = pathlib.Path(repo_id)

                    first_repo = repo_id_temp.parts[0]  # 获取第一个路径部分（索引 1，索引 0 是 "/"）
                    rest_repo = pathlib.Path(*repo_id_temp.parts[1:])  # 剩余部分转换回 Path

                    norm_stats_path_temp = metadata_only_root / repo_id_temp / name_temp / "norm_stats.json"
                    if norm_stats_path_temp.exists():
                        norm_stats_temp = _normalize.deserialize_json(norm_stats_path_temp.read_text())
                        episodes_num_dir = pathlib.Path(self.dataset_root) / repo_id / "meta/info.json"
                        if episodes_num_dir.exists():
                            total_episodes =  json.loads(episodes_num_dir.read_text(encoding="utf-8"))["total_episodes"]
                            N_list.append(total_episodes)
                        else:
                            raise ValueError(f"episodes_num for {episodes_num_dir} not found")
                        # total_episodes =  "total_episodes": 382,
                        norm_stats_list.append(norm_stats_temp)
                    else:
                        raise ValueError(f"norm_stats for {norm_stats_path_temp} not found")
                norm_stats = dict()
                norm_stats_json = dict()
                for key in norm_stats_list[0].keys():
                    norm_stats[key] = []
                    mean_list = []
                    std_list = []
                    for i in range(len(norm_stats_list)):
                        mean_list.append(norm_stats_list[i][key].mean)
                        std_list.append(norm_stats_list[i][key].std)

                    # 转换为 NumPy 数组
                    mean_list = np.array(mean_list)
                    std_list = np.array(std_list)
                    N_list = np.array(N_list)

                    # 计算总均值（逐列计算）
                    total_mean = np.sum(N_list[:, None] * mean_list, axis=0) / np.sum(N_list)

                    # 计算总标准差（逐列计算）
                    total_std = np.sqrt(np.sum(N_list[:, None] * (std_list ** 2 + mean_list ** 2), axis=0) / np.sum(
                        N_list) - total_mean ** 2)
                    norm_stats_json[key] ={}
                    norm_stats_json[key]["mean"] = total_mean.tolist()
                    norm_stats_json[key]["std"] = total_std.tolist()
                    norm_stats[key] = _transforms.NormStats(mean=total_mean, std=total_std)

                # 确保目标目录存在
                norm_stats_path.parent.mkdir(parents=True, exist_ok=True)

                normalize.save(norm_stats_path.parent, norm_stats)
                # # 进行 JSON 写入
                # with norm_stats_path.open("w") as f:
                #     json.dump(norm_stats_json, f)
        elif self.repo_id is not tyro.MISSING:
            norm_stats_path = metadata_dir.parent / self.repo_id /metadata_dir.name / "norm_stats.json"
            print("--------------------------------------------")
            print(norm_stats_path)
            norm_stats = _normalize.deserialize_json(norm_stats_path.read_text()) if norm_stats_path.exists() else None


        if self.use_so3[0] :
            if self.so3_type == "move_torso_arm_gripper_head":
                state_map = "cartesian_so3_dict.cartesian_pose_state_move"
            else:
                state_map ="cartesian_so3_dict.cartesian_pose_state"
        else:
            state_map ="joints_dict.joints_position_state"

        if self.use_so3[1] :
            if self.so3_type == "move_torso_arm_gripper_head":
                action_map ="cartesian_so3_dict.cartesian_pose_command_move"
            else:
                action_map ="cartesian_so3_dict.cartesian_pose_command"
        else:
            action_map ="joints_dict.joints_position_command"
        image_head_map = "images_dict.head.rgb"
        if self.head_img_name is not None and self.head_img_name == "stereo":
            image_head_map = "images_dict.stereo.rgb"
        elif self.head_img_name is not None and self.head_img_name == "stereo_left":
            image_head_map = "images_dict.stereo_left.rgb"
        elif self.head_img_name is not None and self.head_img_name == "stereo_right":
            image_head_map = "images_dict.stereo_right.rgb"
        elif self.image_mask is not None:
            if self.image_mask == "head" or self.image_mask == "head_2":
                image_head_map = "images_dict.head_gripper_mask_2.rgb"
            elif self.image_mask == "head_1":
                image_head_map = "images_dict.head_gripper_mask_1.rgb"
            else:
                image_head_map = "images_dict.head_gripper_mask_2.rgb"

        if self.use_delta_so3_actions == "head_so3":
            head_key = "head_so3_poses"
            head_map = "xyz_so3_state.astribot_head",
        elif self.use_delta_joint_actions == "head_quat":
            head_key = "head_quat_poses"
            head_map = "xyz_quat_state.astribot_head",

        images_map = {
            "cam_high": image_head_map,
        }
        if not self.only_head_image:
            images_map["cam_left_wrist"] = "images_dict.left.rgb"
            images_map["cam_right_wrist"] = "images_dict.right.rgb"

        if "head" in self.use_delta_so3_actions.split("_"):
            repack_transforms = self.repack_transforms or _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": images_map,
                            "state": state_map,
                            "actions": action_map,
                            head_key: head_map,
                            # "prompt": "prompt",
                        }
                    )
                ]
            )
        else:
            repack_transforms = self.repack_transforms or _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": images_map,
                            "state": state_map,
                            "actions": action_map,
                            # "prompt": "prompt",
                        }
                    )
                ]
            )


        data_transforms = _transforms.Group(
            inputs=[s1_policy.S1Inputs(action_dim=model.action_dim, adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[0],so3_type = self.so3_type)],
            outputs=[s1_policy.S1Outputs(action_dim=model.action_dim,adapt_to_pi=self.adapt_to_pi, use_so3=self.use_so3[1],so3_type = self.so3_type)],
        )

        if self.use_delta_so3_actions is not None:

            if self.so3_type == "only_two_arm_gripper":
                delta_action_mask = [9, -1, 9, -1]
            elif self.so3_type == "torso_arm_gripper_head":
                delta_action_mask = [9, 9, -1, 9, -1, 2]
            elif self.so3_type == "move_torso_arm_gripper_head":
                delta_action_mask = [-3,9, 9, -1, 9, -1, 2]

            if self.only_head_image:
                state_trans_head = True
            else:
                state_trans_head = False

            if self.use_hand_cam_trans:
                hand_cam_trans_tpye = "right_hight"
            else:
                hand_cam_trans_tpye = None

            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions_so3(delta_action_mask,self.use_delta_so3_actions,state_trans_head,hand_cam_trans_tpye)],
                outputs=[_transforms.AbsoluteActions_so3(delta_action_mask,self.use_delta_so3_actions,state_trans_head,hand_cam_trans_tpye)],
            )

        elif self.use_delta_joint_actions:
            if self.use_so3[1]:

                if self.so3_type == "only_two_arm_gripper":
                    delta_action_mask = _transforms.make_bool_mask(9, -1, 9, -1)
                elif self.so3_type == "torso_arm_gripper_head":
                    delta_action_mask = _transforms.make_bool_mask(9, 9, -1, 9, -1,2)
                elif self.so3_type == "move_torso_arm_gripper_head":
                    delta_action_mask = _transforms.make_bool_mask(-3, 9, 9, -1, 9, -1,2)
            else:
                delta_action_mask = _transforms.make_bool_mask(4, 7, -1, 7, -1,2)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],

                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )


        return DataConfig(
            repo_id=self.repo_id,
            repo_id_list=self.repo_id_list,
            dataset_root=self.dataset_root,
            norm_stats=norm_stats,
            norm_stats_path=str(norm_stats_path),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=_transforms.Group(
                inputs=[
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model.max_token_len),
                        default_prompt=self.default_prompt,
                    ),
                ]
            ),
            local_files_only=self.local_files_only,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = False
    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # If true, will adapt the joint and gripper values to match the pi runtime. This useful when
    # fine-tuning a pretrained model.
    adapt_to_pi: bool = False
    # If true, will disable syncing the dataset from the huggingface hub.
    local_files_only: bool = True
    # Repack transforms. Default is used if not provided.
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None

    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        norm_stats = None
        if self.repo_id is not tyro.MISSING:
            norm_stats_path = metadata_dir / self.repo_id / "norm_stats.json"
            print("--------------------------------------------")
            print(norm_stats_path)
            norm_stats = _normalize.deserialize_json(norm_stats_path.read_text()) if norm_stats_path.exists() else None

        repack_transforms = self.repack_transforms or _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return DataConfig(
            repo_id=self.repo_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=_transforms.Group(
                inputs=[
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model.max_token_len),
                        default_prompt=self.default_prompt,
                    ),
                ]
            ),
            local_files_only=self.local_files_only,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Number of action dimensions.
    action_dim: int = 24
    # Number of action steps in the horizon.
    action_horizon: int = 50
    # Maximum token length for the prompt.
    max_token_len: int = 48

    # The Flax module representing the neural network implementation; must adhere to the BaseModule interface. We can put
    # it directly into the config like this because unbound Flax modules are just dataclasses.
    module: common.BaseModule = dataclasses.field(default_factory=pi0.Module)
    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for metadata (e.g., norm stats).
    metadata_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often to log training metrics.
    log_interval: int = 100
    # How often to save checkpoints.
    save_interval: int = 1000
    # How often to keep checkpoints.
    keep_interval: int = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # Keyword arguments to pass to the policy's sample method.
    sample_kwargs: dict[str, Any] | None = None

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def metadata_dir(self) -> pathlib.Path:
        """Get the metadata directory for this config."""
        return (pathlib.Path(self.metadata_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def create_model(self) -> _model.Model:
        """Create a model for this config."""
        return _model.Model(
            module=self.module,
            state_dim=self.action_dim,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
        )

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_CONFIGS = [
    #
    # pi0 configs.
    #

    TrainConfig(
        name="hand_right_pnp",  # "pi0_small_s1_so3","pi0_small_s1_so3_only_arm_gripper"
        metadata_base_dir=DatasetRootPath + "s1_pnp/WRC/hand_right_test/725/",
        # metadata_base_dir="/home/gguang/work/data/",
        data=LeRobotS1DataConfig(
            use_so3=[True, True],
            head_img_name="head",
            dataset_root=DatasetRootPath + "s1_pnp/WRC/hand_right_test/725/lerobot_so3_data_30hz",
            use_delta_joint_actions=True,
            use_delta_so3_actions="self",  # "self" "world"
            adapt_to_pi=False,
            # Set this to true if you are using a dataset that is not on the huggingface hub.
        ),

        action_dim=36,
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
        num_workers=1,
        num_train_steps=160_000,
        batch_size=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=160_000, decay_lr=2.5e-6
        ),
    ),

    TrainConfig(
        name="hand_right_pnp_hight",  # "pi0_small_s1_so3","pi0_small_s1_so3_only_arm_gripper"
        metadata_base_dir=DatasetRootPath + "s1_pnp/WRC/hand_right_test/728/",
        # metadata_base_dir="/home/gguang/work/data/",
        data=LeRobotS1DataConfig(
            use_so3=[True, True],
            head_img_name="head",
            dataset_root=DatasetRootPath + "s1_pnp/WRC/hand_right_test/728/lerobot_so3_data_30hz",
            use_delta_joint_actions=True,
            use_delta_so3_actions="self",  # "self" "world"
            adapt_to_pi=False,
            # Set this to true if you are using a dataset that is not on the huggingface hub.
        ),

        action_dim=36,
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
        num_workers=1,
        num_train_steps=160_000,
        batch_size=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=160_000, decay_lr=2.5e-6
        ),
    ),

    TrainConfig(
        name="hand_right_pnp_hight_trans_hand_cam",  # "pi0_small_s1_so3","pi0_small_s1_so3_only_arm_gripper"
        metadata_base_dir=DatasetRootPath + "s1_pnp/WRC/hand_right_test/728/",
        # metadata_base_dir="/home/gguang/work/data/",
        data=LeRobotS1DataConfig(
            use_so3=[True, True],
            head_img_name="head",
            dataset_root=DatasetRootPath + "s1_pnp/WRC/hand_right_test/728/lerobot_so3_data_30hz",
            use_delta_joint_actions=True,
            use_delta_so3_actions="self",  # "self" "world"
            adapt_to_pi=False,
            use_hand_cam_trans=True,
            # Set this to true if you are using a dataset that is not on the huggingface hub.
        ),

        action_dim=36,
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
        num_workers=1,
        num_train_steps=160_000,
        batch_size=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=160_000, decay_lr=2.5e-6
        ),
    ),

    TrainConfig(
        name="hand_right_pnp_hight_trans_hand_cam_list",  # "pi0_small_s1_so3","pi0_small_s1_so3_only_arm_gripper"
        metadata_base_dir=DatasetRootPath + "s1_pnp/WRC/hand_right_test",
        # metadata_base_dir="/home/gguang/work/data/",
        data=LeRobotS1DataConfig(
            use_so3=[True, True],
            head_img_name="head",
            dataset_root=DatasetRootPath + "s1_pnp/WRC/hand_right_test/",
            repo_id_list=["728/lerobot_so3_data_30hz", "729/lerobot_so3_data_30hz",
                          ],
            use_delta_joint_actions=True,
            use_delta_so3_actions="self",  # "self" "world"
            adapt_to_pi=False,
            use_hand_cam_trans=True,
            # Set this to true if you are using a dataset that is not on the huggingface hub.
        ),

        action_dim=36,
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
        num_workers=1,
        num_train_steps=160_000,
        batch_size=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=160_000, decay_lr=2.5e-6
        ),
    ),
    TrainConfig(
        name="vr_umi_single_pnp_list",  # "pi0_small_s1_so3","pi0_small_s1_so3_only_arm_gripper"
        metadata_base_dir=DatasetRootPath + "s1_pnp/vr_umi_test",
        # metadata_base_dir="/home/gguang/work/data/",
        data=LeRobotS1DataConfig(
            use_so3=[True, True],
            head_img_name="head",
            dataset_root=DatasetRootPath + "s1_pnp/vr_umi_test/",
            repo_id_list=["vr/single/lerobot_so3_data_30hz", "umi/single/lerobot_so3_data_30hz",
                          ],
            use_delta_joint_actions=True,
            use_delta_so3_actions="self",  # "self" "world"
            adapt_to_pi=False,
            # Set this to true if you are using a dataset that is not on the huggingface hub.
        ),

        action_dim=36,
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
        num_workers=1,
        num_train_steps=160_000,
        batch_size=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=160_000, decay_lr=2.5e-6
        ),
    ),

    TrainConfig(
        name="gripper_pnp",  # "pi0_small_s1_so3","pi0_small_s1_so3_only_arm_gripper"
        metadata_base_dir=DatasetRootPath + "s1_pnp/WRC/gripper_pnp/to_Cart/807_wrc_night/",
        # metadata_base_dir="/home/gguang/work/data/",
        data=LeRobotS1DataConfig(
            use_so3=[True, True],
            head_img_name="head",
            dataset_root=DatasetRootPath + "s1_pnp/WRC/gripper_pnp/to_Cart/807_wrc_night/lerobot_so3_data_30hz",
            use_delta_joint_actions=True,
            use_delta_so3_actions="self",  # "self" "world"
            adapt_to_pi=False,
            # Set this to true if you are using a dataset that is not on the huggingface hub.
        ),

        action_dim=36,
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
        num_workers=1,
        num_train_steps=160_000,
        batch_size=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=160_000, decay_lr=2.5e-6
        ),
    ),

    TrainConfig(
        name="gripper_pnp_list",  # "pi0_small_s1_so3","pi0_small_s1_so3_only_arm_gripper"
        metadata_base_dir= DatasetRootPath + "s1_pnp/WRC/gripper_pnp/to_Cart",
        # metadata_base_dir="/home/gguang/work/data/",
        data=LeRobotS1DataConfig(
            use_so3=[True, True],
            head_img_name="head",
            dataset_root= DatasetRootPath + "s1_pnp/WRC/gripper_pnp/to_Cart/",
            repo_id_list=["731_left_1toy/lerobot_so3_data_30hz","731_left_2toy/lerobot_so3_data_30hz",
                        "731_right_1toy/lerobot_so3_data_30hz","731_right_2toy/lerobot_so3_data_30hz","731_LR_2toy/lerobot_so3_data_30hz",
                        "801_adj_left/lerobot_so3_data_30hz","801_adj_right/lerobot_so3_data_30hz",
                        "801_hand/lerobot_so3_data_30hz","801_other_1/lerobot_so3_data_30hz","801_other_2/lerobot_so3_data_30hz",
                        "803_adj_74_1/lerobot_so3_data_30hz","803_adj_74_2/lerobot_so3_data_30hz",
                        "803_adj_75_1/lerobot_so3_data_30hz","803_adj_75_2/lerobot_so3_data_30hz",
                        "803_adj_76_1/lerobot_so3_data_30hz","803_adj_76_2/lerobot_so3_data_30hz",
                          ],
            use_delta_joint_actions=True,
            use_delta_so3_actions="self",  # "self" "world"
            adapt_to_pi=False,
            # Set this to true if you are using a dataset that is not on the huggingface hub.
        ),


        action_dim=36,
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
        num_workers=1,
        num_train_steps=160_000,
        batch_size=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=160_000, decay_lr=2.5e-6
        ),
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f"Did you mean '{closest[0]}'? " if closest else ""
        if closest:
            raise ValueError(f"Config '{config_name}' not found. Did you mean '{closest_str}'?")
        raise ValueError(f"Config '{config_name}' not found.")

    return _CONFIGS_DICT[config_name]
