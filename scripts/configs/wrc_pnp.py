from openpi.training.config import TrainConfig
from openpi.training import optimizer as _optimizer
from openpi.data.lerobot_astribot import LeRobotAstribotDataConfig
from openpi.data.common import AssetsConfig
from openpi.models import pi0

CONFIG = TrainConfig(
    name="wrc_pnp_pi0",
    model=pi0.Pi0Config(),
    data=LeRobotAstribotDataConfig(
        assets=AssetsConfig(
            assets_dir="/kpfs-regular/gg/gg/data/s1_pnp/WRC/gripper_pnp/to_Cart/731_left_1toy",
            asset_id="so3_data_30hz/lerobot_astribot",
        ),
        default_prompt="pick the object and place it in the shopping cart",
        dataset_root="/kpfs-regular/gg/data/s1_pnp/WRC/WRC/gripper_pnp/to_Cart",
        repo_id_list=[
            "731_left_1toy/lerobot_so3_data_30hz",
            "731_right_1toy/lerobot_so3_data_30hz",
            "804_gg_left_adj/lerobot_so3_data_30hz",
            "804_gg_right_adj/lerobot_so3_data_30hz",
            "804_gg_many/lerobot_so3_data_30hz",
            "804_gg_green/lerobot_so3_data_30hz",
            "804_zhanting_1/lerobot_so3_data_30hz",
            "804_zhanting_2/lerobot_so3_data_30hz",
            "805_gg_adj_left/lerobot_so3_data_30hz",
            "805_gg_adj_right/lerobot_so3_data_30hz",
            "806_wrc_adj_1/lerobot_so3_data_30hz",
            "806_wrc_adj_2/lerobot_so3_data_30hz",
            "806_wrc_adj_3/lerobot_so3_data_30hz",
            "807_wrc/lerobot_so3_data_30hz",
            "807_wrc_night/lerobot_so3_data_30hz",
        ],
        local_files_only=True,
    ),
    batch_size=32,
    num_workers=32,
    num_train_steps=160_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=60_000, decay_lr=0.5e-6
    ),
)
