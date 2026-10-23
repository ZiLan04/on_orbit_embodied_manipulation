"""Single-camera Starlink cube delta-EEF dataset configuration."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class StarlinkCubeDeltaEefDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.base_view"]
    state_keys = ["state.eef_position", "state.eef_rotation", "state.gripper"]
    action_keys = ["action.delta_eef_position", "action.delta_eef_rotation", "action.gripper_close"]
    action_key_dims = {"action.delta_eef_position": 3, "action.delta_eef_rotation": 3, "action.gripper_close": 1}
    state_key_dims = {"state.eef_position": 3, "state.eef_rotation": 3, "state.gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={key: "min_max" for key in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.delta_eef_position": "min_max",
                "action.delta_eef_rotation": "min_max",
                "action.gripper_close": "binary",
            }),
        ])


ROBOT_TYPE_CONFIG_MAP = {"starlink_cube_delta_eef": StarlinkCubeDeltaEefDataConfig()}
DATASET_NAMED_MIXTURES = {"starlink_cube_vla": [("lerobot", 1.0, "starlink_cube_delta_eef")]}