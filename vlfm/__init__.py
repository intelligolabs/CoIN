# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

from habitat.config.default_structured_configs import LabSensorConfig
from vlfm.measurements import distractor_success

cs = ConfigStore.instance()
@dataclass
class ImageGoalSensorSensorConfig(LabSensorConfig):
    type: str = "ImageGoalSensor"
    image_cache_encoder: str = ""


cs.store(
    package=f"habitat.task.lab_sensors.instance_imagegoal_sensor",
    group="habitat/task/lab_sensors",
    name="instance_imagegoal_sensor",
    node=ImageGoalSensorSensorConfig,
)