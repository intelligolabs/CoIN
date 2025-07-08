# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os

# The following imports require habitat to be installed, and despite not being used by
# this script itself, will register several classes and make them discoverable by Hydra.
# This run.py script is expected to only be used when habitat is installed, thus they
# are hidden here instead of in an __init__.py file. This avoids import errors when used
# in an environment without habitat, such as when doing real-world deployment. noqa is
# used to suppress the unused import and unsorted import warnings by ruff.
import frontier_exploration  # noqa: F401
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401

# import custom task and sensors information
import vlfm.sensors.instance_image_nav # noqa: F401

import vlfm.uncertainty_task.uncertainty_task # noqa: F401
from vlfm.visualizations import maps, top_down_map
from habitat.config.default_structured_configs import (
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)

# silence habitat
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/uncertainty_objectnav_hm3d",
)
def main(cfg: DictConfig) -> None:
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run the following command first:")
        print("python -m vlfm.utils.generate_dummy_policy")
        exit(1)

    cfg = patch_config(cfg)
    with read_write(cfg):
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass

        try:
            cfg.habitat.task.measurements.frontier_exploration_map.draw_goal_aabbs = False
        except:
            raise ValueError("Error in updating top_down_map measurement config")

    print(
        "############################## CONFIG success_distance", cfg.habitat.task.measurements.success.success_distance
    )
    print("############################## CONFIG split", cfg.habitat_baselines.eval.split)
    execute_exp(cfg, "eval")


if __name__ == "__main__":
    main()
