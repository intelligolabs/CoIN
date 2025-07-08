from habitat.core.simulator import Sensor, SensorTypes
from typing import Any, Optional
from habitat.core.registry import registry
import numpy as np
from gym import spaces
import habitat_sim
from habitat_sim.agent.agent import AgentState, SixDOFPose
from habitat_sim import bindings as hsim
from PIL import Image


@registry.register_sensor
class ImageGoalSensor(Sensor):
    r""" """

    cls_uuid: str = "instance_imagegoal"
    IMAGE_SIZE = (1024, 1024)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._current_episode_id = None
        self._current_image_goal = None
        self._sim = kwargs["sim"]

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=255, shape=(*self.IMAGE_SIZE, 3), dtype=np.uint8)

    def _add_sensor(self, sensor_uuid, camera_spec):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = self.IMAGE_SIZE

        if not camera_spec["hfov"] == -1:  # we set it
            spec.hfov = camera_spec["hfov"]

        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sim.add_sensor(spec)

        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()

        agent_state_new = AgentState(
            position=agent_state.position,
            rotation=agent_state.rotation,
            sensor_states={
                **agent_state.sensor_states,
                sensor_uuid: SixDOFPose(
                    position=np.array(camera_spec["position"]),
                    rotation=camera_spec["rotation"],
                ),
            },
        )
        agent.set_state(
            agent_state_new,
            infer_sensor_states=False,
        )

    def _remove_sensor(self, sensor_uuid: str) -> None:
        agent = self._sim.get_agent(0)
        del self._sim._sensors[sensor_uuid]
        hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
        del agent._sensors[sensor_uuid]
        agent.agent_config.sensor_specifications = [
            s for s in agent.agent_config.sensor_specifications if s.uuid != sensor_uuid
        ]

    def get_observation(
        self,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        """
        we create a temporary sensors
        """
        self.sim = kwargs["task"]._sim

        episode_uniq_id = f"{episode.goals_key}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        sensor_uuid = f"{self.cls_uuid}_tmp_sensor"
        camera_spec = episode.camera_spec

        # if we have a viewpoint, better to use get_observation_at
        if camera_spec["from_viewpoint"]:
            self._current_image_goal = self.sim.get_observations_at(camera_spec["position"], camera_spec["rotation"])[
                "rgb"
            ]
            return self._current_image_goal

        # otherwise create sensors
        self._add_sensor(sensor_uuid, camera_spec)

        # otherwisise we retrieve it from the dataset, since episode is new
        self._sim._sensors[sensor_uuid].draw_observation()
        self._current_image_goal = self._sim._sensors[sensor_uuid].get_observation()[:, :, :3]

        self._remove_sensor(sensor_uuid)
        # print("Reloading sensors suite")
        self._current_episode_id = episode_uniq_id  # cache the current image
        h, w = self.IMAGE_SIZE
        Image.fromarray(self._current_image_goal).save(f"instance_image_{h}_{w}.png")
        return self._current_image_goal
