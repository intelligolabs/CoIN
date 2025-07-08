#!/usr/bin/env python3

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import attr
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectGoalNavEpisode
from habitat.tasks.nav.object_nav_task import ObjectNavigationTask
from habitat.tasks.nav.object_nav_task import (
    ObjectViewLocation,
)
from habitat.core.dataset import EpisodeIterator

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.core.utils import not_none_validator


@attr.s(auto_attribs=True, kw_only=True)
class LanguageUncertaintyGoal(ObjectGoal):
    r"""translate in property certain attributes"""

    ##distractor - object of the same category
    distractors: Optional[List[ObjectGoal]] = None


@attr.s(auto_attribs=True, kw_only=True)
class LanguageUncertaintyNavEpisode(ObjectGoalNavEpisode):
    r""" """

    object_instance_id: Optional[int] = None

    # the complete one
    instruction: str = attr.ib(default=None)

    # input to the agent
    instruction_input: str = attr.ib(default=None)

    # camera spec - camera specification to retrieve the target goal image - useful to LLM oracle
    camera_spec: Dict[str, Any] = attr.ib(default=None, validator=not_none_validator)

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_instance_id}"


@registry.register_dataset(name="InstanceUncertaintyLanguageDataset-v1")
class LanguageUncertaintyDatasetV1(PointNavDatasetV1):
    r"""
    Class inherited from PointNavDataset that loads LanguageUncertainty dataset.
    """

    episodes: List[LanguageUncertaintyNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals: Dict[str, Sequence[LanguageUncertaintyGoal]]
    goals_by_instance: Dict[str, Sequence[ObjectGoal]]

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals = {}
        self.goals_by_instance = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            self.episodes[i].goals = [self.goals[self.episodes[i].goals_key]]

        return result

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals = {}
        for i, ep in enumerate(dataset["episodes"]):
            ep = LanguageUncertaintyNavEpisode(**ep)
            dataset["episodes"][i]["goals"] = []

        dataset["goals"] = goals

        return dataset

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> LanguageUncertaintyGoal:

        g = LanguageUncertaintyGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized["category_to_task_category_id"]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized["category_to_scene_annotation_category_id"]

        if len(deserialized["episodes"]) == 0:
            return

        for k, v in deserialized["goals"].items():
            self.goals_by_instance[k] = [self.__deserialize_goal(g) for g in v]

        self.goals = deserialized["goals"]

        for episode in deserialized["episodes"]:
            episode = LanguageUncertaintyNavEpisode(**episode)
            assert len(self.goals_by_instance[episode.goals_key]) == 1, f"More than one goal for {episode.goals_key}"

            episode.goals = self.goals_by_instance[episode.goals_key]
            episode.scene_id = DEFAULT_SCENE_PATH_PREFIX + episode.scene_id

            self.episodes.append(episode)

    def get_episode_iterator(self, *args: Any, **kwargs: Any):
        r"""Gets episode iterator with options. Options are specified in
        :ref:`EpisodeIterator` documentation.

        :param args: positional args for iterator constructor
        :param kwargs: keyword args for iterator constructor
        :return: episode iterator with specified behavior

        To further customize iterator behavior for your :ref:`Dataset`
        subclass, create a customized iterator class like
        :ref:`EpisodeIterator` and override this method.
        """
        # TODO: perform one iteration over the full dataset, than stop
        # kwargs['cycle'] = False
        return EpisodeIterator(self.episodes, *args, **kwargs)


@registry.register_task(name="InstanceUncertaintyLanguageTask-v1")
class InstanceUncertaintyLanguageTaskNavigationTask(ObjectNavigationTask):
    """A task for navigating to a specific object instance specified by a goal
    image. Built on top of ObjectNavigationTask. Used to explicitly state a
    type of the task in config.
    """
