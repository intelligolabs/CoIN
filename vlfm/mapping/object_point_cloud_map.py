# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Union
import cv2
import numpy as np
import open3d as o3d
from vlfm.utils.geometry_utils import (
    extract_yaw,
    get_point_cloud,
    transform_points,
    within_fov_cone,
)
from colorama import Fore
from colorama import init as init_colorama

init_colorama(autoreset=True)
from vlfm.brain.llm_brain_history import LLM_History
from vlfm.brain.vlm_brain_history import VLM_History
from vlfm.oracle.oracle import VLMOracle


class ObjectPointCloudMap:
    clouds: Dict[str, np.ndarray] = {}
    use_dbscan: bool = True

    def __init__(
        self,
        erosion_size: float,
        vlm_agent_brain,
        llm_agent_brain: LLM_History,
        vlm_oracle: VLMOracle,
    ) -> None:
        self._erosion_size = erosion_size
        self.last_target_coord: Union[np.ndarray, None] = None

        self.object_unique_id = 1
        self.detection_cloud = {}  # same logic as clouds, for understand if a detection is seen or not

        self.vlm_agent_brain: VLM_History = vlm_agent_brain
        self.llm_agent_brain: LLM_History = llm_agent_brain
        self.vlm_oracle: VLMOracle = vlm_oracle



    def reset(self, ep_id, target_obj) -> None:
        self.clouds = {}
        self.last_target_coord = None
        self.vlm_agent_brain.reset()
        self.llm_agent_brain.reset()
        self.vlm_oracle.reset()

    def has_object(self, target_class: str) -> bool:
        return target_class in self.clouds and len(self.clouds[target_class]) > 0

    def mask_target_image(self, target_image: np.ndarray, target_object_mask: np.ndarray) -> np.ndarray:

        blurred_img = cv2.GaussianBlur(target_image, (21, 21), sigmaX=40)

        mask = np.copy(target_object_mask)
        mask[mask == True] = 255
        mask = np.expand_dims(mask, axis=-1)

        mask = np.tile(mask, (1, 1, 3))

        out = np.where(mask == (255, 255, 255), target_image, blurred_img)
        return out

    def get_to_the_best_one(self, object_name: str) -> np.ndarray:
        """
        in the config yaml - max steps is 500
        If we reach max_step - offset, we have to go to the best detection
        """
        print(Fore.YELLOW + "Navigating to the best detection of object: ", object_name)
        global_cloud = self.llm_agent_brain.get_best_object_based_on_score()
        if not global_cloud is None:
            self.clouds[object_name] = (
                global_cloud  # we reset all the point cloud to the best one, so we can navigate to it
            )

    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        llama_promt,
        llava_prompt,
        rgb_image,
        target_object,
        total_num_steps: int,
        ep_id: int,
    ) -> None:
        """Updates the object map with the latest information from the agent."""
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy)
        if len(local_cloud) == 0:
            return

        # For second-class, bad detections that are too offset or out of range, we
        # assign a random number to the last column of its point cloud that can later
        # be used to identify which points came from the same detection.
        if too_offset(object_mask):
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
        else:
            # Mark all points of local_cloud whose distance from the camera is too far
            # as being out of range
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0  # 5% margin
            # All values of 1 in within_range will be considered within range, and all
            # values of 0 will be considered out of range; these 0s need to be
            # assigned with a random number so that they can be identified later.

            within_range = within_range.astype(np.float32)
            within_range[within_range == 0] = np.random.rand()
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)

        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = self._get_closest_point(global_cloud, curr_position)
        dist = np.linalg.norm(closest_point[:3] - curr_position)

        if dist < 1.0:
            # Object is too close to trust as a valid object
            return False

        # we also want to discard object that are too small (thus, too far)
        # we calculate the dimension of the object
        precentage_of_target_object_in_the_image = ((object_mask == 1).sum() / object_mask.size) * 100
        precentage_of_target_object_in_the_image = round(precentage_of_target_object_in_the_image, 2)
        if precentage_of_target_object_in_the_image < 4.0:  # target obj is less than 4% of the image dimension
            print(
                Fore.RED
                + f"Discarding object, too small compared to the image [{precentage_of_target_object_in_the_image}]"
            )
            return False
        # we reason only once per new detection, (due to API cost)
        # if it is the same detection we reasoned about, we update its position into the 3D world for better docking
        if self.is_detection_seen(global_cloud, object_name):
            print(Fore.YELLOW + "[INFO] This detection is already seen, skipping ####")

            # if it is the target goal, we update it's position to better docking
            if object_name in self.clouds and not too_offset(object_mask):
                if self.is_detection_seen(global_cloud, object_name, potential_target=True):
                    print(Fore.GREEN + "[INFO] Updating target position")
                    self.clouds[object_name] = global_cloud
            return False
        else:
            print(Fore.LIGHTBLUE_EX + "[INFO] This detection is new, continue with the LLM and LVLM logic")

        #  store in a separate DS the information about detection position
        if object_name in self.detection_cloud:
            self.detection_cloud[object_name] = np.concatenate(
                (self.detection_cloud[object_name], global_cloud), axis=0
            )
        else:
            self.detection_cloud[object_name] = global_cloud

        ####### LMM and VLM LOGIC - as described in the paper
        # 1. Self-questioner
        # 1.1 Generate init description of the image with VLM
        # 1.2 Retrieve more facts about the detected object
        # 1.3 Perception uncertainty estimation
        # 1.4 Detection description refinement.

        # 2. Interaction trigger
        #####

        ## hyperparams
        ALLOWS_SKIP_QUESTION_MODULE = True  # allows to skip question if the similarity score is below a certain threshold, thus reducing the overall amount of question to the human
        MAX_HUMAN_QUESTION_FOR_EACH_DETECTED_OBJ = 4
        THRESHOLD_STOP_SCORE = 7
        THRESHOLD_SKIP_QUESTION = 5
        TAU = 0.75  # see paper for more information

        ##### Self-questioner
        #######
        # generate a description of the current observation the on-board VLM
        distractor_description = self.vlm_agent_brain.get_description_of_the_image(rgb_image, prompt=llava_prompt)

        # retrieve more questions to self-ask regarding the detected object using LLM. These questions are open-ended
        questions_for_detected_object_by_retrieving_more_fact = (
            self.llm_agent_brain.retrieving_more_facts_about_detected_object(distractor_description, target_object)
        )

        # answer the questions using the on-board VLM
        questions_and_answers_for_detected_object_by_retrieving_more_fact = self.vlm_oracle.answer_question_given_image(
            questions_for_detected_object_by_retrieving_more_fact,
            ARE_QUESTIONS_FOR_THE_ORACLE=False,  # we use the on-board VLM, not the VLM-simulated user
            USE_LLM_TO_CHECK_THE_ANSWER=False,  # deprecated
            image_to_be_used=rgb_image,  # we use the current detection
            perform_logits_likelihood=False,  # we don't want to get the prob. distrubtion over the vocabulary now. These questions are open-ended
        )

        # update the distractor description with the updated questions/answers pairs
        for item in questions_and_answers_for_detected_object_by_retrieving_more_fact:
            question, answer = item["question"], item["answer"]
            distractor_description += answer

        # now we want to retrieve self-questions to check the uncertainty of the distractor description. The questions will end with 'Answer with Yes, No, or I don't know'
        self_questioner_questions = self.llm_agent_brain.generate_self_questioner_question_given_distractor_description(
            distractor_description, target_object
        )

        # retrieve the answers uncertainty using the on-board VLM
        self_questioner_answers = self.vlm_oracle.answer_question_given_image(
            self_questioner_questions,
            ARE_QUESTIONS_FOR_THE_ORACLE=False,
            USE_LLM_TO_CHECK_THE_ANSWER=False,
            image_to_be_used=rgb_image,
            perform_logits_likelihood=True,  # note here the param is set to True, we want to get the prob. distrubtion over the vocabulary
        )

        # filter the questions based on the uncertainty of the answers
        # if uncertain, we do not discard here the question, but we put a label of uncertainty
        self_questioner_answers = self.llm_agent_brain.filter_self_questioner_answer_by_uncertainty(
            self_questioner_answers, tau=TAU
        )

        # refine the description using the self-questioner question/answer/uncertainty pairs
        distractor_description, detected_image_attributes = (
            self.llm_agent_brain.refine_image_description_after_self_questioner(
                self_questioner_question_answers_uncertainty=self_questioner_answers,
                target_object=target_object,
                distractor_object_description=distractor_description,
            )
        )

        ##### Interaction Trigger
        #######
        for max_human_interaction_step in range(MAX_HUMAN_QUESTION_FOR_EACH_DETECTED_OBJ):

            # get a similarity score between the target object and the detected object, and a candidate question to the user
            # at first iteration (when the facts are empty, the similarity score will be -1)
            similarity_score_detected_to_target, questions_for_target_object = (
                self.llm_agent_brain.get_similarity_score_and_question_for_target_object(
                    target_object=target_object, detected_object_description=distractor_description
                )
            )

            if (
                ALLOWS_SKIP_QUESTION_MODULE
                and int(similarity_score_detected_to_target) < THRESHOLD_SKIP_QUESTION
                and int(similarity_score_detected_to_target) != -1
            ):
                print(Fore.RED + "[INFO: Agent] The detected object is too different from the target object, SKIP QUESTION and continue navigation...")
                return False

            if int(similarity_score_detected_to_target) != -1:

                information_to_be_saved = dict(
                    object_stop_score=int(similarity_score_detected_to_target),
                    object_map_position=global_cloud,
                    rgb_image=rgb_image,
                    rgb_image_description=distractor_description,
                )
                self.llm_agent_brain.store_information_about_detected_object(
                    f"{target_object}_{self.object_unique_id}", information_to_be_saved, PRINT_INFO=True
                )
                self.object_unique_id += 1

                # test if this is the target object the user is looking for.
                if int(similarity_score_detected_to_target) >= THRESHOLD_STOP_SCORE:
                    print(Fore.GREEN + "The detected object is similar to the target object")

                    break  # we exit the loop, thus add obj to point cloud ecc...

            # we have a question for the VLM-simulated user, we ask it
            oracle_answers_target_image = self.vlm_oracle.answer_question_given_image(
                questions_for_target_object,
                ARE_QUESTIONS_FOR_THE_ORACLE=True,  # here we use the VLM-simulated user, not the on-board VLM
                USE_LLM_TO_CHECK_THE_ANSWER=False,
                image_to_be_used=None,  # we do not use the image here, but the instance image (oracle)
                ep_id=ep_id,
            )

            # we asked a question regarding the target object, we retrieved an answer. We thus update the facts about the target object
            self.llm_agent_brain.updates_known_facts_about_target_object_given_oracle_answers(
                target_object=target_object, oracle_questions_answers=oracle_answers_target_image
            )

        else:
            print(Fore.YELLOW + "[INFO] Reached the max number of questions, moving on...")
            return False  # we have reached the max number of questions, we move on

        # if we are here, the candidate object seems similar to the target object, we add it to the point cloud map as a goal to reach
        print(Fore.GREEN + "Found a possible match, inserting the target into the map")
        if object_name in self.clouds:
            self.clouds[object_name] = np.concatenate((self.clouds[object_name], global_cloud), axis=0)
        else:
            self.clouds[object_name] = global_cloud
        return True

    def is_detection_seen(self, new_detection, target_class, potential_target=False):
        """
        return true if new detection is too close to any seen detection, thus we can skip LLM call
        """
        if target_class in self.detection_cloud:
            # we have multiple object, so handle tha case
            target_cloud = self.get_target_cloud_for_checking_seen_detection(target_class, potential_target)

            distances = np.linalg.norm(target_cloud[:, :3] - new_detection[:, :3][:, None], axis=2)
            seen_threshold = 1.5
            condition = np.any(distances < seen_threshold)

            if condition:
                return True
        return False

    def get_best_object(self, target_class: str, curr_position: np.ndarray) -> np.ndarray:
        target_cloud = self.get_target_cloud(target_class)

        closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2]
        # return None

        if self.last_target_coord is None:
            self.last_target_coord = closest_point_2d
        else:
            # Do NOT update self.last_target_coord if:
            # 1. the closest point is only slightly different
            # 2. the closest point is a little different, but the agent is too far for
            #    the difference to matter much
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
            if delta_dist < 0.1:
                # closest point is only slightly different
                return self.last_target_coord
            elif delta_dist < 0.5 and np.linalg.norm(curr_position - closest_point_2d) > 2.0:
                # closest point is a little different, but the agent is too far for
                # the difference to matter much
                return self.last_target_coord
            else:
                self.last_target_coord = closest_point_2d

        return self.last_target_coord

    def update_explored(self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float) -> None:
        """
        This method will remove all point clouds in self.clouds that were originally
        detected to be out-of-range, but are now within range. This is just a heuristic
        that suppresses ephemeral false positives that we now confirm are not actually
        target objects.

        Args:
            tf_camera_to_episodic: The transform from the camera to the episode frame.
            max_depth: The maximum distance from the camera that we consider to be
                within range.
            cone_fov: The field of view of the camera.
        """
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        for obj in self.clouds:
            within_range = within_fov_cone(
                camera_coordinates,
                camera_yaw,
                cone_fov,
                max_depth * 0.5,
                self.clouds[obj],
            )
            range_ids = set(within_range[..., -1].tolist())
            for range_id in range_ids:
                if range_id == 1:
                    # Detection was originally within range
                    continue
                # Remove all points from self.clouds[obj] that have the same range_id
                self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]

    def get_target_cloud(self, target_class: str) -> np.ndarray:
        target_cloud = self.clouds[target_class].copy()
        # Determine whether any points are within range
        within_range_exists = np.any(target_cloud[:, -1] == 1)
        if within_range_exists:
            # Filter out all points that are not within range
            target_cloud = target_cloud[target_cloud[:, -1] == 1]
        return target_cloud

    def get_target_cloud_for_checking_seen_detection(self, target_class: str, potential_target) -> np.ndarray:
        """
        ssame as above, but work with a separate Datastructure, thus detection are not save as target object,
        but as detection for miniminzing LLM and VLM call
        """
        if potential_target:
            # return the cloud we use for detection, not the ds we use to minimize llm call
            target_cloud = self.clouds[target_class].copy()
        else:
            target_cloud = self.detection_cloud[target_class].copy()
        # Determine whether any points are within range
        within_range_exists = np.any(target_cloud[:, -1] == 1)
        if within_range_exists:
            # Filter out all points that are not within range
            target_cloud = target_cloud[target_cloud[:, -1] == 1]
        return target_cloud

    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        final_mask = object_mask * 255
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)  # type: ignore

        valid_depth = depth.copy()
        valid_depth[valid_depth == 0] = 1  # set all holes (0) to just be far (1)
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy)
        cloud = get_random_subarray(cloud, 5000)
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud)

        return cloud

    def _get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray:
        ndim = curr_position.shape[0]
        if self.use_dbscan:
            # Return the point that is closest to curr_position, which is 2D
            closest_point = cloud[np.argmin(np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1))]
        else:
            # Calculate the Euclidean distance from each point to the reference point
            if ndim == 2:
                ref_point = np.concatenate((curr_position, np.array([0.5])))
            else:
                ref_point = curr_position
            distances = np.linalg.norm(cloud[:, :3] - ref_point, axis=1)

            # Use argsort to get the indices that would sort the distances
            sorted_indices = np.argsort(distances)

            # Get the top 20% of points
            percent = 0.25
            top_percent = sorted_indices[: int(percent * len(cloud))]
            try:
                median_index = top_percent[int(len(top_percent) / 2)]
            except IndexError:
                median_index = 0
            closest_point = cloud[median_index]
        return closest_point


def open3d_dbscan_filtering(points: np.ndarray, eps: float = 0.2, min_points: int = 100) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps, min_points))

    # Count the points in each cluster
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Exclude noise points, which are given the label -1
    non_noise_labels_mask = unique_labels != -1
    non_noise_labels = unique_labels[non_noise_labels_mask]
    non_noise_label_counts = label_counts[non_noise_labels_mask]

    if len(non_noise_labels) == 0:  # only noise was detected
        return np.array([])

    # Find the label of the largest non-noise cluster
    largest_cluster_label = non_noise_labels[np.argmax(non_noise_label_counts)]

    # Get the indices of points in the largest non-noise cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # Get the points in the largest non-noise cluster
    largest_cluster_points = points[largest_cluster_indices]

    return largest_cluster_points


def visualize_and_save_point_cloud(point_cloud: np.ndarray, save_path: str) -> None:
    """Visualizes an array of 3D points and saves the visualization as a PNG image.

    Args:
        point_cloud (np.ndarray): Array of 3D points with shape (N, 3).
        save_path (str): Path to save the PNG image.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z, c="b", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig(save_path)
    plt.close()


def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    """
    This function returns a subarray of a given 3D points array. The size of the
    subarray is specified by the user. The elements of the subarray are randomly
    selected from the original array. If the size of the original array is smaller than
    the specified size, the function will simply return the original array.

    Args:
        points (numpy array): A numpy array of 3D points. Each element of the array is a
            3D point represented as a numpy array of size 3.
        size (int): The desired size of the subarray.

    Returns:
        numpy array: A subarray of the original points array.
    """
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]


def too_offset(mask: np.ndarray) -> bool:
    """
    This will return true if the entire bounding rectangle of the mask is either on the
    left or right third of the mask. This is used to determine if the object is too far
    to the side of the image to be a reliable detection.

    Args:
        mask (numpy array): A 2D numpy array of 0s and 1s representing the mask of the
            object.
    Returns:
        bool: True if the object is too offset, False otherwise.
    """
    # Find the bounding rectangle of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Calculate the thirds of the mask
    third = mask.shape[1] // 3

    # Check if the entire bounding rectangle is in the left or right third of the mask
    if x + w <= third:
        # Check if the leftmost point is at the edge of the image
        # return x == 0
        return x <= int(0.05 * mask.shape[1])
    elif x >= 2 * third:
        # Check if the rightmost point is at the edge of the image
        # return x + w == mask.shape[1]
        return x + w >= int(0.95 * mask.shape[1])
    else:
        return False
