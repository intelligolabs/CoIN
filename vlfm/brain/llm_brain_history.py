from colorama import Fore
from colorama import init as init_colorama
from retrying import retry

init_colorama(autoreset=True)
import yaml
import vlfm.utils.prompts as prompts
import re
from typing import List, Dict, Any
import torch
from copy import deepcopy
import logging


class LLM_History:
    """
    This class store the connector for the LLM and it's past conversation.
    """

    def __init__(self, llm_client) -> None:
        self.LLM_CLIENT = llm_client
        self.ep_id = None

        self.questions_and_answers_database: List[dict] = (
            []
        )  # contains a list of question and answer. This provides more context to the LLM to reason about the score

        self.target_object_informations = ""  # this variable maintains the information about the target object, mimicking what's happening in real-life situation

        # the following variable maintains a graph about object found -> scores associated to it and their
        self.objects_graph_informations: Dict[str, Dict[str, Any]] = {}


    def get_best_object_based_on_score(self):
        """
        Get the object with the highest score
        """
        best_object = None
        best_score = -1
        for object_id, object_informations in self.objects_graph_informations.items():
            if object_informations["object_stop_score"] > best_score:
                best_object = object_informations["object_map_position"]
                best_score = object_informations["object_stop_score"]
        return best_object

    def store_information_about_detected_object(
        self, object_id: str, object_informations: Dict[str, Any], PRINT_INFO=False
    ):
        """
        Store information about the detected object
        Minimum information to store are:
            - object_id: unique identifier for the object
            - object position in the map
            - object score for deciding stop navigation
        """
        if object_id is None:
            raise ValueError("object_id cannot be None")
        if "object_map_position" not in object_informations:
            raise ValueError("object_map_position is required")
        if "object_stop_score" not in object_informations:
            raise ValueError("object_stop_score is required")

        self.objects_graph_informations[object_id] = object_informations
        if PRINT_INFO:
            print(Fore.LIGHTMAGENTA_EX + "######### INFO: [ Object saved into the map]")
            print(
                Fore.LIGHTMAGENTA_EX
                + f"######### INFO: [ {object_id}: Score -> {object_informations['object_stop_score']}"
            )


    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def generate_self_questioner_question_given_distractor_description(
        self, distractor_description, target_object
    ) -> List[str]:
        """
        self question module new pipeline. Generate up to x quesiton, with uncertainty estimation.
        """
        prompt = prompts.LLM_SELF_QUESTIONER_GIVEN_DISTRACTOR_DESCRIPTION.format(
            distractor_object_description=distractor_description,
            target_object=target_object,
            facts_about_the_target_picture=self.target_object_informations or "No information available",
            uncertain_answer_choice_placeholder=prompts.UNCERTAIN_ANSWER_CHOICE_PLACEHOLDER,
        )

        response = self.LLM_CLIENT.ask(prompt=prompt)
        print(
            Fore.YELLOW + "[INFO: LLM] Generate self-questions to be answer with uncertainty estimation \n" + response
        )
        print("---" * 5)
        try:
            yaml_start = response.find("YAML_START")
            yaml_end = response.find("YAML_END")
            if yaml_start != -1 and yaml_end != -1:
                yaml_string = response[yaml_start + len("YAML_START") : yaml_end].strip()
                parsed_yaml = yaml.safe_load(yaml_string)

                # Extract questions for target objects
                questions_for_detected_object = [
                    self_question.strip() for self_question in parsed_yaml["questions_for_detected_object"].values()
                ]
            else:
                print("YAML section not found in the string.")
                raise Exception("YAML section not found in the string.")
            return questions_for_detected_object

        except Exception as e:
            logging.exception(response)
            logging.info("This was the prompt")
            logging.exception(prompt)
            print(e)
            raise e

    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def retrieving_more_facts_about_detected_object(self, distractor_description, target_object) -> List[str]:
        """
        retrieving_more_facts_about_detected_object
        """
        prompt = prompts.LMM_RETRIEVE_FACTS_FROM_DESCRIPTION
        prompt = prompt.format(
            target_object=target_object,
            distractor_object_description=distractor_description,
            facts_about_the_target_picture=self.target_object_informations or "No information available",
        )

        response = self.LLM_CLIENT.ask(prompt=prompt)
        print(
            Fore.YELLOW
            + "[INFO: LLM] Retrieving more facts about the detected object (open ended questions). \n"
            + response
        )
        print("---" * 5)
        try:
            yaml_start = response.find("YAML_START")
            yaml_end = response.find("YAML_END")

            if yaml_start != -1 and yaml_end != -1:
                yaml_string = response[yaml_start + len("YAML_START") : yaml_end].strip()
                parsed_yaml = yaml.safe_load(yaml_string)

                questions_for_detected_object = [
                    question_for_detected_object.strip()
                    for question_for_detected_object in parsed_yaml["questions"].values()
                ]

                return questions_for_detected_object
            else:
                print("YAML section not found in the string.")
                raise Exception("YAML section not found in the string.")
        except Exception as e:
            print(e)
            logging.exception(response)
            logging.info("This was the prompt")
            logging.exception(prompt)
            raise e

    # max retry 10 times, wait 10 seconds between each retry
    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def filter_self_questioner_answer_by_uncertainty(
        self, self_questioner_question_answers_lilekihood_pairs: List[dict], tau=0.5, offset=None
    ):
        """
        We performs the self-questioning mechanismi to the LVLM, and obtains question, answer and the likelihood of the answer.
        We now need to filter this.
        """
        results = deepcopy(self_questioner_question_answers_lilekihood_pairs)
        for i, item in enumerate(self_questioner_question_answers_lilekihood_pairs):
            question, answer, tokens_likelihood = item["question"], item["answer"], item["logits_likelihood"] # here we are not interested in the answer, we look at the prob. distribution of the tokens

            tokens_probs = torch.tensor([post_prob for answer, post_prob in tokens_likelihood])
            assert len(tokens_probs) == 3, "We should have 3 tokens in the likelihood distribution (yes, no, uncertain)"

            # compute the entropy
            entropy = -torch.sum(tokens_probs * torch.log(tokens_probs))
            entropy_max = torch.log(torch.tensor(len(tokens_probs)))
            entropy_normalized = entropy / entropy_max
            if offset is None:
                label = "certain" if entropy_normalized <= tau else "uncertain"
            else:
                label = "certain" if entropy_normalized - offset <= tau else "uncertain"
            results[i]["certainty_label"] = label
            results[i]["normalized_entropy_value"] = entropy_normalized.cpu().item()
        return results

    # max retry 10 times, wait 10 seconds between each retry
    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def refine_image_description_after_self_questioner(
        self,
        self_questioner_question_answers_uncertainty: List[dict],
        target_object: str,
        distractor_object_description: str,
    ):
        """
        We have the answer from the self-questioner. We can now refine the image description.
        """
        questions_answer_string = ""
        for item in self_questioner_question_answers_uncertainty:
            question, answer, certainty_label = item["question"], item["answer"], item["certainty_label"]
            questions_answer_string += f"- Question: {question} - Answer: {answer} - Certainty: {certainty_label}\n"

        prompt = prompts.LLM_REFINE_DETECTED_OBJECT_DESCRIPTION.format(
            target_object=target_object,
            distractor_object_description=distractor_object_description,
            list_questions_answers_uncertainty_labels=questions_answer_string,
        )
        response = self.LLM_CLIENT.ask(prompt=prompt)
        print(
            Fore.BLUE
            + f"[INFO: LLM] Refine image description using the on-board VLM, the question/answer pairs and the uncertainty associated to them: \n{response}"
        )
        print("---" * 5)
        try:
            yaml_data = response.split("YAML_START")[1].split("YAML_END")[0].strip()
            parsed_yaml = yaml.safe_load(yaml_data)
            update_detected_obj_description = parsed_yaml["image_description_refined"]

            attributes = parsed_yaml["attributes_of_the_image"]

            # Print each attribute:value pair
            image_attributes = {}
            for attribute, value in attributes.items():
                image_attributes[attribute] = value

            return update_detected_obj_description, image_attributes
        except Exception as e:
            print(e)
            logging.exception(response)
            logging.info("This was the prompt")
            logging.exception(prompt)
            raise e

    # max retry 10 times, wait 10 seconds between each retry
    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def get_similarity_score_and_question_for_target_object(self, target_object: str, detected_object_description: str):
        """ """
        prompt = prompts.LLM_SIMILARITY_SCORE_AND_QUESTION_TO_TARGET.format(
            target_object=target_object,
            distractor_object_description=detected_object_description,
            facts_about_the_target_picture=self.target_object_informations or "No information available",
        )
        response = self.LLM_CLIENT.ask(prompt=prompt)
        print(Fore.BLUE + f"[INFO: LLM] Get similarity score and question for the user (if necessary) \n {response}")
        print("---" * 5)
        try:
            yaml_data = response.split("YAML_START")[1].split("YAML_END")[0].strip()
            parsed_yaml = yaml.safe_load(yaml_data)

            similarity_score = parsed_yaml["similarity_score"]
            questions_for_human = [q.strip() for q in parsed_yaml["questions"].values()]

            return similarity_score, questions_for_human
        except Exception as e:
            print(e)
            logging.exception(response)
            logging.info("This was the prompt")
            logging.exception(prompt)
            raise e

    # max retry 10 times, wait 10 seconds between each retry
    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def updates_known_facts_about_target_object_given_oracle_answers(
        self, target_object: str, oracle_questions_answers: List[dict]
    ):
        questions_answer_string = ""
        for item in oracle_questions_answers:
            question, answer = item["question"], item["answer"]
            questions_answer_string += f"- Question: {question} - Answer: {answer}\n"

        prompt = prompts.LLM_FACTS_UPDATER_AFTER_IS_THIS_TARGET_OBJECT_ORACLE_QUESTION_V1.format(
            target_object=target_object,
            oracle_questions_answer=questions_answer_string,
            facts_about_the_target_picture=self.target_object_informations,
        )

        response = self.LLM_CLIENT.ask(prompt=prompt)
        print(Fore.BLUE + f"[INFO: LLM] Updating know facts using answers from the user. \n {response}")
        try:
            yaml_data = response.split("YAML_START")[1].split("YAML_END")[0].strip()
            parsed_yaml = yaml.safe_load(yaml_data)

            new_facts = parsed_yaml["facts"]
            self.target_object_informations = new_facts
            return new_facts
        except Exception as e:
            print(e)
            logging.exception(response)
            logging.info("This was the prompt")
            logging.exception(prompt)
            raise e

    def reset(self):
        self.ep_id = None
        self.questions_and_answers_database: List[dict] = (
            []
        )  # contains a list of question and answer. This provides more context to the LLM to reason about the score
        self.target_object_informations = ""  # this variable maintains the information about the target object, mimicking what's happening in real-life situation

        # the following variable maintains a graph about object found -> scores associated to it and their
        self.objects_graph_informations: Dict[str, Dict[str, Any]] = {}