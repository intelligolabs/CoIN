from PIL import Image
from vlfm.utils.prompts import (
    LLaVa_TARGET_OBJECT_IS_DETECTED,
    LLM_IS_THIS_THE_TARGET_IMAGE_HUMAN_FEEDBACK_ORACLE_V1,
)
from typing import List
import logging
from colorama import Fore
from colorama import init as _init_co
from retrying import retry

_init_co(autoreset=True)
import yaml
import numpy as np


class VLMOracle:

    def __init__(self, llava_client, llm_client) -> None:
        self.LMM_CLIENT = llava_client
        self.LLM_CLIENT = llm_client
        self.instance_image = None
        self.ep_id = None
        self.instance_image_description = None  # description of the target image
        self.ask_to_human_episode_counter = {}

        # this ensure that no LMM is used to answer question, and that LLM does not perform answer checking
        # useful to use it for API call
        self.HUMAN_HAS_TO_ANSWER_THE_QUESTION = False

        # this value is set by the main process, only available to the oracle
        self.is_this_the_target_object = False

    def get_instance_image(self):
        return self.instance_image

    def set_instance_image(self, instance_image, target_object):
        self.instance_image = instance_image
        self.set_image_description(instance_image, target_object)

    # max retry 10 times, wait 10 seconds between each retry
    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def set_image_description(self, image_description, target_object):
        prompt = LLaVa_TARGET_OBJECT_IS_DETECTED.format(target_object=target_object)

        image_description, _ = self.LMM_CLIENT.ask(np.array(image_description), prompt=prompt)

        self.instance_image_description = image_description
        print(Fore.LIGHTCYAN_EX + f"Image description: {image_description}")
        # Image.fromarray(self.instance_image).save("instance_image.png")

    def get_description_of_the_image(self):
        return self.instance_image_description

    def reset(self):
        self.instance_image = None
        self.ep_id = None
        self.instance_image_description = None

    def how_many_question_to_the_user(self, ep_id):
        return self.ask_to_human_episode_counter.get(ep_id, 0)

    # max retry 10 times, wait 10 seconds between each retry
    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def answer_question_given_image(
        self,
        questions: List[str],
        ARE_QUESTIONS_FOR_THE_ORACLE=True,
        USE_LLM_TO_CHECK_THE_ANSWER=True,
        image_to_be_used=None,
        perform_logits_likelihood=False,
        ep_id=None,
    ) -> List[dict]:
        """
        This module allows questions to be answered given an image
        ARE_QUESTIONS_TO_THE_HUMAN:
            True: If the question are given to the human, the oracle will answer considering the target image
            False: the agent needs more information from the current image, thus it can used the VQA system as well without bothering the human (self-questions), Image param must be set in this case
        USE_LLM_TO_CHECK_THE_ANSWER: If True, the oracle will use the LLM to check the answer. If False, the oracle will provide the answer directly.
        """
        assert len(questions) > 0, "Questions must be provided"
        if ARE_QUESTIONS_FOR_THE_ORACLE:
            assert image_to_be_used is None, "if question are for the human, image_to_be_used must not be provided"
        else:
            # the model can ask information about the current detection, which are answered by the VQA model wihtout any humab oracle intervention ofc.
            assert image_to_be_used is not None, "if question are not for the human, image_to_be_used must be provided"
            assert not USE_LLM_TO_CHECK_THE_ANSWER, "LLM should not be used as checker for VQA response."

        IMAGE_TO_BE_USED_BY_VQA_MODEL = self.instance_image if ARE_QUESTIONS_FOR_THE_ORACLE else image_to_be_used
        assert IMAGE_TO_BE_USED_BY_VQA_MODEL is not None, "Image must be provided"

        if not ARE_QUESTIONS_FOR_THE_ORACLE:
            print(Fore.YELLOW + "[INFO: On-board VLM] Answerinq questions using the on-board VLM model.")
        else:
            print(
                Fore.BLUE
                + "[INFO: VLM_simulated user] Answering questions using the VLM simulated user and the high-def image."
            )
        response_array = []
        for question in questions:
            prompt = question
            if self.HUMAN_HAS_TO_ANSWER_THE_QUESTION and ARE_QUESTIONS_FOR_THE_ORACLE:
                output = input(Fore.LIGHTRED_EX + f"Question: {question}\n{Fore.GREEN}Answer: ")
                logits_likelihood = None
            else:
                output, logits_likelihood = self.LMM_CLIENT.ask(
                    np.array(IMAGE_TO_BE_USED_BY_VQA_MODEL),
                    prompt=prompt,
                    return_token_likelihood=perform_logits_likelihood,
                )
                if perform_logits_likelihood:
                    assert (
                        logits_likelihood is not None
                    ), "logits_likelihood must be provided if perform_logits_likelihood is True"
            print(f"\t -> Question: {question}")
            print(f"\t -> Answer: {output}")
            print("\n")
            response = {"question": question, "answer": output, "logits_likelihood": logits_likelihood}
            response_array.append(response)

        try:
            if ep_id is not None:  # thus in self-questioner mode
                if ep_id not in self.ask_to_human_episode_counter:
                    self.ask_to_human_episode_counter[ep_id] = 0
                self.ask_to_human_episode_counter[ep_id] += len(response_array)
                print(
                    Fore.LIGHTCYAN_EX
                    + f"Total number of questions asked to the human: {self.ask_to_human_episode_counter[ep_id]}"
                )
        except:
            pass

        # if we don't want to validate the answer, we return the response array
        if not USE_LLM_TO_CHECK_THE_ANSWER or self.HUMAN_HAS_TO_ANSWER_THE_QUESTION:
            return response_array

        raise NotImplementedError("The LLM answer checking is not implemented yet.")

    def reset(self):

        self.instance_image = None
        self.ep_id = None
        self.instance_image_description = None  # description of the target image

        # this value is set by the main process, only available to the oracle
        self.is_this_the_target_object = False
