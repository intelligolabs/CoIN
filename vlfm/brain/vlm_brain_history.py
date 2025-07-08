from PIL import Image
from colorama import Fore
from colorama import init as init_colorama

init_colorama(autoreset=True)
import numpy as np
from vlfm.utils.prompts import LLava_REDUCE_FALSE_POSITIVE


class Conversation:
    def __init__(self) -> None:
        self.conversation = []

    def add_message(self, role, text, image):
        content = {"type": "image", "image": image, "text": text}
        self.conversation.append({"role": role, "content": [content]})

    def get_convrersation(self):
        return self.conversation

    def reset(self):
        self.conversation = []


class VLM_History:
    """
    This class store the connector for the vlm (LLaVa in this case) and it's past conversation.
    """

    def __init__(self, llava_client) -> None:
        self.llava_client = llava_client
        self.ep_id = None
        self.conversation = Conversation()

    def get_description_of_the_image(self, image, prompt):
        output, _ = self.llava_client.ask(image, prompt=prompt)
        print(Fore.LIGHTMAGENTA_EX + "[INFO: On-board VLM] " + output)
        return output

    def reduce_detector_false_positive(
        self, detected_image: np.ndarray, target_object: str, get_logits: bool = False
    ) -> str:
        prompt = LLava_REDUCE_FALSE_POSITIVE.format(target_object=target_object)
        response = self.llava_client.ask(detected_image, prompt=prompt, return_token_likelihood=get_logits)
        return response

    def reset(self):
        self.ep_id = None
        self.conversation.reset()