from typing import Any, Optional
import numpy as np
import os
import torch
from PIL import Image
import random

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from colorama import Fore, Style
from colorama import init as init_colorama

init_colorama(autoreset=True)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class LLavaNext:
    def __init__(
        self,
        model_type: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        max_new_tokens=2000,
        device: Optional[Any] = None,
        load_in_4_bit=False,
    ) -> None:
        seed_everything(42)
        self.max_new_tokens = max_new_tokens
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.processor = LlavaNextProcessor.from_pretrained(model_type)
        self.model_name = model_type
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_type,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_4bit=load_in_4_bit,
        )

    def custom_decode(self, output, inputs=None):
        """certain llava version returns out in a different format"""
        if self.model_name in [
            "llava-hf/llama3-llava-next-8b-hf",
            "llava-hf/llava-v1.6-vicuna-13b-hf",
            "llava-hf/llava-v1.6-34b-hf",
        ]:
            if inputs is None:
                return self.processor.decode(output[0], skip_special_tokens=True)
            else:
                return self.processor.decode(
                    output[0][inputs["input_ids"].size(1) :], skip_special_tokens=True
                ).strip()
        else:
            return self.processor.decode(output[0], skip_special_tokens=True)

    def ask(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
        return_logits: Optional[bool] = False,
        scores_for_linear_probe: Optional[bool] = False,
    ) -> str:
        """Generates a caption for the given image.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str, optional): An optional prompt to provide context and guide
                the caption generation. Can be used to ask questions about the image.

        Returns:
            dict: The generated caption.

        """
        if return_logits:
            # the prompt must contains ' You must answer only with Yes, No, or ?=I don't know.'
            if "You must answer only with Yes, No, or ?=I don't know" not in prompt:
                prompt += " You must answer only with Yes, No, or ?=I don't know."
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        print(Fore.YELLOW + f"[INFO] Asking the model: {prompt}")
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        if not return_logits:
            if not scores_for_linear_probe:
                output = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
                output = self.custom_decode(output, inputs)
                output = output.split("[/INST]")[-1].strip()

                return {"lmm_output": output}
            else:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=scores_for_linear_probe,
                    output_scores=scores_for_linear_probe,
                )
                output_str = self.custom_decode(output["sequences"], inputs)
                output_str = output_str.split("[/INST]")[-1].strip()
                return {"lmm_output": output_str, "scores_for_LP": output["scores"]}

        else:
            # we want to get also the likelihood of the tokens (useful when self.questioning)
            # in greedy deconding, logits should be equal to the scores
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            output_decoded = self.custom_decode(output["sequences"], inputs)
            output_string = output_decoded.split("[/INST]")[-1].strip()

            # get the actual likelihood of the tokens
            # answer should only contains one word  -> 'Yes', 'No', or '?'

            # TODO switch to scores? for gredy decoding, logits should be equal to the scores
            logits = output["scores"]

            for timestep_t in range(len(logits)):
                logits_time_t = logits[timestep_t][0]  # -> shape num_vocab
                logits_time_t = torch.softmax(logits_time_t, dim=-1)

                # get tokens likelihood
                top_k = 3
                topk_scores, topk_indices = torch.topk(logits_time_t, top_k)
                topk_indices = topk_indices.cpu().numpy()

                token_probs_to_be_returned = [] # here we have probs
                token_scores_to_be_returned = [] # hjere we have scores
                for i in range(top_k):
                    decoded: str = self.custom_decode([topk_indices[i]])
                    if decoded.strip() in ["", ":", "`"]:
                        continue
                    likelihood = round(logits_time_t[topk_indices[i]].cpu().item(), 3)
                    # print(Fore.RED + "\t\t ->" + decoded)
                    # print(f"\t The likelihood of the token '{decoded}'[{topk_indices[i]}] is: ", likelihood)
                    # assert decoded.lower() in ["yes", "no", "?"], f"Max tok. lilekihood not found in {decoded}" # too strict

                    token_probs_to_be_returned.append((decoded, likelihood))
                    token_scores_to_be_returned.append((decoded, output['scores'][timestep_t][0][topk_indices[i]].cpu().item()))
                    # assert decoded in ["Yes", "No", "?"], f"Max tok. lilekihood not found in {decoded}"

                if len(token_probs_to_be_returned) > 0:
                    break
            # print(Fore.RED + f"Used output at time step {timestep_t}")
            # TODO this should be changed - rightn now don't have time to change the occurence in the cde
            # logits is probs (we perform softmax),
            # raw_logits instead is the scores before softmax
            return {"lmm_output": output_string, "logits": token_probs_to_be_returned, "raw_logits":token_scores_to_be_returned}


if __name__ == "__main__":
    seed_everything(42)
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    # model_name = "llava-hf/llama3-llava-next-8b-hf"
    llava_model = LLavaNext(
        model_type=model_name, load_in_4_bit=True, max_new_tokens=100
    )

    rand_image = torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8).numpy()

    # print(llava_model.ask(rand_image, prompt="spell this string: hello world"))
    # print(
    #     llava_model.ask(
    #         rand_image, prompt="spell letter by letter the following: engono"
    #     )
    # )
    # out = llava_model.ask(
    #     rand_image,
    #     prompt="What is the color of the sky in this image?",
    #     return_logits=False,
    # )
    # print(Fore.GREEN + out["lmm_output"])
    out = llava_model.ask(
        rand_image,
        prompt="sking the model: Is the chair in the room a recliner? You must answer only with Yes, No or ?=I don't know.",
        return_logits=True,
    )
    print(Fore.GREEN + out["lmm_output"])
    print(out["logits"])
