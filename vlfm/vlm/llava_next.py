from typing import Any, Optional
from flask import jsonify
import numpy as np
import os
import torch
from PIL import Image
import random
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
from colorama import Fore
from colorama import init as init_colorama

init_colorama(autoreset=True)


# seed torch
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
        max_new_tokens=500,
        device: Optional[Any] = None,
    ) -> None:
        seed_everything(42)
        self.max_new_tokens = max_new_tokens
        self.processor = LlavaNextProcessor.from_pretrained(model_type)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_type,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={
                "image_newline": 0,
                "multi_modal_projector": 0,
                "vision_tower": 0,  # Place the vision tower on GPU 0
                "language_model": 1,  # Place the language model on GPU 1
            },
            # quantization_config=quantization_config,
        )

        print(Fore.GREEN + f"Model loaded! with max new tokens {self.max_new_tokens}")

    def ask(
        self, image: np.ndarray, prompt: Optional[str] = None, return_token_likelihood: Optional[bool] = False
    ) -> str:
        """Generates a caption for the given image.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str, optional): An optional prompt to provide context and guide
                the caption generation. Can be used to ask questions about the image.

        Returns:
            dict: The generated caption.

        """
        if return_token_likelihood:
            # the prompt must contains ' You must answer only with Yes, No, or ?=I don't know.'
            if "You must answer only with Yes, No, or ?=I don't know" not in prompt:
                prompt += " You must answer only with Yes, No, or ?=I don't know."

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

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        if not return_token_likelihood:
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

            output = self.processor.decode(output[0], skip_special_tokens=True)
            output = output.split("[/INST]")[-1].strip()
            return {"lmm_output": output}
        else:
            # we want to get also the likelihood of the tokens (useful when self-questioning)
            # in greedy deconding, logits should be equal to the scores
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )
            output_decoded = self.processor.decode(output["sequences"][0], skip_special_tokens=True)
            output_string = output_decoded.split("[/INST]")[-1].strip()

            # get the actual likelihood of the tokens
            # answer should only contains one word  -> 'Yes', 'No', or '?'

            logits = output["logits"]
            print(Fore.YELLOW + prompt)
            print(Fore.GREEN + output_string)

            for timestep_t in range(len(logits)):
                logits_time_t = logits[timestep_t][0]  # -> shape num_vocab
                logits_time_t = torch.softmax(logits_time_t, dim=-1)

                # get tokens likelihood
                top_k = 3
                topk_scores, topk_indices = torch.topk(logits_time_t, top_k)
                topk_indices = topk_indices.cpu().numpy()

                values_to_be_returned = []
                for i in range(top_k):
                    decoded: str = self.processor.decode([topk_indices[i]], skip_special_tokens=True)
                    likelihood = round(logits_time_t[topk_indices[i]].cpu().item(), 3)

                    values_to_be_returned.append((decoded, likelihood))

                break
            return {"lmm_output": output_string, "likelihood": values_to_be_returned}


class LLavaNextClient:
    def __init__(self, port: int = 12189):
        self.url = f"http://localhost:{port}/llava_next"

    def ask(self, image: np.ndarray, prompt: Optional[str] = None, return_token_likelihood=False) -> str:

        if prompt is None:
            prompt = "Describe the image in detail, also with details from its surroundings."
        with torch.no_grad():
            response = send_request(
                self.url,
                image=image,
                prompt=prompt,
                return_token_likelihood=return_token_likelihood,
                request_timeout=15,
            )
        torch.cuda.empty_cache()
        if return_token_likelihood:

            return response["response"]["lmm_output"], response["response"]["likelihood"]
        else:
            return response["response"]["lmm_output"], None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8070)
    args = parser.parse_args()
    seed_everything(42)
    print("Loading model...")

    class LLavaNextServer(ServerMixin, LLavaNext):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_busy = False

        def process_payload(self, payload: dict) -> dict:
            if self.is_busy:
                return jsonify({"error": "Server is busy"}), 503  # Return HTTP 503 (Service Unavailable)

            self.is_busy = True
            try:
                image = str_to_image(payload["image"])
                response = {
                    "response": self.ask(
                        image,
                        prompt=payload.get("prompt"),
                        return_token_likelihood=payload.get("return_token_likelihood", False),
                    )
                }
            finally:
                self.is_busy = False
            return response

    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    llava = LLavaNextServer(model_type=model_name)
    print(f"Model - {model_name} loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(llava, name="llava_next", port=args.port)
