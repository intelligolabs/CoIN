import os, json
from vqa_evaluator import VQAEvaluator
from tqdm import tqdm
from colorama import Fore
from colorama import init as init_colorama
import numpy as np

init_colorama(autoreset=True)
import torch
import argparse
import random


def filter_by_uncertainty(logits, answer, tau=0.7):
    posterior_prob = torch.tensor([item[1] for item in logits])
    # assert len(posterior_prob) == 3, f"Expected 3 logits, got {len(posterior_prob)}, {logits} with answer {answer}"
    if len(posterior_prob) != 3:
        raise Exception(
            f"Expected 3 logits, got {len(posterior_prob)}, {logits} with answer {answer}"
        )
    entropy = -torch.sum(posterior_prob * torch.log(posterior_prob))
    entropy_max = torch.log(torch.tensor(len(posterior_prob)))
    entropy_normalized = entropy / entropy_max

    # with this formulation, if entropy norm is less that tau we are certain, otherwise we are uncertain
    if entropy_normalized > tau:
        return "I don't know"
    else:
        return answer


if __name__ == "__main__":
    # run wiht PYTHONPATH="." python dataset_creator_vqa/llava_vqa_eval.py  --model="llava-v1.6-mistral-7b-hf"
    # get model to test
    parser = argparse.ArgumentParser(description="VQA Evaluator")
    parser.add_argument(
        "--model",
        type=str,
        help="Model to test",
        default="llava-v1.6-mistral-7b-hf",
        choices=[
            "llava-v1.6-mistral-7b-hf",
            "llama3-llava-next-8b-hf",
            "llava-v1.6-vicuna-13b-hf",
            "llava-v1.6-34b-hf",
        ],
        required=False,
    )
    args = parser.parse_args()
    model = args.model

    LLAVA_FILE_ROOT = f"models_results/{model}.json"
    VQA_DATASET_ROOT_PATH = "idkvqa_gt.json"
    with open(VQA_DATASET_ROOT_PATH, "r") as f:
        vqa_dataset_root = json.load(f)

    with open(LLAVA_FILE_ROOT, "r") as f:
        llava_results_file = json.load(f)

    vqa_evaluator = VQAEvaluator(vqa_dataset_root, cost=1)

    ################
    ### Analyze the stats from standard LLaVA Model
    ################
    model_question_answer_pairs_test = dict(original_llava=[])
    for image_filename, question in tqdm(
        list(vqa_evaluator.questions_iterator()), desc="Processing images"
    ):
        # get the answer from the llava model
        found = False
        for item in llava_results_file["images"][image_filename][
            "questions_answers_pairs"
        ]:
            if item["question"] == question:
                model_answer = item["answer"]
                logits = item["logits"]
                found = True
                break
        assert found, f"Question not found in the llava results file: {question}"
        # origina llava
        model_question_answer_pairs_test["original_llava"].append((
            image_filename,
            question,
            model_answer,
        ))
        # print(image_filename)

        # entropy based filtering

        for tau in np.arange(0, 1, 0.01):
            try:
                filtered_answer = filter_by_uncertainty(logits, model_answer, tau=tau)
                if f"entropy_llava_tau_{tau}" not in model_question_answer_pairs_test:
                    model_question_answer_pairs_test[f"entropy_llava_tau_{tau}"] = []
                model_question_answer_pairs_test[f"entropy_llava_tau_{tau}"].append((
                    image_filename,
                    question,
                    filtered_answer,
                ))
            except Exception as e:
                # TODO what is the best way to handle this? The model is not followuing the prompt
                print(Fore.RED + "[ERRRO] Error in filtering by uncertainty")
                model_question_answer_pairs_test[f"entropy_llava_tau_{tau}"].append((
                    image_filename,
                    question,
                    random.choice(list(vqa_evaluator.reverse_choices_labels.keys())),
                ))

    best_tau = 0
    best_er = 0
    for test, data in model_question_answer_pairs_test.items():
        er = vqa_evaluator.model_get_effective_reliability(data)
        if test == "original_llava":
            print(Fore.YELLOW + "[INFO] Original LLava Model has an er of ", er)
            continue

        if er > best_er:
            best_er = er
            best_tau = float(test.split("_")[-1])
    print(Fore.YELLOW + f"[INFO] Model tested: {model}")

    print(
        Fore.GREEN
        + f"Best Model effective reliability found with tau[{best_tau}]: er -> {best_er}"
    )
