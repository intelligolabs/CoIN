import os, json
from vqa_evaluator import VQAEvaluator
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import hashlib
from llava_next_wrapper import LLavaNext
from colorama import Fore
from colorama import init as init_colorama
from tqdm import tqdm

init_colorama(autoreset=True)


def _generate_local_data(images_path, gt_json_path):
    if os.path.exists(images_path) or os.path.exists(gt_json_path):
        print(f"[WARN] Image path '{images_path}' or groundtruth path '{gt_json_path}' already exist. Removing them...")

    # Create local image folder
    os.makedirs(images_path)

    repo = "ftaioli/IDKVQA"
    dataset = load_dataset(repo)
    # Merge the two splits (in the paper we use the full dataset)
    splits = list(dataset.column_names.keys())

    # Annotation file
    annotations = dict(choice_label={"0": "Yes", "1": "No", "2": "I don't know"}, images=dict())

    for split in splits:
        os.makedirs(os.path.join(images_path, split))
        for row in tqdm(dataset[split], desc="Generating images..."):
            image = row["image"]
            sha1 = hashlib.sha1(image.tobytes()).digest().hex()
            name_file = f"{split}_{sha1}.png"
            image_path = os.path.join(images_path, split, name_file)

            # Save the image locally with name schema {split}_{sha}.png inside {images_path}/{split}
            image.save(image_path)

            if name_file not in annotations["images"].keys():
                annotations["images"][name_file] = dict(result="accepted", questions_answers_pairs=[])

            counts = row["answers"]
            counts = dict({(v[0], str(v[1])) for v in counts.items()})

            answers = []
            for c in counts.values():
                answers.append(str(c))
            num_annotators = len(answers)
            annotations["images"][name_file]["questions_answers_pairs"].append(
                dict(
                    question=row["question"],
                    counts=counts,
                    answers=answers,
                    annotators_number=num_annotators,
                )
            )
        with open(gt_json_path, "w") as write_file:
            json.dump(annotations, write_file)


if __name__ == "__main__":
    # run with
    # CUDA_VISIBLE_DEVICES="0,1,2,3"  PYTHONPATH="." python dataset_creator_vqa/llava_vqa_get_answer.py
    """
    Idea: we load the dataset, we got the accepted images and we run llava to get responses and logits.
    The results is saved to a json file, so it easier and faster to load the results and do the analysis.
    """
    MODEL_HF = "llava-hf/llava-v1.6-mistral-7b-hf"  # not all llava models have the ouput in the same format, for example, llama

    llava_result_file_json_path = MODEL_HF.split("/")[-1] + ".json"

    IMAGES_DS_ROOT = "images_idkvqa/"
    VQA_DATASET_ROOT_PATH = "idkvqa_gt.json"

    _generate_local_data(IMAGES_DS_ROOT, VQA_DATASET_ROOT_PATH)

    LLAVA_RESULTS_PATH = "models_results/"

    if not os.path.exists(LLAVA_RESULTS_PATH):
        os.makedirs(LLAVA_RESULTS_PATH)

    if not os.path.exists(os.path.join(LLAVA_RESULTS_PATH, llava_result_file_json_path)):
        with open(os.path.join(LLAVA_RESULTS_PATH, llava_result_file_json_path), "w") as f:
            json.dump({"images": {}}, f)
    else:
        print(
            Fore.RED + f"[ERROR] File already exists, not supported yet. Removing the existing one and recreating it."
        )
        os.remove(os.path.join(LLAVA_RESULTS_PATH, llava_result_file_json_path))
        with open(os.path.join(LLAVA_RESULTS_PATH, llava_result_file_json_path), "w") as f:
            json.dump({"images": {}}, f)

        # raise Exception("File already exists, not supported yet.")

    with open(os.path.join(LLAVA_RESULTS_PATH, llava_result_file_json_path), "r") as f:
        llava_results_file = json.load(f)

    assert os.path.exists(VQA_DATASET_ROOT_PATH), f"Dataset not found at {VQA_DATASET_ROOT_PATH}"

    # Load the dataset
    with open(VQA_DATASET_ROOT_PATH, "r") as f:
        dataset = json.load(f)

    llava_model = LLavaNext(model_type=MODEL_HF)
    print(Fore.YELLOW + f"[INFO] Model Loaded - {MODEL_HF}")

    vqa_evaluator = VQAEvaluator(dataset, n=3)
    for image_filename, question in tqdm(list(vqa_evaluator.questions_iterator()), desc="Processing images"):
        # load image
        scene = image_filename.split("_")[0]
        image = Image.open(os.path.join(IMAGES_DS_ROOT, scene, image_filename))

        output = llava_model.ask(image, question, return_logits=True)
        answer, logits = output["lmm_output"], output["logits"]

        if image_filename not in llava_results_file["images"]:
            llava_results_file["images"][image_filename] = dict(questions_answers_pairs=[])
        llava_results_file["images"][image_filename]["questions_answers_pairs"].append(
            dict(question=question, answer=answer, logits=logits)
        )

        # save the data to disk
        with open(os.path.join(LLAVA_RESULTS_PATH, llava_result_file_json_path), "w") as f:
            json.dump(llava_results_file, f, indent=4)
        # print(llava_results_file['images'][image_filename]['questions_answers_pairs'])

        # input tp continue
        # input("Press Enter to continue...")
