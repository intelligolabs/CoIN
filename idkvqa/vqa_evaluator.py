from typing import List, Dict
import os
import json
import random
from tqdm import tqdm
import numpy as np
from colorama import Fore
from colorama import init as init_colorama

init_colorama(autoreset=True)


class VQAEvaluator:
    def __init__(
        self, ground_truth_data: List[Dict], cost: int = 1, n: int = 2, log=True
    ):
        """
        ds ground truth of the released vqa dataset
        cost: cost associated to a wrong answer. See paper for more details
        n: number of decimal places to round the result
        """
        self.ground_truth_data = ground_truth_data

        self.choices_labels: Dict[str, str] = (
            None  # mapping str -> reason - 0 - yes, 1 - no, 2 - i don't know
        )
        self.reverse_choices_labels: Dict[str, str] = (
            None  # mapping reason -> str - yes - 0, no - 1, i don't know - 2
        )
        self.cost = cost
        self.n = n
        # Denominator in the accuracy computation ("# of identical answers"/denominator_accuracy)
        # Usually, it's 3
        self.denominator_accuracy = 3
        self.log = log
        self.process_data()

    def process_data(self):
        self.choices_labels = self.ground_truth_data["choice_label"]
        self.choices_labels = {
            key: value
            for key, value in self.choices_labels.items()
            if key in {"0", "1", "2"}
        }  # yes, no, i don't know
        self.reverse_choices_labels = {
            v.lower(): k for k, v in self.choices_labels.items()
        }

        self.ground_truth_data = [
            dict(image=k, data=v)
            for k, v in self.ground_truth_data["images"].items()
            if v["result"] == "accepted"
        ]
        for item in self.ground_truth_data:
            data = item["data"]
            assert "questions_answers_pairs" in data, (
                "Questions and answers pairs not found in the dataset"
            )
            assert data["questions_answers_pairs"], (
                "Questions and answers pairs are empty"
            )
            assert len(data["questions_answers_pairs"]) > 0, (
                "Questions and answers pairs are empty"
            )
            for qa_pair in data["questions_answers_pairs"]:
                assert "question" in qa_pair, "Question not found in the dataset"
                assert "answers" in qa_pair, "Answers not found in the dataset"
                assert "counts" in qa_pair, "Counts not found in the dataset"
                assert qa_pair["question"], "Question is empty"

    def print_info(self):
        print(
            Fore.YELLOW
            + f"[INFO] Ground truth data loaded with {len(self.ground_truth_data)} different images"
        )
        print(Fore.YELLOW + f"[INFO] Choices labels: {self.choices_labels}")

        tot_question = len(list(self._questions_answers_iterator()))
        print(Fore.YELLOW + f"[INFO] Total question annotated: {tot_question}")

    def VQA_accuracy(self, answer_key, gt_counts):
        count = gt_counts[answer_key]
        count_check = sum(int(count) for count in gt_counts.values())
        assert count_check == self.denominator_accuracy, (
            Fore.RED
            + f"We have less annotation for this question, found only {count_check} instead of {self.denominator_accuracy}"
        )
        return min(float(count) / self.denominator_accuracy, 1)

    def effective_reliability(
        self, g_of_x: bool, gt_answers_key: List, gt_counts: dict, model_answer_key: str
    ):
        """
        Compute the effective reliability of a model answer given the ground truth answer
        g_of_x: True if the model want to answer, False otherwise
        params: gt_answers_key: answers key by the annotators
        params: gt_counts: count for each asnwer key, in dict form {"key": "# of ann. that answered 'key'"}
        params: model_answer_key: model answer key
        """
        for ans in gt_answers_key:
            assert ans in list(self.choices_labels.keys()), (
                f"Invalid ground truth answers {ans}: one is not in {self.choices_labels.keys()}"
            )

        assert model_answer_key in self.choices_labels.keys(), (
            f"Invalid model answer '{model_answer_key}': not in {self.choices_labels}"
        )

        if g_of_x:
            accuracy = self.VQA_accuracy(model_answer_key, gt_counts)
            if accuracy > 0:
                return accuracy
            else:
                return -self.cost
        else:
            return 0

    def questions_iterator(self):
        for item in self._questions_answers_iterator():
            yield item[0], item[1]  # we don't need the answer

    def _answer_to_key(self, answer):
        answer = answer.lower()
        try:
            return self.reverse_choices_labels[
                answer
            ]  # yes -> 0, no -> 1, i don't know -> 2
        except:
            # 1. the answer could be 'No, I think ..., thus check if no or yes are present
            if answer.startswith("no"):
                return self.reverse_choices_labels["no"]
            elif answer.startswith("yes"):
                return self.reverse_choices_labels["yes"]
            elif np.any([
                answer.startswith(i)
                for i in [
                    "? i don't know",
                    "i don't know",
                    "?i don't know",
                    "?=i don't know",
                ]
            ]):
                return self.reverse_choices_labels["i don't know"]
            raise ValueError(f"Invalid answer: {answer}")

    def _key_to_answer(self, key):
        return self.choices_labels[key]  # 0 -> yes, 1 -> no, 2 -> i don't know

    def model_get_effective_reliability(self, model_question_answer_pairs) -> float:
        """
        Returns the effective reliability of a model given the question and answer pairs
        params: model_question_answer_pairs: list of tuples (image_filename, question, model_answer)
        """
        assert len(model_question_answer_pairs) == len(
            list(self._questions_answers_iterator())
        ), (
            Fore.RED
            + f"Model answers are not the same as the ground truth answers. Found {len(model_question_answer_pairs)} instead of {len(list(self._questions_answers_iterator()))}"
        )
        effective_reliability = []
        for item in tqdm(
            model_question_answer_pairs,
            desc="Computing effective reliability",
            disable=not self.log,
        ):
            image_filename, question, model_answer = item

            try:
                gt_answer_keys, gt_counts = self._get_ground_truth_answer(
                    image_filename, question
                )

                model_answer_key = self._answer_to_key(model_answer)
                assert model_answer_key in ["0", "1", "2"], "Answer must be 0, 1, or 2"

                effective_reliability.append(
                    self.effective_reliability(
                        g_of_x=model_answer_key
                        in ["0", "1"],  # the model want to answer
                        gt_answers_key=gt_answer_keys,
                        gt_counts=gt_counts,
                        model_answer_key=model_answer_key,
                    )
                )
            except:
                print(
                    f"Image: {image_filename}, Question: {question}, Model answer: {model_answer}"
                )
                print("Skipping this question")
                raise ValueError("Error")
        effective_reliability = sum(effective_reliability) / len(effective_reliability)
        return round(100 * effective_reliability, self.n)

    def _get_ground_truth_answer(self, image_filename, question):
        for (
            image,
            gt_question,
            gt_answers,
            gt_counts,
        ) in self._questions_answers_iterator():
            if not image == image_filename:
                continue
            if not gt_question == question:
                continue

            return gt_answers, gt_counts

        raise ValueError(
            f"Image '{image_filename}' with question {question}not found in the dataset"
        )

    def _questions_answers_iterator(self):
        """
        provide an iterator for quesiton and answer. It skipped question that are not labelled correcly
        """
        for item in self.ground_truth_data:
            data = item["data"]
            image = item["image"]
            questions_answer_pairs = data["questions_answers_pairs"]

            for item in questions_answer_pairs:
                qt_question, qt_answer, qt_counts = (
                    item["question"],
                    item["answers"],
                    item["counts"],
                )
                # Check wheter all the answers are correct labels
                if np.any([ans not in self.choices_labels for ans in qt_answer]):
                    # print("skipping question", qt_question, "because it is not labelled correctly", qt_answer)
                    continue

                yield image, qt_question, qt_answer, qt_counts

    def test_random_model_q_of_x_should_be_accuracy(self):
        """
        a property of the effective reliability function is the following:
            effe. reliab. is equal to the accuracy of the model when the model abstains from answering when the answer is not correct
        TODO PROVE THE FOLLOWING: this does not hold if one of the answers is "I don't know"
        """
        effective_reliability = []
        accuracies = []
        for _, _, gt_answers_key, gt_counts in self._questions_answers_iterator():
            # TODO set not to '2' due to the above
            answer_key = random.choice(["0", "1"])

            accuracy = self.VQA_accuracy(answer_key, gt_counts)
            accuracies.append(accuracy)

            # If it's not correct, it abstains
            wrong = accuracy == 0.0
            if wrong:
                # Set it to abstain
                answer_key = "2"

            # wants to answer if key is not "i don't know"
            g_of_x = answer_key in ["0", "1"]
            effective_reliability.append(
                self.effective_reliability(
                    g_of_x=g_of_x,
                    gt_answers_key=gt_answers_key,
                    gt_counts=gt_counts,
                    model_answer_key=answer_key,
                )
            )
        return round(
            100 * sum(effective_reliability) / len(effective_reliability), self.n
        ), round(100 * sum(accuracies) / len(effective_reliability), self.n)

    def evaluate_random_model(self):
        effective_reliability = []
        for _, _, gt_answers, gt_scores in self._questions_answers_iterator():
            effective_reliability.append(
                self.effective_reliability(
                    random.choice([False, True]),
                    gt_answers,
                    gt_scores,
                    random.choice(list(self.choices_labels.keys())),
                )
            )
        effective_reliability = sum(effective_reliability) / len(effective_reliability)
        return round(100 * effective_reliability, self.n)


if __name__ == "__main__":
    """run whit:
    PYTHONPATH="." python dataset_creator_vqa/vqa_evaluator.py
    """
    random.seed(42)
    VQA_DATASET_ROOT_PATH = "final_annotations.json"
    assert os.path.exists(VQA_DATASET_ROOT_PATH), (
        f"Dataset not found at {VQA_DATASET_ROOT_PATH}"
    )

    # Load the dataset
    with open(VQA_DATASET_ROOT_PATH, "r") as f:
        dataset = json.load(f)

    vqa_evaluator = VQAEvaluator(dataset)
    vqa_evaluator.print_info()
    print("Effective accuracy random model", vqa_evaluator.evaluate_random_model())

    effective_acc, acc = vqa_evaluator.test_random_model_q_of_x_should_be_accuracy()
    assert effective_acc == acc, (
        f"Test Lemma 1. Effective acc. {effective_acc} != {acc}"
    )
    print("Test Lemma 1. Effective acc:", effective_acc, "acc:", acc)
