#!/usr/bin/env python3
"""Given a data file with LM QA predictions, evaluate the predictions.
"""
import argparse
import json
import logging
import statistics
import sys
from copy import deepcopy
from tqdm import tqdm
from xopen import xopen
import string
from typing import List
import regex


logger = logging.getLogger(__name__)


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


METRICS = [
    (best_subspan_em, "best_subspan_em"),
]


def main(
    input_path,
    output_path,
):
    all_examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)

    # Compute normal metrics in parallel, if applicable
    logger.info("Computing metrics")
    all_example_metrics = []
    for example in tqdm(all_examples):
        all_example_metrics.append(get_metrics_for_example(example))

    # Average metrics across examples
    for (_, metric_name) in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        logger.info(f"{metric_name}: {average_metric_value}")

    if output_path:
        with xopen(output_path, "w") as f:
            for (example_metrics, example) in all_example_metrics:
                example_with_metrics = deepcopy(example)
                for metric_name, metric_value in example_metrics.items():
                    example_with_metrics[f"metric_{metric_name}"] = metric_value
                f.write(json.dumps(example_with_metrics) + "\n")


def get_metrics_for_example(example):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with model predictions and answers.", required=True)
    parser.add_argument(
        "--output-path",
        help="Path to write data with model predictions, answers, and scores.",
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
