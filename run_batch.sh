#!/bin/bash
# This script runs the evaluation of the CoIN-Bench dataset on different splits


# 1. val unseen
CUDA_VISIBLE_DEVICES="0" python -m vlfm.run habitat.task.measurements.success.success_distance="0.25" habitat_baselines.eval.split="val_unseen" habitat.dataset.data_path="CoIN-Bench/val_unseen/val_unseen.json.gz"

# 2. val_seen_synonyms
time CUDA_VISIBLE_DEVICES="0" python -m vlfm.run habitat.task.measurements.success.success_distance="0.25" habitat_baselines.eval.split="val_seen_synonyms" habitat.dataset.data_path="CoIN-Bench/val_seen_synonyms/val_seen_synonyms.json.gz"

# 3. val_seen
time CUDA_VISIBLE_DEVICES="0" python -m vlfm.run habitat.task.measurements.success.success_distance="0.25" habitat_baselines.eval.split="val_seen" habitat.dataset.data_path="CoIN-Bench/val_seen/val_seen.json.gz"       
