#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}
export SAM_PORT=${SAM_PORT:-12183}

export LLava_PORT=${LLava_PORT:-12189}
export LLAMA_PORT=${LLAMA_PORT:-12190}

CUDA_DEVICE=0

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Split both panes horizontally
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2
tmux split-window -h -t ${session_name}:0.3

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

tmux send-keys -t ${session_name}:0.3 "CUDA_VISIBLE_DEVICES=\"${CUDA_DEVICE},${CUDA_DEVICE+1}\" ${VLFM_PYTHON} -m vlfm.vlm.llava_next --port ${LLava_PORT}" C-m
# tmux send-keys -t ${session_name}:0.4 "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE+1} ${VLFM_PYTHON} -m vlfm.vlm.llama_3 --port ${LLAMA_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
