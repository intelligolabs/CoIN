<p align="center">
  <img src="docs/teaser.jpg" >
  </p>
</p>


<h1 align="center">
Collaborative Instance Object Navigation: Leveraging Uncertainty-Awareness to Minimize Human-Agent Dialogues [ICCV 25]
</h1>

<div>
    <p align="center">
    🎉 ICCV 2025 - <a href="https://intelligolabs.github.io/CoIN">Project Page</a></strong> 🎉 
    </p>
</div>
<div>
    <p align="center">
    <a href='https://francescotaioli.github.io/' target='_blank'>Francesco Taioli</a>,
    <a href='https://github.com/e-zorzi' target='_blank'>Edoardo Zorzi</a>,
    Gianni Franchi,
    Alberto Castellini, Alessandro Farinelli, Marco Cristani, Yiming Wang
    </p>
</div>

<h3 align="center">
<!-- <a href="">Paper</a> | -->
 <!-- <a href="">Video</a> | -->
 <!-- Accepted to
  <a href=""></a></h3> -->


<hr>

<!-- > [!IMPORTANT]
> Consider citing our paper:
> ```BibTeX

>   ``` -->


## Abstract
Existing embodied instance goal navigation tasks, driven by natural language, assume human users to provide complete and nuanced instance descriptions prior to the navigation, which can be impractical in the real world as human instructions might be brief and ambiguous. To bridge this gap, we propose a new task, Collaborative Instance Navigation (CoIN), with dynamic agent-human interaction during navigation to actively resolve uncertainties about the target instance in natural, template-free, open-ended dialogues. To address CoIN, we propose a novel method, Agent-user Interaction with UncerTainty Awareness (AIUTA), leveraging the perception capability of Vision Language Models (VLMs) and the capability of Large Language Models (LLMs). First, upon object detection, a Self-Questioner model initiates a self-dialogue to obtain a complete and accurate observation description, while a novel uncertainty estimation technique mitigates inaccurate VLM perception. Then, an Interaction Trigger module determines whether to ask a question to the user, continue or halt navigation, minimizing user input. For evaluation, we introduce CoIN-Bench, a benchmark supporting both real and simulated humans. AIUTA achieves competitive performance in instance navigation against state-of-the-art methods, demonstrating great flexibility in handling user inputs.

# Quick start

## Installation
The codebase is built on top of [VLFM](https://github.com/bdaiinstitute/vlfm). You should clone this repo, and see VLFM repository for the installation instructions. Note that we **do not** need the Matterport scene dataset, since we use the ```HM3D``` dataset.

After setting up VLFM, you can install some additional requirements for CoIN with the following command:
```bash
pip install -r our_requirements.txt
```
## Download CoIN-Bench
You can download the CoIN-Bench dataset from [HuggingFace](https://huggingface.co/datasets/ftaioli/CoIN-Bench). Do this in the root directory of this repository.

## LLM key
To use the LLM client, you need to set up an environment variable with your OpenAI key (or Groq key).

Simply set the `COIN_LLM_CLIENT_KEY` environmental variable to your OpenAI key (or [Groq, free key](https://groq.com/), or Cerebras etc.), like

```bash
export COIN_LLM_CLIENT_KEY=<your_key>
```

in the file where you export your env variables.

If you want to use Groq (or alternative providers), the variable in ```test_with_groq``` should be set to ```True``` in ```vlfm/policy/base_objectnav_policy.py```.

# Usage
## Evaluation
You should be ready to go. To launch the eval:
1. Launch the VLFM server( G.Dino, LLava, etc):
```bash
./scripts/launch_vlm_servers.sh
```
2. Launch the evaluation script:
```bash
./run_batch.sh
```
## Multimodal Uncertainty Estimation
In the ```notebook``` folder, run ```perception_uncertainty_estimation.ipynb``` for a quick, well-commented example of multimodal uncertainty estimation.

## (Optional) Check our IDKVQA dataset.
You can download the IDKVQA dataset from [HuggingFace](https://huggingface.co/datasets/ftaioli/IDKVQA).

To reproduce the baseline in our paper, run the following command:

> python idkvqa/llava_vqa_get_answer.py

This will run llava on the IDKVQA dataset and generate all the answers (contained in the folder 'model_results'). Then, run the evaluation script:

> python idkvqa/llava_vqa_eval.py

## :black_nib: Citation

If you use CoIN in your research, please use the following BibTeX entry.

```
@InProceedings{Taioli_2025_ICCV,
    author    = {Taioli, Francesco and Zorzi, Edoardo and Franchi, Gianni and Castellini, Alberto and Farinelli, Alessandro and Cristani, Marco and Wang, Yiming},
    title     = {Collaborative Instance Object Navigation: Leveraging Uncertainty-Awareness to Minimize Human-Agent Dialogues},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {18781-18792}
}
```

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>. (Website branch)