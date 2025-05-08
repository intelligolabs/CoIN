
<h1 align="center">
Collaborative Instance Navigation:
Leveraging Agent Self-Dialogue to Minimize User Input
</h1>

<div>
    <p align="center">
    <a href='https://francescotaioli.github.io/' target='_blank'>Francesco Taioli</a>,
    Edoardo Zorzi,
    Gianni Franchi,
    Alberto Castellini, Alessandro Farinelli, Marco Cristani, Yiming Wang
    </p>
</div>

<h3 align="center">
<!-- <a href="">Paper</a> | -->
 <!-- <a href="">Video</a> | -->
 <!-- Accepted to
  <a href=""></a></h3> -->

<div align="center">
  <strong><a href="https://intelligolabs.github.io/CoIN">Project Page</a></strong>
</div>


<p align="center">
contact: <code>francesco.taioli@polito.it</code>
</p>
<hr>

<!-- > [!IMPORTANT]
> Consider citing our paper:
> ```BibTeX

>   ``` -->


## Abstract
Existing embodied instance goal navigation tasks, driven by natural language, assume human users to provide complete and nuanced instance descriptions prior to the navigation, which can be impractical in the real world as human instructions might be brief and ambiguous. To bridge this gap, we propose a new task, Collaborative Instance Navigation (CoIN), with dynamic agent-human interaction during navigation to actively resolve uncertainties about the target instance in natural, template-free, open-ended dialogues. To address CoIN, we propose a novel method, Agent-user Interaction with UncerTainty Awareness (AIUTA), leveraging the perception capability of Vision Language Models (VLMs) and the capability of Large Language Models (LLMs). First, upon object detection, a Self-Questioner model initiates a self-dialogue to obtain a complete and accurate observation description, while a novel uncertainty estimation technique mitigates inaccurate VLM perception. Then, an Interaction Trigger module determines whether to ask a question to the user, continue or halt navigation, minimizing user input. For evaluation, we introduce CoIN-Bench, a benchmark supporting both real and simulated humans. AIUTA achieves competitive performance in instance navigation against state-of-the-art methods, demonstrating great flexibility in handling user inputs.

Table of contents
=================
Release timeline:
- [ ] AIUTA (upon acceptance)
  - [x] Multimodal Uncertainty Estimation (```notebook``` folder)
- [ ] CoIN-Bench (upon acceptance)
- [ ] IDKVQA dataset (upon acceptance)

# Quick start
## Multimodal Uncertainty Estimation
In the ```notebook``` folder, run ```perception_uncertainty_estimation.ipynb``` for a quick, well-commented example of multimodal uncertainty estimation.



# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>. (Website branch)
