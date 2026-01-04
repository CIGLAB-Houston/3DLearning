# 3D-Learning


---

##  Overview

This repository implements **3D-Learning: Diffusion-Augmented Distributionally Robust Decision-Focused Learning**, a framework designed to improve the robustness of Predict-then-Optimize (PTO) pipelines under out-of-distribution (OOD) conditions.

In many computing and networked systems (e.g., cloud LLM serving, data center demand response, and edge workload scheduling), machine learning models are first used to predict contextual information, which is then consumed by downstream optimization or decision-making modules. While effective under in-distribution settings, such predictors often suffer from severe performance degradation when encountering OOD inputs, leading to suboptimal or unstable decisions.

To address this challenge, this codebase implements **Distributionally Robust Decision-Focused Learning (DR-DFL)** using a diffusion-based approach. Instead of relying on classical DRO methods with hand-crafted ambiguity sets, **3D-Learning parameterizes the uncertainty set using a diffusion model**, enabling the search for worst-case data distributions that remain realistic and consistent with observed data.

The implementation demonstrates that 3D-Learning achieves a favorable balance between average-case and worst-case performance, significantly improving OOD generalization compared to classical DRO and data augmentation baselines.

---

<p align="center">
  <a href="Documents/3D Learning.pdf">
    <img src="Documents/Framework.jpg" width="800">
  </a>
</p>

<p align="center">
  <em>Figure 1: Overview of the 3D-Learning framework.</em>
</p>


📄 **Paper link:**  
https://arxiv.org/abs/2510.22757

---

##  Features
Specifically, this repository provides:
- Training pipelines for decision-focused learning under distributional robustness
- Diffusion-based generation of adversarial yet realistic worst-case distributions
- End-to-end integration of prediction models and downstream decision objectives
- Experimental evaluation on decision-centric tasks such as LLM resource provisioning

---

##  Environment & Dependencies

### Python Version
- Python **>= 3.9** is recommended (Python **= 3.12.0** is used in our work)

### Core Dependencies

The main dependencies used in this project are:

| Package           | Version        |
|-------------------|----------------|
| matplotlib        | 3.10.8         |
| numpy             | 2.4.0          |
| pandas            | 2.3.3          |
| perlin_noise      | 1.14           |
| Pillow            | 12.1.0         |
| psutil            | 5.9.0          |
| scikit_learn      | 1.8.0          |
| scipy             | 1.16.3         |
| torch             | 2.7.0+cu128    |
| torchvision       | 0.22.0+cu128   |
| torch_fidelity    | 0.3.0          |
| tqdm              | 4.67.1         |


A full list of dependencies is provided in `requirements.txt`.

---

## Installation

### 1. Create and activate a Conda virtual environment

```bash
conda create -n 3d-learning python=3.12.0
conda activate 3d-learning
```
### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run

Please navigate to the `Exp_Script/Azure` directory and run `exp_scrip.py` to execute the experiments.

The script is pre-configured with the main experimental settings used in the paper, corresponding to the results reported in **Table 1**.



