# Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards

<div align="center">

[![arXiv](https://img.shields.io/badge/arxiv-2503.11240-b31b1b)](https://arxiv.org/abs/2503.11240)&nbsp;
[![code](https://img.shields.io/badge/code-B2--DiffuRL-blue)](https://github.com/hu-zijing/B2-DiffuRL)&nbsp;

</div>

<p align="center">
<img src="assets/method.png" width=90%>
<p>

## Introduction

This work presents a novel RL-based framework that addresses the sparse reward problem when training diffusion models. Our framework, named $\text{B}^2\text{-DiffuRL}$, employs two strategies: **B**ackward progressive training and **B**ranch-based sampling. For one thing, backward progressive training focuses initially on the final timesteps of denoising process and gradually extends the training interval to earlier timesteps, easing the learning difficulty from sparse rewards. For another, we perform branch-based sampling for each training interval. By comparing the samples within the same branch, we can identify how much the policies of the current training interval contribute to the final image, which helps to learn effective policies instead of unnecessary ones. $\text{B}^2\text{-DiffuRL}$ is compatible with existing optimization algorithms. Extensive experiments demonstrate the effectiveness of $\text{B}^2\text{-DiffuRL}$ in improving prompt-image alignment and maintaining diversity in generated images.

## Run

```bash
bash run_process.sh > log/exp_B2DiffuRL_b5_p3
```
This will start fine-tuning, and store the results under `model/`. The pipeline consists of **sampling** by `run_sample.py`, **evaluation** by `run_select.py` and **training** by `run_train.py`. 

The full hyperparameters are shown in `config/stage_process.py`, and many of them can be modified in `run_process.sh`. Please note that the default parameters are not meant to achieve best performance. 

## Acknowlegement

This repository was built with much reference to the following repositories: 

* [DDPO](https://github.com/jannerm/ddpo)
* [D3PO](https://github.com/yk7333/D3PO/tree/main)
* [DDPO-Pytorch](https://github.com/kvablack/ddpo-pytorch)
* [DPOK](https://github.com/google-research/google-research/tree/master/dpok)

## Citation
If our work assists your research, feel free to cite us using:

```
@misc{hu2025betteralignmenttrainingdiffusion,
      title={Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards}, 
      author={Zijing Hu and Fengda Zhang and Long Chen and Kun Kuang and Jiahui Li and Kaifeng Gao and Jun Xiao and Xin Wang and Wenwu Zhu},
      year={2025},
      eprint={2503.11240},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.11240}, 
}
```
