# Topological Mammography Tri-Shell Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23964083.svg)](https://doi.org/10.5281/zenodo.23964083)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official PyTorch implementation of the **Tri-Shell Topological Latent Modulation (TLM)** framework for robust mammographic mass classification, multi-center domain adaptation, and clinical screening triage.

## Abstract
This repository contains the authoritative codebase for our Q1 journal study on geometry-preserving Tri-Shell architectures, multi-zonal persistent homology extraction, and robust uncertainty-calibrated deep learning for digital mammography.

## Repository Structure
- `topological_latent_modulation.py`: Core PyTorch modules for Concentric Tri-Shell gating and persistent homology feature integration.
- `benchmark_ablation_simulation.py`: Ablation and empirical evaluation scripts.
- `requirements.txt`: Python package dependencies.
- `LICENSE`: Open-source MIT License.

## Installation
Clone the repository and install dependencies:

- git clone https://github.com/mkarimkhani/topological-mammography-tri-shell.git
- cd topological-mammography-tri-shell
- pip install -r requirements.txt

## Citation
If you use this code in your research, please cite our corresponding paper:

```bibtex
@software{karimkhani2026trishell,
  author       = {Karimkhani, Mohsen},
  title        = {Concentric Tri-Shell Topological Latent Modulation for Calibrated Mammographic Mass Triage},
  month        = nov,
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.23964083},
  url          = {https://doi.org/10.5281/zenodo.23964083}
}
