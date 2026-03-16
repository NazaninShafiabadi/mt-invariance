# mt-invariance

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](link_to_paper)
[![Dataset](https://img.shields.io/badge/Dataset-BiMultiSD--XLT-blue.svg)](link_to_huggingface_or_data_folder)

This repository contains the official code, data, and evaluation framework for the paper **"Biases in Translation: Assessing Opinion Distortion in Machine Translated Texts"** (Shafiabadi and Yvon, LREC 2026).

### Overview
Machine translation (MT) systems are often assumed to preserve the subjective meaning of source texts, but subtle translation artifacts can easily distort a speaker's intended position. This repository provides tools to detect and quantify these shifts. 

**This repository includes:**
* **The Evaluation Framework:** Scripts to train and apply a stance classifier, translate texts, and run statistical hypothesis tests (population-level shifts and paired flips) to quantify stance distortion in MT outputs.
* **BiMultiSD-XLT:** A harmonized multilingual stance detection corpus standardized for binary stance classification, containing easily-separable native and translated stance-bearing comments.
* **Curated Calibration Set:** 100 high-quality French stance-reversed examples for controlled perturbation testing.

---

## ⚙️ Setup and Installation

## 🚀 Usage / Reproducing Results

## 📖 Citation
If you use this code or the BiMultiSD-XLT dataset in your research, please cite our paper:
```
@inproceedings{shafiabadi-yvon-2026-biases,
  title     = {Biases in Translation: Assessing Opinion Distortion in Machine Translated Texts},
  author    = {Shafiabadi, Nazanin and Yvon, François},
  booktitle = {Proceedings of the {{Fifteenth International Conference}} on {{Language Resources}} and {{Evaluation}} ({{LREC}}'26)},
  year      = {2026}
}
```
