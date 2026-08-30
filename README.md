# mt-invariance

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://doi.org/10.63317/2pjio9ho8rxg)
[![Dataset](https://img.shields.io/badge/Dataset-BiMultiSD--XLT-blue.svg)](https://github.com/NazaninShafiabadi/mt-invariance/tree/main/data)

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
@inproceedings{shafiabadi-etal-2026-biases,
  title = {Biases in Translation: Assessing Opinion Distortion in Machine Translated Texts},
  author = {Shafiabadi, Nazanin and Yvon, François},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {8596--8614},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/2pjio9ho8rxg},
  abstract = {Current machine translation (MT) evaluation practices largely assume that high lexical and semantic fidelity implies preservation of meaning. We question this assumption by introducing a framework for detecting and quantifying translation-induced distortion—the systematic alteration of a text’s subjective properties during translation. Focusing on stance as a socially consequential property, we formalize stance preservation as an invariance problem and adapt two classical statistical tests, McNemar’s test and the two-proportion Z-test, to diagnose systematic opinion shifts between source texts and their translations. Unlike standard MT metrics such as BLEU or COMET, which prioritize surface similarity and adequacy, our approach explicitly targets preservation of subjective meaning. In controlled experiments with synthetically distorted translations, we demonstrate that the proposed tests are sensitive to graded levels of stance manipulation. We apply our framework to evaluate twelve multilingual models and find that none reliably preserve stance across all tested language directions. Our findings reveal a critical gap in current MT evaluation practices and highlight the need for explicit evaluation of subjective meaning preservation in socially and politically sensitive contexts.}
}
```
