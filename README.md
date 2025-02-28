# Compositionality in Large Language Models: Testing VP-Ellipsis in BERT

This repository contains the code and resources for the paper:

> **“Compositionality in Large Language Models: Testing VP-Ellipsis within BERT.”**

The project investigates whether BERT encodes compositionality by testing its ability to detect hidden structures in elliptical sentences. We generate sentence triplets—**full**, **elliptical**, and **contrast**—and compare embeddings to see if BERT represents elliptical sentences similarly to their fully stated counterparts.

## Table of Contents
- [Compositionality in Large Language Models: Testing VP-Ellipsis in BERT](#compositionality-in-large-language-models-testing-vp-ellipsis-in-bert)
  - [Table of Contents](#table-of-contents)
  - [Repository Overview](#repository-overview)
  - [Dependencies and Installation](#dependencies-and-installation)
    - [Installation](#installation)
  - [Usage](#usage)
    - [1. Generate Sentences](#1-generate-sentences)

---

## Repository Overview

This project explores **verb-phrase ellipsis (VP-Ellipsis)** in BERT embeddings, focusing on whether the model encodes missing syntactic elements. The workflow includes:

1. **Sentence Generation**  
   - Context-Free Grammar (CFG) to produce elliptical sentences paired with full and contrast versions.

2. **Embedding Extraction**  
   - Using `bert-large-uncased-whole-word-masking` to obtain `[CLS]` vectors.

3. **Metric Computation**  
   - Cosine similarity, Euclidean distance, and a ratio measuring how similar elliptical–full pairs are versus elliptical–contrast pairs.

4. **Statistical Analysis & Visualization**  
   - Distribution plots, correlation heatmaps, summary statistics.

---

## Dependencies and Installation

Below are the main dependencies needed for this project. Adjust to your environment as necessary:

- **Python 3.7+**  
- [PyTorch](https://pytorch.org/)  
- [Transformers (Hugging Face)](https://github.com/huggingface/transformers)  
- [NLTK](https://www.nltk.org/)  
- [Stanza](https://stanfordnlp.github.io/stanza/) (optional, depending on your NLP tasks)  
- [Pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Seaborn](https://seaborn.pydata.org/)  
- [SciPy](https://scipy.org/)  
- [scikit-learn](https://scikit-learn.org/stable/)

### Installation
1. Clone or download this repository.
2. Install required libraries
3. Update any file paths inside the scripts to match your local environment.
4. Update any metrics columns for any particular metrics analysis.

---

## Usage

### 1. Generate Sentences
- **Script:** `Sentence_Generator.py`  
- **Description:** Uses a CFG (via NLTK) to produce elliptical, full, and contrast sentences. Outputs a CSV of generated sentences.

**Run:**
```bash
python Sentence_Generator.py
