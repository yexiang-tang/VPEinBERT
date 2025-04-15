# Compositionality in LLMs: testing VP-Ellipsis within BERT
This repository is part of Yexiang Tang's Honors Thesis *Compositionality in Large-Language-Models: Testing VP-Ellipsis with BERT*. The study investigates whether transformer-based models like BERT exhibit **compositional understanding**—the principle that complex meanings are built from the meanings and structure of simpler parts—by analyzing how BERT encodes verb-phrase ellipsis (VP-Ellipsis).

## Overview

- **Goal**: Test whether BERT detects silenced syntactic structures in elliptical sentences, similar to how humans process language compositionally.

- **Approach**: Compare the sentence embeddings of elliptical sentences to their full and contrast counterparts.

- **Key Hypothesis**: If BERT is compositional, embeddings of elliptical sentences should resemble their full versions more than unrelated contrast sentences.

---

## Methodology

### Dataset
We generated **3,600+ sentence triplets** with a **Context-Free Grammar (CFG)** via the NLTK toolkit:

- **Full Sentence**: e.g., _“Sally teaches history, and John teaches history.”_
- **Elliptical Sentence**: e.g., _“Sally teaches history, and John does.”_
- **Contrast Sentence**: e.g., _“Sally teaches history, and John hikes.”_

Sentence structures were carefully controlled and categorized into three syntactic types:  
- **Type-0**: Basic clause  
- **Type-1**: Auxiliary-enhanced clause  
- **Type-2**: Negation-enhanced clause

Each contrast verb was selected to either be semantically similar (selected group) or dissimilar (random group) to the original verb, and lexical similarity was measured and validated.

### Model

We used the `bert-large-uncased-whole-word-masking` variant from Hugging Face’s Transformers library. Embeddings were extracted from the `[CLS]` token of the final layer for each sentence. No additional fine-tuning was performed.

---

## Metrics & Analysis

Three metrics were used to evaluate embedding similarity:

- **Cosine Similarity**
- **Euclidean Distance**
- **Cosine Similarity Ratio (Elliptical–Full / Elliptical–Contrast)**  
  A ratio > 1 indicates BERT encodes the ellipsis structure similarly to the full sentence.

---

## Key Findings

- **Evidence of Compositionality**: On average, elliptical–full pairs were more similar than elliptical–contrast pairs.
- **Effect Size is Small**: The mean cosine similarity ratio was only slightly above 1 (~1.037).
- **Structural Variation Matters**: Some clause types (especially negation-enhanced) disrupted the expected similarity pattern.
- **Word-Specific Effects not Present**: No strong correlations between individual words and metrics, indicating the content and strcuture—not particular lexical choice—are the key driver.

---

## Code Components

This repository includes all scripts used in dataset construction, model evaluation, and result visualization:

| File | Description |
|------|-------------|
| `Sentence_Generator.py` | Generates syntactically controlled sentence triplets using NLTK's CFG. |
| `embedding_vector_generator.py` | Passes sentences through BERT and extracts `[CLS]` token embeddings. |
| `metrics_calculation.py` | Computes cosine similarity and Euclidean distance between embeddings. |
| `metrics_summary.py` | Summarizes metric statistics and outputs CSVs for analysis. |
| `analysis_in_words.py` | Analyzes how individual words influence similarity metrics, includes visualizations. |
| `visualizations.py` | Creates heatmaps, boxplots, and histograms to illustrate experimental results. |

Each component is modular, allowing for easy adaptation to new datasets.

---

## Paper & Citation

For full background, theoretical discussion, and results, see the full paper:  
[`Compositionality in Large-Language-Models.pdf`](./Compositionality_in_Large_Language_Models.pdf)

If you use or build on this work, please cite appropriately (citation format coming soon).

---

## Future Work

Potential directions include:
- Scaling up dataset size
- Testing other hidden structures (e.g., sluicing, gapping, anaphora)
- Exploring different linguistic theories

---