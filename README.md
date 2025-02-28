# Compositionality in LLMs: testing VP-Ellipsis within BERT

## 1. Brief Summary of the Paper

**Context & Motivation**  
Compositionality is the principle that a complex expression’s meaning is determined by the meanings of its parts and their structure. Traditional symbolic theories, such as Jerry Fodor’s Language of Thought Hypothesis, posit that cognition is governed by rule-based symbolic operations. In contrast, Large Language Models (LLMs) like BERT use distributed representations and data-driven learning. This thesis investigates whether BERT encodes compositionality by focusing on verb-phrase ellipsis (VP-Ellipsis)—cases where a verb phrase is silenced but implicitly understood (e.g., “Sarah teaches history, but John does not”). Detecting this hidden structure would provide evidence of compositional reasoning rather than mere pattern-matching.

**Key Findings**  
The key observances are:

- Moderate Evidence of Compositionality: Elliptical–full pairs generally scored higher similarity than elliptical–contrast pairs.
- Variability Across Structures: The negation-enhanced clause structures were more inconsistent, sometimes dropping below the expected ratio of 1.
- Partial but Not Robust: While BERT appears to capture some silenced structures, the overall effect sizes were small, indicating only a weak compositional understanding.

These outcomes suggest BERT indeed encodes certain compositional relationships, though further research is needed to fully determine how robustly and systematically LLMs capture compositionality.

---

## 2. Experiment and Dataset

**Experiment Design**  
Our primary experiment tests whether BERT detects silenced verb-phrase structures. 
We created **sentence triplets**—consisting of *elliptical*, *full*, and *contrast* sentences—to determine if 
elliptical embeddings align more similarly with their full counterparts than unrelated contrasts.
The key hypothesis is that a higher similarity between elliptical–full pairs (relative to elliptical–contrast) 
would indicate that BERT encodes silenced structure in elliptical sentences, suggesting compositional understanding.

**Dataset**  
We generated **3,600+ sentence triplets** via a **Context-Free Grammar (CFG)** generaotr in Python 
(for more detail, find [NLTK library’s grammar generator](https://www.nltk.org/) (Qi et al., 2020)).
Each triplet includes:  
- **Elliptical sentence** (e.g., “Sally teaches history, and Bill does”)  
- **Full sentence** (e.g., “Sally teaches history, and Bill teaches history”)  
- **Contrast sentence** (e.g., "Sally teahces history, and Bill hikes.")

---
## 3. Model

**Model Architecture**  
We utilize the `bert-large-uncased-whole-word-masking` variant of BERT (24 layers, 1024-dimensional hidden states) because:
- **Bidirectional Context**: It processes both left and right contexts simultaneously, ideal for capturing nuanced sentence structures.
- **Robust Embeddings**: We can directly analyze embedding vectors (rather than generated text), making it easier for internal analysis of BERT.
- **Implementation**: We employ the [Hugging Face Transformers](https://github.com/huggingface/transformers) library in a Python environment.

During the experiment, each sentence (elliptical, full, or contrast) is tokenized and fed into the BERT model. We then extract the final-layer `[CLS]` token embedding (size 1×1024) to represent the entire sentence. No additional fine-tuning steps were performed; we rely on BERT’s pre-trained weights to measure how well it inherently encodes the silenced structures.

---

## 4. Results
**Key Metrics**  
We evaluated the following metrics to measure sentence similarity:
- **Cosine Similarity**: Reflects how closely two embedding vectors align in angle.  
- **Euclidean Distance**: Overall distance in the high-dimensional space.  
- **Ratio (Elliptical–Full vs. Elliptical–Contrast)**: A score >1 suggests elliptical sentences are embedded more closely to their full versions than to unrelated contrasts.
 



