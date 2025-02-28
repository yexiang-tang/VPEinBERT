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
The method was to determine whether BERT detects **silenced verb-phrase structures** (VP-Ellipsis). Specifically, we generated pairs (or triplets) of sentences—one elliptical (with a missing verb phrase), one fully stated, and one contrast sentence with a distinct meaning but identical surface structure. We then used BERT to encode each sentence and measured their embedding similarities (cosine similarity, Euclidean distance, and a ratio of cosine similarity) to assess whether elliptical sentences aligned more with their full counterparts than with unrelated contrasts.

**Dataset**  
- **Sentence Generation**: We used a Context-Free Grammar (CFG) in Python/NLTK to systematically produce **3,616 sentence triplets**:  
  1. **Elliptical** sentence (e.g., "Sarah teaches history, and John does.")  
  2. **Full** sentence (fully tated the verb phrase, e.g., "Sarah teaches history, and John teaches history.")  
  3. **Contrast** sentence (same surface structure, different meaning, e.g., "Sarah teaches history, and John hikes")  
- **Structure Types**: We included various verb tenses and auxiliaries (e.g., present tense, modals, negation) to cover multiple syntactic forms.  
- **Verb Variation**: Contrast sentences were subdivided into a *selected* group (semantically closer to the elliptical verb) and a *random* group (more distant verbs) to test sensitivity to lexical similarity.  
- **Availability**: The dataset is generated programmatically; you can reproduce or customize it by running `Sentence_Generator.py` in this repository. If needed, you can adapt the CFG rules for different sentence structures.


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
 



