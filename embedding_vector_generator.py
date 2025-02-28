import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------
# 1. MODEL CONFIGURATION
# ----------------------------------------------------
MODEL_NAME = "bert-large-uncased-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ----------------------------------------------------
# 2. HELPER FUNCTION: EXTRACT [CLS] VECTOR
# ----------------------------------------------------
def extract_cls_vector(sentence: str):
    inputs = tokenizer(
        sentence, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # [0, 0] = first sequence, first token (i.e., [CLS])
    cls_vector = outputs.last_hidden_state[0, 0]
    return cls_vector.numpy()

# ----------------------------------------------------
# 3. PROCESS & STORE EMBEDDINGS
# ----------------------------------------------------
def store_embeddings(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    required_columns = ["elliptical_sentences", "full_sentences", 
                        "contrast_sentences", "tree_type"]
    
    # Validate required columns
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the input CSV.")

    embeddings_list = []

    # Extract embeddings for each row
    for _, row in df.iterrows():
        ell_vec = extract_cls_vector(row["elliptical_sentences"])
        full_vec = extract_cls_vector(row["full_sentences"])
        contrast_vec = extract_cls_vector(row["contrast_sentences"])

        embeddings_list.append({
            "tree_type": row["tree_type"],
            "elliptical_sentences": row["elliptical_sentences"],
            "full_sentences": row["full_sentences"],
            "contrast_sentences": row["contrast_sentences"],
            "ell_vec": ell_vec.tolist(),
            "full_vec": full_vec.tolist(),
            "contrast_vec": contrast_vec.tolist(),
        })

    # Save embeddings to CSV
    embeddings_df = pd.DataFrame(embeddings_list)
    embeddings_df.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")

# ----------------------------------------------------
# 4. EXECUTION
# ----------------------------------------------------
input_path = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\Generated sentences\sentences_random.CSV"
output_path = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\embedding_vectors_random.CSV"
store_embeddings(input_path, output_path)
