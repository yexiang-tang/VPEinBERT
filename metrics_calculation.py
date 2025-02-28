import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import csv

# Define file paths
file_path = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\embedding_vectors_random.CSV"
output_path = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\results_random.CSV"

# Load the CSV file
data = pd.read_csv(file_path)

results = []
for _, row in data.iterrows():
    try:
        # Parse embeddings from string to list
        ell_vec = np.array(ast.literal_eval(row['ell_vec']))
        full_vec = np.array(ast.literal_eval(row['full_vec']))
        contrast_vec = np.array(ast.literal_eval(row['contrast_vec']))
        
        # Store tree_type and original sentences
        tree_type = row['tree_type']
        elliptical_sentence = row['elliptical_sentences']
        full_sentence = row['full_sentences']
        contrast_sentence = row['contrast_sentences']
        
        # Cosine similarities
        x = cosine_similarity(ell_vec.reshape(1, -1), full_vec.reshape(1, -1))[0, 0]  # Elliptical vs Full
        y = cosine_similarity(ell_vec.reshape(1, -1), contrast_vec.reshape(1, -1))[0, 0]  # Elliptical vs Contrast
        ratio = x / y if y != 0 else float('inf')  # Avoid division by zero
        
        # Euclidean distances
        euclidean_full = np.linalg.norm(ell_vec - full_vec)  # Elliptical vs Full
        euclidean_contrast = np.linalg.norm(ell_vec - contrast_vec)  # Elliptical vs Contrast
        
        # Results
        results.append({
            "Tree Type": tree_type,
            "Elliptical Sentence": elliptical_sentence,
            "Full Sentence": full_sentence,
            "Contrast Sentence": contrast_sentence,
            "Cosine Similarity (Elliptical vs Full)": x,
            "Cosine Similarity (Elliptical vs Contrast)": y,
            "Cosine Similarity Ratio (X/Y)": ratio,
            "Euclidean Distance (Elliptical vs Full)": euclidean_full,
            "Euclidean Distance (Elliptical vs Contrast)": euclidean_contrast,
        })
    except (ValueError, SyntaxError): 
        continue

# Write the results to a CSV file
with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "Tree Type",
        "Elliptical Sentence", 
        "Full Sentence", 
        "Contrast Sentence", 
        "Cosine Similarity (Elliptical vs Full)", 
        "Cosine Similarity (Elliptical vs Contrast)", 
        "Cosine Similarity Ratio (X/Y)", 
        "Euclidean Distance (Elliptical vs Full)", 
        "Euclidean Distance (Elliptical vs Contrast)"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # Header
    writer.writerows(results)  # Data

print(f"Results saved to {output_path}")
