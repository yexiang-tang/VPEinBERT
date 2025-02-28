# This program calculates corresponding statistics for each metrics 
# and stores the result into a CSV file.


import pandas as pd

# File paths
input_select = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\results_selected.CSV"
input_random = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\results_random.CSV"
output_file = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\summary.CSV"

def compute_statistics(data, label):
    metrics = {
        "Cosine Similarity (Elliptical vs Full)": "Cosine Sim (Ell vs Full)",
        "Cosine Similarity (Elliptical vs Contrast)": "Cosine Sim (Ell vs Contrast)",
        "Cosine Similarity Ratio (X/Y)": "Cosine Sim Ratio (X/Y)",
        "Euclidean Distance (Elliptical vs Full)": "Euclidean Dist (Ell vs Full)",
        "Euclidean Distance (Elliptical vs Contrast)": "Euclidean Dist (Ell vs Contrast)"
    }

    # Overall statistics
    overall_stats = []
    for column, short_name in metrics.items():
        overall_stats.append({
            "Source": label,
            "Tree Type": "Overall",
            "Metric": short_name,
            "Mean": data[column].mean(),
            "Median": data[column].median(),
            "Std Dev": data[column].std(),
            "Min": data[column].min(),
            "Max": data[column].max()
        })

    # Statistics by Tree Type
    tree_stats = []
    for tree_type, group in data.groupby("Tree Type"):
        for column, short_name in metrics.items():
            tree_stats.append({
                "Source": label,
                "Tree Type": tree_type,
                "Metric": short_name,
                "Mean": group[column].mean(),
                "Median": group[column].median(),
                "Std Dev": group[column].std(),
                "Min": group[column].min(),
                "Max": group[column].max()
            })

    # Combine overall and tree-specific stats
    return pd.DataFrame(overall_stats + tree_stats)

# Load and process the selected contrast
data_1 = pd.read_csv(input_select)
stats_1 = compute_statistics(data_1, label="selected contrast")

# Load and process the random contrast
data_2 = pd.read_csv(input_random)
stats_2 = compute_statistics(data_2, label="random contrast")

# Combine results and save
combined_stats = pd.concat([stats_1, stats_2], ignore_index=True)
combined_stats.to_csv(output_file, index=False)

print(f"Result saved to {output_file}")
