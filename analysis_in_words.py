# This program calculates the correlation between words and metrics. With visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_ind

# 1. LOAD DATA
file_path = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\results_denoted_random.CSV" # Change the file rout accordingly
df = pd.read_csv(file_path)

# Ensure "Tree Type" is correctly typed
df["Tree Type"] = df["Tree Type"].astype(int)

# Define keywords for detection
keywords = ["and", "but", "can", "will", "cannot", "does", "did", "not"]

# 2. CORRELATION: KEYWORD PRESENCE vs. Metric
# Create a temporary DataFrame with binary columns for each keyword
df_temp = pd.DataFrame({
    kw: df["Elliptical Sentence"].str.contains(rf"\b{kw}\b", case=False, na=False).astype(int)
    for kw in keywords
})

# Add similarity ratio for correlation, it can be changed based on what metric we are analyzing
df_temp["Ratio"] = df["Cosine Similarity Ratio (X/Y)"]

# Compute and plot correlation matrix
corr_matrix = df_temp.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Verb Presence vs. Ratio of Similarity (random)")
plt.show()

# 3. SPEARMAN CORRELATION: TREE TYPE vs. Metric
tree_types = df["Tree Type"].unique()
tree_type_correlation = {}

for tree_type in tree_types:
    is_tree = (df["Tree Type"] == tree_type).astype(int)
    correlation, _ = spearmanr(is_tree, df["Cosine Similarity Ratio (X/Y)"])  # Again, this can be altered based on the need of metric
    tree_type_correlation[tree_type] = correlation

# Convert correlation dictionary to DataFrame
tree_ratio_corr_df = pd.DataFrame.from_dict(
    tree_type_correlation, orient='index', columns=["Cosine Similarity Ratio (X/Y)"] # Change the name if you need to
)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(tree_ratio_corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xlabel("Metric")
plt.ylabel("Tree Type")
plt.title("Heatmap: Correlation of Tree Type with Cosine Similarity Ratio (selected)")
plt.show()

# 4. DIVERGING BAR CHART FOR CORRELATIONS
# Sort DataFrame and define colors
tree_ratio_corr_df = tree_ratio_corr_df.sort_values("Cosine Similarity Ratio (X/Y)")    # Change the column if needed
colors = tree_ratio_corr_df["Cosine Similarity Ratio (X/Y)"].apply(lambda x: "red" if x > 0 else "blue")    # Same as above

# Plot diverging bar chart
plt.figure(figsize=(8, 6))
bars = plt.barh(
    tree_ratio_corr_df.index.astype(str),
    tree_ratio_corr_df["Cosine Similarity Ratio (X/Y)"],
    color=colors
)

# Annotate bars
for bar in bars:
    plt.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.2f}",
        ha="left" if bar.get_width() > 0 else "right",
        va="center",
        color="black",
        fontsize=12,
    )

plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Spearman Correlation with Cosine Similarity Ratio")
plt.ylabel("Tree Type")
plt.title("Diverging Bar Chart of Correlation of Tree Type with Cosine Similarity Ratio (random)")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.show()
