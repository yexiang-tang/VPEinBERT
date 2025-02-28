# This program visualizes metrics. 

import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import shapiro
import scipy.stats as stats
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# Load the datasets
selected_group = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\metrics_selected.CSV"
random_group = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\metrics_random.CSV"
df_selected = pd.read_csv(selected_group)
df_random = pd.read_csv(random_group)

# Filter the data for a tree type if needed. Feel free to comment out this part if there is no need to filter
df_tree_selected = df_selected[df_selected["Tree Type"] == 4]
df_tree_random = df_random[df_random["Tree Type"] == 4]

# Extract the column, rename the column based on need
ell_full = pd.to_numeric(
    df_selected["Cosine Similarity (Elliptical vs Full)"],
    errors='coerce'
    ).dropna()

ell_contrast = pd.to_numeric(
    df_selected["Cosine Similarity (Elliptical vs Contrast)"],
    errors='coerce'
    ).dropna()

cos_ratio_selected = pd.to_numeric(
    df_selected["Cosine Similarity Ratio (X/Y)"],
    errors='coerce'
    ).dropna()

cos_ratio_random = pd.to_numeric(
    df_random["Cosine Similarity Ratio (X/Y)"],
    errors='coerce'
    ).dropna()

# Test for normality
shapiro_ell_full = shapiro(ell_full)
shapiro_ell_contrast = shapiro(ell_contrast)

print(f"Shapiro-Wilk test (ell vs. full): p-value = {shapiro_ell_full.pvalue:.5f}")
print(f"Shapiro-Wilk test (ell vs. contrast): p-value = {shapiro_ell_contrast.pvalue:.5f}")

# Apply K-S Test and print the result
ks_stat, p_value_ks = ks_2samp(ell_full, ell_contrast)

print(f"K-S Statistic: {ks_stat:.3f}")
print(f"P-value: {p_value_ks:.10f}")

# Calculate the means and median
mean_ell_full = np.mean(ell_full)
mean_ell_contrast = np.mean(ell_contrast)
mean_cos_ratio_selected = np.mean(cos_ratio_selected)
mean_cos_ratio_random = np.mean(cos_ratio_random)

median_ell_full = np.median(ell_full)
median_ell_contrast = np.median(ell_contrast)
median_cos_ratio_selected = np.median(cos_ratio_selected)
median_cos_ratio_random= np.median(cos_ratio_random)

# Histogram if there is a pair of dataset
plt.figure(figsize=(10, 6))
sns.histplot(ell_full, bins=200, color='blue', alpha=0.6, label=f'ell vs. full (mean={mean_ell_full:.2f}\nmedian = {median_ell_full:.2f})', kde=True)
sns.histplot(ell_contrast, bins=200, color='red', alpha=0.6, label=f'ell vs. contrast (mean={mean_ell_contrast:.2f}\nmedian = {median_ell_contrast:.2f})', kde=True)

# Print the statistics on the plot
plt.text(0.95, 0.95, f'K-S stat: {ks_stat:.3f}\np-value: {p_value_ks:.5f}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.xlim(0.8, 1) # scale for similarity hist
# plt.xlim(0, 20) # scale for distance hist

plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Distribution of Cosine Similarity of Negation Trees (Type-2) (random)")
plt.legend()
plt.grid(True)
plt.show()

# Box plot 
data_boxplot = pd.DataFrame({
    'Cosine Similarity': np.concatenate([ell_full, ell_contrast]),
    'Group': ['Elliptical vs. Full'] * len(ell_full) + ['Elliptical vs. Contrast'] * len(ell_contrast)
})

plt.figure(figsize=(8, 6))
sns.boxplot(x='Group', y='Cosine Similarity', data=data_boxplot, palette=['blue', 'red'])

# plt.ylim(0.95, 1.05) # scale for ratio box plot
# plt.ylim(0, 6)  # scale for distance box plot
plt.ylim(0.95, 1) # scale for similarity box plot


plt.title("Comparison of Cosine Similarity Distributions of Negation tree (type-2) (selected)")
plt.xlabel("Group")
plt.ylabel("Cosine Similarity")
plt.grid(True)

plt.show()


# Histrogram of ratio
plt.figure(figsize=(10, 6))
sns.histplot(cos_ratio_random, bins=1500, color='blue', alpha=0.6, label=f'mean={mean_cos_ratio_random:.5f}\nmedian={median_cos_ratio_random:.5f}', kde=True)

plt.xlim(0.5, 1.5)

plt.axvline(x=1, color='red', linestyle='-', linewidth=2.5, alpha=0.8)

plt.text(1.1, max(plt.gca().get_ylim()) * 0.5, 
         "Denser distribution on the right\n"
        "implies a stronger sign\n"
        "for silenced structure detection",
         fontsize=10, color="darkred", bbox=dict(facecolor='white', alpha=0.7))

plt.text(0.6, max(plt.gca().get_ylim()) * 0.5, 
         "Denser distribution on the left\n"
          "implies a weaker sign\n"
          "for silenced structure detection",
         fontsize=10, color="darkred", bbox=dict(facecolor='white', alpha=0.7))

# Print the statistics on the plot
plt.xlabel("Ratio of Cosine Similarity ")
plt.ylabel("Density")
plt.title("Distribution of Ratio of Cosine Similarity (random)" )
plt.legend()
plt.grid(True)
plt.show()


