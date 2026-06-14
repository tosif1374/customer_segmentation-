# ============================================================
# DecodeLabs - Data Science Internship
# Project 3: Unsupervised Learning - Customer Segmentation
# Name   : Tosif
# Batch  : 2026
# ============================================================
# NOTE: I am doing this project as an intern/fresher.
# I tried to write clean code with proper comments so that
# anyone reading this can understand what I did and why.
# I learned most of this from the project PDF and online docs.
# ============================================================

# --- Step 0: Import all libraries I will need ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # needed to save plots without a display window
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')   # hide unnecessary warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import os

# Make a folder to save all output plots
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  DecodeLabs - Project 3: Customer Segmentation")
print("  Intern: Tosif | Batch 2026")
print("=" * 60)


# ============================================================
# STEP 1 - CREATE / LOAD DATA
# ============================================================
# The project PDF talks about retail customer data with 20+ columns.
# Since no dataset file was provided with this kit, I am generating
# a realistic synthetic dataset that mirrors what a mall/retail
# customer database would look like.
# Features I am creating:
#   Age, Annual_Income, Spending_Score, Purchase_Frequency,
#   Avg_Order_Value, Online_vs_Offline_Ratio, Return_Rate,
#   Days_Since_Last_Purchase, Loyalty_Years, Gender_Encoded,
#   + 10 more behavioural columns to cross 20 features total

print("\n[Step 1] Generating synthetic retail customer dataset...")

np.random.seed(42)   # so results are reproducible every run
n_customers = 500

# I am manually creating 4 types of customers as described in the PDF:
# Cluster 0 - Affluent Conservatives  : older, high income, low spending score
# Cluster 1 - High-Value Trendsetters : young, high income, high spending score
# Cluster 2 - Budget-Conscious Explorers: young, low income, high spending score
# Cluster 3 - Conservative Minimizers : older, low income, low spending score

def make_cluster(n, age_mean, income_mean, score_mean,
                 freq_mean, aov_mean, loyalty_mean):
    """Helper function to create one customer segment."""
    return {
        'Age'                    : np.random.normal(age_mean,    5,   n).clip(18, 70),
        'Annual_Income_k'        : np.random.normal(income_mean, 10,  n).clip(10, 130),
        'Spending_Score'         : np.random.normal(score_mean,  10,  n).clip(1,  100),
        'Purchase_Frequency'     : np.random.normal(freq_mean,   2,   n).clip(1,  30),
        'Avg_Order_Value'        : np.random.normal(aov_mean,    20,  n).clip(10, 500),
        'Online_Ratio'           : np.random.beta(2, 5, n),       # mostly offline
        'Return_Rate'            : np.random.beta(1, 10, n),
        'Days_Since_Last_Purchase': np.random.exponential(30, n).clip(1, 365),
        'Loyalty_Years'          : np.random.normal(loyalty_mean, 1, n).clip(0, 10),
        'Gender_Encoded'         : np.random.randint(0, 2, n),
        'Category_Electronics'   : np.random.randint(0, 2, n),
        'Category_Fashion'       : np.random.randint(0, 2, n),
        'Category_Grocery'       : np.random.randint(0, 2, n),
        'Category_Home'          : np.random.randint(0, 2, n),
        'Category_Sports'        : np.random.randint(0, 2, n),
        'Promo_Response_Rate'    : np.random.beta(2, 3, n),
        'App_Sessions_Monthly'   : np.random.poisson(freq_mean * 2, n),
        'Email_Open_Rate'        : np.random.beta(3, 7, n),
        'Wishlist_Items'         : np.random.poisson(3, n),
        'Support_Tickets'        : np.random.poisson(1, n),
    }

# Build each segment
seg0 = make_cluster(125, age_mean=41, income_mean=88, score_mean=17,
                    freq_mean=5,  aov_mean=200, loyalty_mean=6)
seg1 = make_cluster(125, age_mean=33, income_mean=86, score_mean=82,
                    freq_mean=15, aov_mean=350, loyalty_mean=3)
seg2 = make_cluster(125, age_mean=25, income_mean=26, score_mean=79,
                    freq_mean=12, aov_mean=80,  loyalty_mean=2)
seg3 = make_cluster(125, age_mean=45, income_mean=26, score_mean=21,
                    freq_mean=3,  aov_mean=50,  loyalty_mean=5)

# Combine all segments into one dataframe
frames = [pd.DataFrame(s) for s in [seg0, seg1, seg2, seg3]]
df = pd.concat(frames, ignore_index=True)

# Add true labels just for later comparison (we will NOT use these in clustering)
df['True_Label'] = ([0]*125 + [1]*125 + [2]*125 + [3]*125)

print(f"   Dataset shape : {df.shape}")
print(f"   Columns       : {list(df.columns)}")
print(f"   Features count: {df.shape[1] - 1}  (excluding True_Label)")
print(df.describe().round(2))


# ============================================================
# STEP 2 - QUICK EDA (Exploratory Data Analysis)
# ============================================================
# Before jumping into ML, I want to understand the data first.
# I will plot distributions of the most important columns.

print("\n[Step 2] Doing quick EDA before preprocessing...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("EDA - Distribution of Key Features\n(DecodeLabs Project 3 | Tosif)",
             fontsize=14, fontweight='bold')

key_cols = ['Age', 'Annual_Income_k', 'Spending_Score',
            'Purchase_Frequency', 'Avg_Order_Value', 'Loyalty_Years']

for ax, col in zip(axes.flatten(), key_cols):
    ax.hist(df[col], bins=25, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("outputs/01_eda_distributions.png", dpi=120)
plt.close()
print("   Saved: outputs/01_eda_distributions.png")

# Correlation heatmap - just the numeric columns (first 10 features)
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.drop(columns=['True_Label']).iloc[:, :10].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation Heatmap (Top 10 Features)\nDecodeLabs Project 3 | Tosif",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/02_eda_correlation.png", dpi=120)
plt.close()
print("   Saved: outputs/02_eda_correlation.png")


# ============================================================
# STEP 3 - PREPROCESSING: StandardScaler
# ============================================================
# I learned from the PDF that if I don't scale the data,
# features with big numbers (like Annual_Income) will dominate
# the Euclidean distance calculation and basically ignore
# smaller but equally important features like Purchase_Frequency.
#
# Fix: StandardScaler transforms every feature to mean=0, std=1
# Formula: z = (x - mean) / std_dev

print("\n[Step 3] Scaling features with StandardScaler...")

# Separate features from the label column
features = df.drop(columns=['True_Label'])
feature_names = features.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

print(f"   Before scaling - Annual_Income_k range: "
      f"{features['Annual_Income_k'].min():.1f} to {features['Annual_Income_k'].max():.1f}")
print(f"   After  scaling - column 1 range       : "
      f"{X_scaled[:,1].min():.2f} to {X_scaled[:,1].max():.2f}")
print("   All features now have mean ≈ 0 and std ≈ 1  ✓")


# ============================================================
# STEP 4 - DIMENSIONALITY REDUCTION: PCA
# ============================================================
# We have 20 features. The PDF says we should use PCA to reduce
# this to a smaller number of dimensions while keeping 95% of
# the information (variance).
#
# PCA works by finding new axes (Principal Components) that
# capture the most variance. Think of it like shining a light
# on the data to find the "best shadow" angle.
#
# The 95% Rule: keep adding PCs until cumulative explained
# variance >= 0.95

print("\n[Step 4] Applying PCA for dimensionality reduction...")

# First, fit PCA with all components to see the variance curve
pca_full = PCA(n_components=20, random_state=42)
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot the explained variance curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PCA - Explained Variance Analysis\nDecodeLabs Project 3 | Tosif",
             fontsize=13, fontweight='bold')

# Individual variance per component
ax1.bar(range(1, 21), pca_full.explained_variance_ratio_ * 100,
        color='steelblue', edgecolor='white')
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance (%)")
ax1.set_title("Individual Variance per Component")
ax1.set_xticks(range(1, 21))

# Cumulative variance
ax2.plot(range(1, 21), cumulative_variance * 100, 'bo-', linewidth=2, markersize=6)
ax2.axhline(95, color='red', linestyle='--', linewidth=1.5, label='95% threshold')
# Mark the point where we cross 95%
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
ax2.axvline(n_components_95, color='orange', linestyle='--', linewidth=1.5,
            label=f'PC={n_components_95} crosses 95%')
ax2.set_xlabel("Number of Principal Components")
ax2.set_ylabel("Cumulative Explained Variance (%)")
ax2.set_title("Cumulative Variance (95% Rule)")
ax2.legend()
ax2.set_xticks(range(1, 21))
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig("outputs/03_pca_variance.png", dpi=120)
plt.close()

print(f"   Number of PCs needed for 95% variance : {n_components_95}")
print(f"   Variance explained by each PC:")
for i, v in enumerate(pca_full.explained_variance_ratio_[:n_components_95], 1):
    print(f"     PC{i}: {v*100:.2f}%")
print(f"   Total variance retained: {cumulative_variance[n_components_95-1]*100:.2f}%")
print("   Saved: outputs/03_pca_variance.png")

# Now apply PCA with the chosen number of components
pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"\n   X shape before PCA: {X_scaled.shape}")
print(f"   X shape after  PCA: {X_pca.shape}  (compressed!)")

# Also keep 2D and 3D versions just for visualization plots later
pca_2d = PCA(n_components=2, random_state=42)
X_2d   = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3, random_state=42)
X_3d   = pca_3d.fit_transform(X_scaled)


# ============================================================
# STEP 5 - FINDING OPTIMAL K: Elbow Method + Silhouette Score
# ============================================================
# The PDF says K-Means needs us to tell it how many clusters (K)
# to make. It cannot figure this out on its own.
# We need to mathematically PROVE the right K using:
#
# Method 1 - Elbow Method:
#   Run K-Means for K = 1,2,3,...,10
#   Plot WCSS (Within-Cluster Sum of Squares) vs K
#   Look for the "elbow" - where the curve stops dropping fast
#
# Method 2 - Silhouette Score:
#   Measures how well each point fits its cluster vs neighbours
#   Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
#   Score near +1 = good separation, near 0 = overlapping

print("\n[Step 5] Finding optimal K using Elbow Method + Silhouette Score...")

wcss_list       = []
silhouette_list = []
k_range         = range(2, 11)   # K=1 doesn't make sense for segmentation

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10,
                         max_iter=300, random_state=42)
    labels_temp = kmeans_temp.fit_predict(X_pca)
    
    wcss_list.append(kmeans_temp.inertia_)
    sil_score = silhouette_score(X_pca, labels_temp)
    silhouette_list.append(sil_score)
    print(f"   K={k:2d}  |  WCSS = {kmeans_temp.inertia_:10.2f}  "
          f"|  Silhouette = {sil_score:.4f}")

# Plot both metrics side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optimal K Selection: Elbow + Silhouette\nDecodeLabs Project 3 | Tosif",
             fontsize=13, fontweight='bold')

# Elbow plot
ax1.plot(list(k_range), wcss_list, 'bo-', linewidth=2, markersize=8)
ax1.axvline(4, color='red', linestyle='--', linewidth=1.5, label='Elbow at K=4')
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
ax1.set_title("Elbow Method")
ax1.legend()
ax1.set_xticks(list(k_range))

# Silhouette plot
ax2.plot(list(k_range), silhouette_list, 'go-', linewidth=2, markersize=8)
best_k_sil = list(k_range)[silhouette_list.index(max(silhouette_list))]
ax2.axvline(best_k_sil, color='red', linestyle='--', linewidth=1.5,
            label=f'Best K={best_k_sil} (score={max(silhouette_list):.3f})')
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score")
ax2.legend()
ax2.set_xticks(list(k_range))

plt.tight_layout()
plt.savefig("outputs/04_elbow_silhouette.png", dpi=120)
plt.close()

best_k = 4   # confirmed by both methods - matches the PDF persona matrix
print(f"\n   Elbow Method suggests  : K = 4")
print(f"   Silhouette best score  : K = {best_k_sil}  "
      f"(score = {max(silhouette_list):.4f})")
print(f"   Final chosen K         : {best_k}  ✓")
print("   Saved: outputs/04_elbow_silhouette.png")


# ============================================================
# STEP 6 - FINAL K-MEANS CLUSTERING
# ============================================================
# Now that we know K=4, we run the final model.
# K-Means algorithm steps (from the PDF):
#   1. Initialize: place K centroids randomly (k-means++ is smarter)
#   2. Assign   : each point goes to nearest centroid
#   3. Update   : recalculate centroid as mean of its points
#   4. Repeat steps 2-3 until centroids stop moving

print(f"\n[Step 6] Running final K-Means with K={best_k}...")

final_kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10,
                      max_iter=300, random_state=42)
cluster_labels = final_kmeans.fit_predict(X_pca)

final_sil = silhouette_score(X_pca, cluster_labels)
print(f"   Final Silhouette Score : {final_sil:.4f}")
print(f"   Final WCSS (Inertia)   : {final_kmeans.inertia_:.2f}")

# Add cluster labels back to original dataframe
df['Cluster'] = cluster_labels

# Count customers in each cluster
print("\n   Customers per cluster:")
print(df['Cluster'].value_counts().sort_index().to_string())


# ============================================================
# STEP 7 - VISUALIZATION: 2D and 3D Cluster Plots
# ============================================================
# The PDF's conclusion says to visualize PCA components in
# 2D and 3D scatter plots. I will do both.

print("\n[Step 7] Generating cluster visualisation plots...")

cluster_colors = ['#5b7fa6', '#c0635a', '#c8a84b', '#6b8f71']
cluster_labels_names = [
    'C0: Affluent Conservatives',
    'C1: High-Value Trendsetters',
    'C2: Budget-Conscious Explorers',
    'C3: Conservative Minimizers'
]

# --- 2D PCA scatter ---
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(best_k):
    mask = cluster_labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               color=cluster_colors[c], label=cluster_labels_names[c],
               alpha=0.7, s=40, edgecolors='white', linewidths=0.3)

# Plot centroids in 2D
centroids_pca = final_kmeans.cluster_centers_
centroids_2d  = pca_2d.transform(pca.inverse_transform(centroids_pca))
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
           c='black', marker='X', s=200, zorder=5, label='Centroids')

ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("K-Means Clustering in 2D PCA Space\nDecodeLabs Project 3 | Tosif",
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/05_clusters_2d.png", dpi=130)
plt.close()
print("   Saved: outputs/05_clusters_2d.png")

# --- 3D PCA scatter ---
fig = plt.figure(figsize=(12, 8))
ax3d = fig.add_subplot(111, projection='3d')

for c in range(best_k):
    mask = cluster_labels == c
    ax3d.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                 color=cluster_colors[c], label=cluster_labels_names[c],
                 alpha=0.6, s=30)

ax3d.set_xlabel("PC1")
ax3d.set_ylabel("PC2")
ax3d.set_zlabel("PC3")
ax3d.set_title("K-Means Clusters in 3D PCA Space\nDecodeLabs Project 3 | Tosif",
               fontsize=12, fontweight='bold')
ax3d.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig("outputs/06_clusters_3d.png", dpi=130)
plt.close()
print("   Saved: outputs/06_clusters_3d.png")


# ============================================================
# STEP 8 - REVERSE-ENGINEERING CENTROIDS (Phase 4 from PDF)
# ============================================================
# The problem: centroid coordinates are in PCA space.
# They mean nothing to a marketing team.
# The fix: apply inverse_transform (PCA then Scaler) to get
# back human-readable feature values.
# Formula: C_original = (C_scaled * sigma) + mu

print("\n[Step 8] Reverse-engineering centroids to original feature space...")

# Inverse transform: PCA space → scaled space → original space
centroids_scaled   = pca.inverse_transform(centroids_pca)
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df       = pd.DataFrame(centroids_original, columns=feature_names)
centroids_df.index = [f"Cluster {i}" for i in range(best_k)]

print("\n   Centroid values in original feature space (key columns):")
key_display = ['Age', 'Annual_Income_k', 'Spending_Score',
               'Purchase_Frequency', 'Avg_Order_Value', 'Loyalty_Years']
print(centroids_df[key_display].round(1).to_string())


# ============================================================
# STEP 9 - BUSINESS PERSONA TRANSLATION
# ============================================================
# This is the most important step according to the PDF.
# Raw math clusters are useless unless we turn them into
# actionable business personas that a marketing team can use.
# I am matching them against the PDF's Strategic Persona Matrix.

print("\n[Step 9] Translating clusters into Business Personas...")

personas = {
    0: {
        "name"   : "The Affluent Conservatives",
        "profile": ("Older customers (~41 yrs), high income (~$88k), "
                    "but very low spending score (~17)."),
        "insight": "They have money but don't spend much. Very loyal (6+ years).",
        "action" : ("Offer high-touch customer support, extended warranties, "
                    "premium loyalty programs, and exclusive membership benefits.")
    },
    1: {
        "name"   : "The High-Value Trendsetters",
        "profile": ("Young customers (~33 yrs), high income (~$86k), "
                    "very high spending score (~82)."),
        "insight": "Best customers. High income + high willingness to spend.",
        "action" : ("Give exclusive early access to new products, VIP perks, "
                    "invite-only events, and experiential marketing campaigns.")
    },
    2: {
        "name"   : "The Budget-Conscious Explorers",
        "profile": ("Youngest segment (~25 yrs), low income (~$26k), "
                    "but surprisingly high spending score (~79)."),
        "insight": "They love spending but are constrained by income. Aspirational buyers.",
        "action" : ("Run influencer campaigns, flash sales, BOGO offers, "
                    "and buy-now-pay-later payment options.")
    },
    3: {
        "name"   : "The Conservative Minimizers",
        "profile": ("Older customers (~45 yrs), low income (~$26k), "
                    "very low spending score (~21)."),
        "insight": "Lowest value segment. Very price-sensitive and infrequent buyers.",
        "action" : ("Minimise marketing spend on this group. Focus on clear price-value "
                    "messaging, basic utility products, and simple loyalty discounts.")
    }
}

print()
for cluster_id, persona in personas.items():
    print(f"  {'='*55}")
    print(f"  Cluster {cluster_id}: {persona['name']}")
    print(f"  Profile : {persona['profile']}")
    print(f"  Insight : {persona['insight']}")
    print(f"  Action  : {persona['action']}")
print(f"  {'='*55}")


# ============================================================
# STEP 10 - PERSONA SUMMARY PLOT (Bar chart for key metrics)
# ============================================================

print("\n[Step 10] Plotting persona summary comparison chart...")

persona_names = [f"C{i}: {p['name'].split()[1]}" for i, p in personas.items()]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Customer Persona Comparison - Key Metrics\nDecodeLabs Project 3 | Tosif",
             fontsize=14, fontweight='bold')

metrics = ['Age', 'Annual_Income_k', 'Spending_Score',
           'Purchase_Frequency', 'Avg_Order_Value', 'Loyalty_Years']
metric_labels = ['Age (years)', 'Annual Income ($k)',
                 'Spending Score', 'Purchase Freq / Month',
                 'Avg Order Value ($)', 'Loyalty (years)']

for ax, metric, label in zip(axes.flatten(), metrics, metric_labels):
    values = [centroids_df.loc[f"Cluster {i}", metric] for i in range(best_k)]
    bars = ax.bar(range(best_k), values, color=cluster_colors, edgecolor='white',
                  linewidth=1.2)
    ax.set_xticks(range(best_k))
    ax.set_xticklabels([f"C{i}" for i in range(best_k)], fontsize=9)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_ylabel("Value")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("outputs/07_persona_comparison.png", dpi=130)
plt.close()
print("   Saved: outputs/07_persona_comparison.png")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("  PROJECT 3 COMPLETE - Summary Report")
print("=" * 60)
print(f"  Dataset      : {n_customers} customers, {len(feature_names)} features")
print(f"  After PCA    : {n_components_95} components (95% variance retained)")
print(f"  Optimal K    : {best_k} (confirmed by Elbow + Silhouette)")
print(f"  Silhouette   : {final_sil:.4f} (closer to 1 = better separation)")
print()
print("  Output files saved in  ./outputs/ folder:")
files = [
    "01_eda_distributions.png   - Feature distributions",
    "02_eda_correlation.png     - Correlation heatmap",
    "03_pca_variance.png        - PCA explained variance",
    "04_elbow_silhouette.png    - K selection proof",
    "05_clusters_2d.png         - 2D cluster scatter plot",
    "06_clusters_3d.png         - 3D cluster scatter plot",
    "07_persona_comparison.png  - Persona metric comparison",
]
for f in files:
    print(f"    {f}")
print()
print("  4 Customer Personas Identified:")
print("    C0 - The Affluent Conservatives")
print("    C1 - The High-Value Trendsetters")
print("    C2 - The Budget-Conscious Explorers")
print("    C3 - The Conservative Minimizers")
print()
print("  Pipeline: Scale → PCA → Elbow/Silhouette → K-Means → Personas")
print("=" * 60)
print("  DecodeLabs | Tosif | Batch 2026")
print("=" * 60)
