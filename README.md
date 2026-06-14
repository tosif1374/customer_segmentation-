# DecodeLabs – Data Science Internship
## Project 3: Unsupervised Learning (Customer Segmentation)

**Intern:** Tosif  
**Batch:** 2026  
**Track:** Data Science  
**Powered by:** DecodeLabs  

---

## What is this project about?

This is Project 3 of my Data Science internship at DecodeLabs.  
The goal was to take raw, unlabeled retail customer data and use **Unsupervised Learning** to find hidden groups (segments) of customers — without anyone telling the model what the groups should be.

The final output is not just clusters — it is **Business Personas** that a marketing team can actually use to make decisions.

---

## The Pipeline (4 phases from the training kit)

```
Raw Data → SCALE → COMPRESS (PCA) → CLUSTER (K-Means) → TRANSLATE (Personas)
```

| Phase | Technique | Why |
|-------|-----------|-----|
| 1. Scale | StandardScaler | Without scaling, big-number features (like Income) dominate distance calculations and ignore smaller but equally important ones |
| 2. Compress | PCA (95% rule) | We had 20 features. High dimensions break Euclidean distance (Curse of Dimensionality). PCA squeezes 20 columns into ~15 components while keeping 95% of information |
| 3. Cluster | K-Means (K=4) | Groups customers by mathematical distance in PCA space. K was proven using Elbow Method + Silhouette Score |
| 4. Translate | Centroid inverse transform | Centroids live in PCA space — meaningless to a business. We reverse the transforms to get human-readable metrics (Age, Income, Score) |

---

## Project Structure

```
project3_customer_segmentation/
│
├── customer_segmentation.py      ← Main script (run this)
├── README.md                     ← You are reading this
│
└── outputs/                      ← All plots are saved here
    ├── 01_eda_distributions.png  ← Feature distribution histograms
    ├── 02_eda_correlation.png    ← Correlation heatmap
    ├── 03_pca_variance.png       ← Explained variance curve (95% rule)
    ├── 04_elbow_silhouette.png   ← K selection proof
    ├── 05_clusters_2d.png        ← 2D PCA cluster scatter plot
    ├── 06_clusters_3d.png        ← 3D PCA cluster scatter plot
    └── 07_persona_comparison.png ← Bar chart comparing all 4 personas
```

---

## How to Run

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Run the script

```bash
python customer_segmentation.py
```

That's it. The script will:
- Generate the dataset
- Do EDA
- Scale, apply PCA, find optimal K, cluster, and translate
- Save all 7 plots in the `outputs/` folder
- Print a full summary in the terminal

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Numerical operations |
| `pandas` | ≥ 1.5 | Data handling and DataFrames |
| `matplotlib` | ≥ 3.6 | Plotting |
| `seaborn` | ≥ 0.12 | Heatmaps and styled plots |
| `scikit-learn` | ≥ 1.2 | StandardScaler, PCA, KMeans, silhouette_score |

---

## Key Concepts I Used (and learned from the training kit)

### StandardScaler
Maps every feature to mean = 0, std = 1.  
Formula: `z = (x - μ) / σ`  
Without this, K-Means would be biased toward high-magnitude features.

### PCA — Principal Component Analysis
An unsupervised linear transformation that finds new axes (Principal Components) capturing the most variance.  
Think of it as shining a light at the best angle to cast the most informative shadow.  
Eigenvalue equation: `Σv = λv`  
I kept enough components to explain ≥ 95% cumulative variance.

### Elbow Method
Plots WCSS (Within-Cluster Sum of Squares) vs K.  
The "elbow" is where adding more clusters stops giving meaningful improvement.  
Formula: `WCSS = Σ ||x - μk||²`

### Silhouette Score
Measures how well a point fits its own cluster vs the nearest other cluster.  
Formula: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`  
Score near +1 → well separated. Score near 0 → overlapping.

### Centroid Inverse Transform (Phase 4)
Centroids exist in PCA space. To interpret them:  
`C_original = (C_scaled ⊙ σ) + μ`  
This maps abstract coordinates back to real features like Age and Income.

---

## Results

**Optimal K = 4** (confirmed by both Elbow Method and Silhouette Score)

### 4 Customer Personas Discovered

| Cluster | Persona Name | Age | Income | Spending Score | Key Action |
|---------|-------------|-----|--------|---------------|------------|
| C0 | Affluent Conservatives | ~41 | $88k | 17 (low) | Loyalty programs, warranties, high-touch support |
| C1 | High-Value Trendsetters | ~33 | $86k | 82 (high) | VIP perks, early access, experiential marketing |
| C2 | Budget-Conscious Explorers | ~25 | $26k | 79 (high) | Flash sales, BOGO, buy-now-pay-later |
| C3 | Conservative Minimizers | ~45 | $26k | 21 (low) | Clear value messaging, basic utility, minimal spend |

---

## Output Plots

| File | What it shows |
|------|--------------|
| `01_eda_distributions.png` | Histogram of 6 key features before any processing |
| `02_eda_correlation.png` | Correlation heatmap of first 10 features |
| `03_pca_variance.png` | How many PCs are needed to retain 95% variance |
| `04_elbow_silhouette.png` | Mathematical proof for choosing K=4 |
| `05_clusters_2d.png` | Final clusters visualised in 2D PCA space |
| `06_clusters_3d.png` | Final clusters visualised in 3D PCA space |
| `07_persona_comparison.png` | Side-by-side bar chart comparing all 4 personas |

---

## Note from the intern

> I am a final-year B.Tech CSE (AI & ML) student doing this as part of my DecodeLabs Data Science internship.  
> I tried to write the code as simply and clearly as possible so that anyone — technical or not — can follow along.  
> Every comment in the script explains not just *what* the code does but *why* that step is needed.  
> The concepts here (PCA, K-Means, Silhouette Score) were something I had studied theoretically before,  
> but this project helped me actually implement them end-to-end in a real pipeline — which is very different.

---

*DecodeLabs | Data Science | Batch 2026*
