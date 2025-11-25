# Amazon Product Review Segmentation 🛍️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Library](https://img.shields.io/badge/Library-Gensim%20%7C%20Scikit--Learn-orange)

**An NLP pipeline comparing TF-IDF, Word2Vec, and Latent Dirichlet Allocation (LDA) to automatically categorize unstructured customer feedback.**

## 📖 Project Overview
In e-commerce, manually tagging thousands of customer reviews is inefficient and unscalable. This project develops an unsupervised machine learning solution to automatically discover latent topics within Amazon product reviews.

By benchmarking **10 different experimental setups**, I determined that probabilistic topic modeling (LDA) significantly outperforms traditional clustering methods (K-Means on TF-IDF) for short-text classification.

### Key Objectives
* **Structure the Unstructured:** Convert raw text into meaningful vector representations.
* **Benchmark Techniques:** Compare Sparse (TF-IDF), Dense (Word2Vec), and Probabilistic (LDA) approaches.
* **Visualize Insights:** Use t-SNE to project high-dimensional clusters into interpretable 2D space.

---

## 📊 Key Results
The study found that **Latent Dirichlet Allocation (LDA)** with 10 topics provided the distinct separation of product categories.

| Model | Technique | Silhouette Score | Performance |
| :--- | :--- | :--- | :--- |
| **Baseline** | TF-IDF + K-Means | 0.0115 | ❌ Poor Separation |
| **Intermediate** | Word2Vec + K-Means | 0.1424 | ⚠️ Moderate Overlap |
| **Winner** | **LDA Topic Modeling** | **0.4628** | ✅ **Distinct Clusters** |

### Visual Evidence
**1. Model Performance:**
![Model Comparison](images/model_performance_comparison.png)
*LDA outperformed the baseline by over 40x in terms of cluster cohesion.*

**2. Cluster Separation (t-SNE):**
![t-SNE Projection](images/tsne_lda_clusters.png)
*The final model successfully segmented reviews into "Books" (Cluster 0), "Movies" (Cluster 1), and "Music" (Cluster 2).*

---

## 🛠️ Methodology
The project follows a standard data science lifecycle:

1.  **Data Ingestion:** Loaded ~36,000 Amazon product reviews (FastText format).
2.  **Preprocessing:**
    * Regex cleaning (punctuation/special characters).
    * Tokenization & Lowercasing.
    * **Stopword Removal:** Custom list to remove high-frequency noise.
    * **Lemmatization:** Converting words to their base root (e.g., "running" -> "run").
3.  **Feature Extraction & Modeling:**
    * *Experiment 1-4:* TF-IDF Vectorization with K-Means.
    * *Experiment 5-6:* Word2Vec Embeddings with K-Means.
    * *Experiment 7-10:* LDA (Latent Dirichlet Allocation) with varying topic counts (k=5, 7, 10).
4.  **Evaluation:** Used **Silhouette Coefficient** to measure cluster density and separation.

---
