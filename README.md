# Neighborhood-Based Clustering

Python-based implementation of a **Neighborhood-Based Clustering algorithm** for mixed numerical and nominal data, developed as a project for the **Data Mining** course at Warsaw University of Technology.

## Overview

Clustering real-world data is challenging when datasets contain both **numerical and nominal attributes**. Traditional algorithms such as k-means are primarily designed for numerical data and require preprocessing or encoding for nominal attributes.

This project implements a **Neighborhood-Based Clustering (NBC)** algorithm that relies on *local neighborhood relationships* rather than global distance thresholds. By combining k-nearest neighbors (kNN), reverse k-nearest neighbors (R-kNN), and a **Neighborhood-based Density Factor (NDF)**, the algorithm is able to detect clusters of arbitrary shape and varying density in mixed-type datasets. It identifies points as types: Dense Points (DP), Even Points (EP), Sparse Points (SP).

## Algorithm Workflow

1. **Data Loading & Preprocessing**

   * Load dataset
   * Separate numerical and nominal attributes

2. **Distance Computation**

   * Numerical attributes: normalized Euclidean distance
   * Nominal attributes: match-based distance
   * Combined mixed-type distance

3. **Neighborhood Construction**

   * Compute k-nearest neighbors (kNB)
   * Compute reverse k-nearest neighbors (R-kNB)
   * Calculate Neighborhood-based Density Factor (NDF)
   * Classify points as DP, EP, or SP

4. **Clustering**

   * Expand clusters from DP and EP points using ND-reachability
   * Assign cluster labels
   * Treat remaining SP points as potential noise

5. **Evaluation**

   * Silhouette coefficient
   * Adjusted Rand Index
   * Cluster visualization
   * Similarity heatmaps

## Datasets

Experiments are conducted on publicly available mixed-type datasets, including:

* **Bank Marketing** – client subscription prediction
* **Flags** – country classification based on flag attributes
* **Wine Quality** – wine quality prediction using physicochemical tests

Datasets can be obtained from the **UCI Machine Learning Repository**.

## Running the Project

### Requirements

* Python 3.9+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Running

Parameters such as the number of neighbors `k` can be adjusted to analyze their impact on clustering performance.

Example workflows for different datasets:

```bash
python src/main.py --dataset data/bank_marketing.csv --k 20 --limit
```

```bash
python src/main.py --dataset data/flags.csv --k 8
```

```bash
python src/main.py --dataset data/wine_quality.csv --k 50 --limit 2000
```

Parameters such as the number of rows `limit` can be adjusted to test using smaller dataset, because of computational resources.

```bash
python src/main.py --dataset data/bank_marketing.csv --k 10 --limit 2000
```
