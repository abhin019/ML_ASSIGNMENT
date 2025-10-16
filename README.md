# Horror Story Recommendation System

A machine learning-based recommendation system for horror stories using hybrid approaches combining content-based filtering, gradient boosting, and support vector regression.

## Overview

This system recommends horror stories by:
- Analyzing story content using TF-IDF vectorization
- Using ML models (Gradient Boosting, SVR) for quality prediction
- Combining multiple approaches in hybrid systems
- Achieving **79.7% precision** with Gradient Boosting (26.7% improvement over baseline)

## Features

- **Content-Based Filtering**: TF-IDF similarity matching
- **ML Models**: Gradient Boosting Ranker & Support Vector Regression
- **Hybrid Systems**: Combines content + ML predictions
- **Interactive Interface**: Real-time recommendations via command-line
- **Evaluation**: 5-fold cross-validation with Precision@5 and NDCG@5

## Installation

```bash
git clone https://github.com/yourusername/horror-story-recommender.git
cd horror-story-recommender
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

Place `creepypastas.xlsx` in the project directory.

## Usage

**Validate Dataset:**
```bash
python samp_assign.py
```

**Train Models:**
```bash
python assignment.py
```

**Interactive Recommendations:**
- Enter story title: `Mirror Image`
- Or description: `scary haunted house with ghosts`
- Type `exit` to quit

## Models

1. **Baseline**: Pure content-based (TF-IDF + cosine similarity)
2. **ML Model 1**: Gradient Boosting Ranker (pairwise ranking)
3. **ML Model 2**: Support Vector Regression (rating prediction)
4. **Hybrid 1**: Content + GB (gentler weighting)
5. **Hybrid 2**: Content + SVR (conservative weighting)

## Results

**Dataset**: 3,485 horror stories with 4,563 combined features (2,000 TF-IDF + 2,481 tags + 82 categories)

| System | Precision@5 | NDCG@5 |
|--------|-------------|---------|
| Content-Based (Baseline) | 0.6290 | 0.8847 |
| **ML Model 1: Gradient Boosting** | **0.7970** | **0.9268** |
| ML Model 2: Support Vector Regression | 0.7870 | 0.9185 |
| Hybrid 1: Content + GB | 0.7620 | 0.9085 |
| Hybrid 2: Content + SVR Enhanced | 0.7370 | 0.9108 |

**Best Model: Gradient Boosting Quality Weights**
- Test Precision@5: 0.7970 (79.7%)
- Test NDCG@5: 0.9268 (92.68%)
- Improvement: +26.7% over baseline

**5-Fold Cross-Validation Results:**
- Precision@5: 0.6650 ± 0.0147
- NDCG@5: 0.8938 ± 0.0016

## Dataset Requirements

Excel file (`creepypastas.xlsx`) with columns:
- `story_name`, `body`, `average_rating`
- `tags`, `categories`, `estimated_reading_time`,`publish_date`

## Project Structure

```
horror-story-recommender/
├── samp_assign.py      # Data quality check
├── assignment.py   # Main recommendation system
├── creepypastas.xlsx          # Dataset (not included)
└── README.md                  # This file
```

## Technical Details

- **Features**: 0.7×TF-IDF + 0.2×Tags + 0.1×Categories
- **TF-IDF**: 2000 features, bigrams, min_df=3, max_df=0.85
- **Evaluation**: 80/20 train-test split, 5-fold CV
- **Threshold**: Rating ≥8.0 = relevant


## About `samp_assign.py`

**Purpose:** Pre-flight data quality check before training.

**What it does:** Validates dataset structure, analyzes ratings/text/metadata, checks recommendation system feasibility.

**Runtime:** ~5 seconds

**Output:** Dataset stats (3,485 stories, mean rating 7.57), metadata analysis (2,481 tags, 82 categories), suitability assessment.

