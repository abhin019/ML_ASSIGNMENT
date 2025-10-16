# Horror Story Recommendation System

A machine learning-based recommendation system for horror stories using hybrid approaches combining content-based filtering, gradient boosting, and support vector regression.

## Overview

This system recommends horror stories by:
- Analyzing story content using TF-IDF vectorization
- Using ML models (Gradient Boosting, SVR) for quality prediction
- Combining multiple approaches in hybrid systems

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
python dataset_validation.py
```

**Train Models:**
```bash
python recommendation_system.py
```

**Interactive Recommendations:**
- Enter story title: `Mirror Image`
- Or description: `scary haunted house with ghosts`
- Type `exit` to quit

## Models

1. **Baseline**: Pure content-based (TF-IDF + cosine similarity)
2. **ML Model 1**: Gradient Boosting Ranker (pairwise ranking)
3. **ML Model 2**: Support Vector Regression (rating prediction)
4. **Hybrid 1**: Content (60%) + GB (40%)
5. **Hybrid 2**: Content (50%) + SVR (30%) + Rating patterns (20%)

## Results

**Dataset**: 3,485 horror stories with 4,563 combined features (2,000 TF-IDF + 2,481 tags + 82 categories)

| System | Precision@5 | NDCG@5 |
|--------|-------------|---------|
| Content-Based (Baseline) | 0.6410 | 0.8837 |
| ML Model 1: Gradient Boosting | 1.0000 | 0.9520 |
| **ML Model 2: SVR** | **1.0000** | **0.9901** |
| Hybrid 1: Content + GB | 0.8230 | 0.9366 |
| Hybrid 2: Content + SVR + Rating | 0.9560 | 0.9653 |

**Best Model: Support Vector Regression**
- Initial Precision@5: 1.0000 | NDCG@5: 0.9901

**5-Fold Cross-Validation Results:**
- Precision@5: 0.6822 ± 0.0191 ✓ Excellent stability
- NDCG@5: 0.8914 ± 0.0055

## Dataset Requirements

Excel file (`creepypastas.xlsx`) with columns:
- `story_name`, `body`, `average_rating`
- `tags`, `categories`, `estimated_reading_time`

## Project Structure

```
horror-story-recommender/
├── dataset_validation.py
├── recommendation_system.py
├── creepypastas.xlsx
└── README.md
```

## Technical Details

- **Features**: 0.7×TF-IDF + 0.2×Tags + 0.1×Categories
- **TF-IDF**: 2000 features, bigrams, min_df=3, max_df=0.85
- **Evaluation**: 80/20 train-test split, 5-fold CV
- **Threshold**: Rating ≥8.0 = relevant

## License

MIT License

## Contact

Open an issue on GitHub for questions or feedback.

---

*Educational project demonstrating recommendation system techniques.*
