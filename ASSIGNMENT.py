import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import ndcg_score, mean_squared_error
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

print("=" * 110)
print("ML MODELS + HYBRID SYSTEMS FOR STORY RECOMMENDATION (FIXED)")
print("=" * 110)

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================

print("\nSTEP 1: DATA LOADING AND PREPROCESSING")
print("-" * 110)

df = pd.read_excel('creepypastas.xlsx')
data = df.copy()

# Clean
data['body'] = data['body'].fillna('').str.replace('\n', ' ').str.strip()
data = data[data['body'].str.len() > 100].reset_index(drop=True)
data['reading_time_mins'] = data['estimated_reading_time'].str.extract(r'(\d+)').astype(float)
data['reading_time_mins'] = data['reading_time_mins'].fillna(data['reading_time_mins'].median())
data['rating'] = data['average_rating']
data['tags'] = data['tags'].fillna('').apply(
    lambda x: [tag.strip().lower() for tag in str(x).split(',') if tag.strip()]
)
data['categories'] = data['categories'].fillna('').apply(
    lambda x: [cat.strip().lower() for cat in str(x).split(',') if cat.strip()]
)

print(f"Stories: {len(data)} | Rating mean: {data['rating'].mean():.2f}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print("\nSTEP 2: FEATURE ENGINEERING")
print("-" * 110)

# TF-IDF
tfidf_vec = TfidfVectorizer(stop_words='english', max_features=2000, min_df=3, max_df=0.85, ngram_range=(1,2), sublinear_tf=True)
tfidf_matrix = tfidf_vec.fit_transform(data['body'])

# Tags
all_tags = sorted(list(set([t for tags in data['tags'] for t in tags])))
tag_matrix = np.zeros((len(data), len(all_tags)))
for i, tags in enumerate(data['tags']):
    for tag in tags:
        if tag in all_tags:
            tag_matrix[i, all_tags.index(tag)] = 1

# Categories
all_cats = sorted(list(set([c for cats in data['categories'] for c in cats])))
cat_matrix = np.zeros((len(data), len(all_cats)))
for i, cats in enumerate(data['categories']):
    for cat in cats:
        if cat in all_cats:
            cat_matrix[i, all_cats.index(cat)] = 1

from scipy.sparse import hstack, csr_matrix
content_features = hstack([tfidf_matrix * 0.7, csr_matrix(tag_matrix) * 0.2, csr_matrix(cat_matrix) * 0.1])

print(f"TF-IDF: {tfidf_matrix.shape} | Tags: {tag_matrix.shape} | Categories: {cat_matrix.shape}")
print(f"Combined features: {content_features.shape}")

# ============================================================================
# STEP 3: FIXED EVALUATION FUNCTION
# ============================================================================

def evaluate_recommendations(train_idx, test_idx, quality_weights=None, k=5, threshold=8.0):
    """
    FIXED: Direct test-to-train similarity evaluation
    Computes similarity from each test story directly to training stories
    """
    precisions = []
    ndcgs = []
    
    train_features = content_features[train_idx]
    
    # Sample test set for reasonable runtime
    test_sample = test_idx[:min(200, len(test_idx))]
    
    for test_original_idx in test_sample:
        # Compute similarity from this test story to all training stories
        test_feature = content_features[test_original_idx]
        similarities = cosine_similarity(test_feature, train_features)[0]
        
        # Apply quality weighting if provided
        if quality_weights is not None:
            similarities = similarities * quality_weights
        
        # Get top-k recommendations
        top_k_positions = np.argsort(similarities)[::-1][:k]
        rec_indices = [train_idx[pos] for pos in top_k_positions]
        rec_ratings = data['rating'].iloc[rec_indices].values
        
        # Precision: fraction of recommendations above threshold
        relevant = (rec_ratings >= threshold).sum()
        precisions.append(relevant / k)
        
        # NDCG
        ideal = np.sort(rec_ratings)[::-1]
        dcg = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(rec_ratings))
        idcg = sum((2**ideal_r - 1) / np.log2(i+2) for i, ideal_r in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    return {
        'precision': np.mean(precisions), 
        'ndcg': np.mean(ndcgs),
        'std_precision': np.std(precisions),
        'std_ndcg': np.std(ndcgs)
    }

# ============================================================================
# STEP 4: BASELINE - PURE CONTENT-BASED
# ============================================================================

print("\n\nSTEP 3: BASELINE - PURE CONTENT-BASED SIMILARITY")
print("-" * 110)

train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)

baseline_result = evaluate_recommendations(train_idx, test_idx, quality_weights=None)
print(f"Content-Based Baseline - Precision@5: {baseline_result['precision']:.4f} | NDCG@5: {baseline_result['ndcg']:.4f}")

# ============================================================================
# STEP 5: ML MODEL 1 - GRADIENT BOOSTING FOR QUALITY PREDICTION
# ============================================================================

print("\n\nSTEP 4: ML MODEL 1 - GRADIENT BOOSTING QUALITY PREDICTOR")
print("-" * 110)

def create_pairwise_data(X, y, n_pairs=5000):
    """Create pairwise training data for ranking"""
    X_pairs = []
    y_pairs = []
    
    n_samples = min(len(y), 1000)
    indices = np.random.choice(len(y), n_samples, replace=False)
    
    for _ in range(n_pairs):
        i, j = np.random.choice(indices, 2, replace=False)
        if y[i] != y[j]:
            diff = X[i].toarray() - X[j].toarray()
            X_pairs.append(diff[0])
            y_pairs.append(1 if y[i] > y[j] else 0)
    
    return np.array(X_pairs), np.array(y_pairs)

print("Training Gradient Boosting Ranker...")
X_pairs_train, y_pairs_train = create_pairwise_data(
    content_features[train_idx], 
    data['rating'].iloc[train_idx].values
)

gb_ranker = GradientBoostingRegressor(
    n_estimators=50, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)
gb_ranker.fit(X_pairs_train, y_pairs_train)

# FIXED: Use GB to predict quality scores, then weight content similarity
gb_scores = gb_ranker.predict(content_features[train_idx].toarray())
gb_weights = (gb_scores - gb_scores.min()) / (gb_scores.max() - gb_scores.min() + 1e-8)

# Boost high-quality stories
gb_quality_weights = np.power(gb_weights, 0.5)  # Square root to moderate the effect

gb_result = evaluate_recommendations(train_idx, test_idx, quality_weights=gb_quality_weights)
print(f"GB Quality-Weighted - Precision@5: {gb_result['precision']:.4f} | NDCG@5: {gb_result['ndcg']:.4f}")

# ============================================================================
# STEP 6: ML MODEL 2 - SUPPORT VECTOR REGRESSION
# ============================================================================

print("\nSTEP 5: ML MODEL 2 - SUPPORT VECTOR REGRESSION (SVR)")
print("-" * 110)

print("Training SVR for rating prediction...")
# Use subset for SVR (memory constraints)
sample_size = min(2000, len(train_idx))
sample_train_idx = np.random.choice(train_idx, sample_size, replace=False)

X_svr = content_features[sample_train_idx].toarray()
y_svr = data['rating'].iloc[sample_train_idx].values

svr_model = SVR(kernel='rbf', C=10, gamma='scale')
svr_model.fit(X_svr, y_svr)

# FIXED: Predict quality for all training stories and use as weights
svr_scores = svr_model.predict(content_features[train_idx].toarray())
svr_weights = (svr_scores - svr_scores.min()) / (svr_scores.max() - svr_scores.min() + 1e-8)

# Boost high-quality stories
svr_quality_weights = np.power(svr_weights, 0.5)

svr_result = evaluate_recommendations(train_idx, test_idx, quality_weights=svr_quality_weights)
print(f"SVR Quality-Weighted - Precision@5: {svr_result['precision']:.4f} | NDCG@5: {svr_result['ndcg']:.4f}")

# ============================================================================
# STEP 7: HYBRID 1 - CONTENT + GB RANKER
# ============================================================================

print("\n\nSTEP 6: HYBRID 1 - CONTENT-BASED + GRADIENT BOOSTING")
print("-" * 110)

# Combine: content similarity weighted by GB quality predictions
# Use stronger weighting for high-quality predictions
hybrid1_weights = np.power(gb_weights, 0.3)  # Gentler boost

hybrid1_result = evaluate_recommendations(train_idx, test_idx, quality_weights=hybrid1_weights)
print(f"Hybrid 1 (Content + GB) - Precision@5: {hybrid1_result['precision']:.4f} | NDCG@5: {hybrid1_result['ndcg']:.4f}")





# STEP 8: HYBRID 2 - CONTENT + SVR (Remove Direct Rating Boost)
print("\nSTEP 8: HYBRID 2 - CONTENT + SVR ENHANCED")
print("-" * 110)

# Use SVR predictions more conservatively
hybrid2_weights = np.power(svr_weights, 0.3)  # Even gentler boost

hybrid2_result = evaluate_recommendations(train_idx, test_idx, quality_weights=hybrid2_weights)
print(f"Hybrid 2 (Content + SVR Enhanced) - Precision@5: {hybrid2_result['precision']:.4f} | NDCG@5: {hybrid2_result['ndcg']:.4f}")

# ============================================================================
# STEP 9: MODEL COMPARISON
# ============================================================================

print("\n\n" + "=" * 110)
print("STEP 8: MODEL COMPARISON & RANKING")
print("=" * 110)

results = {
    'Content-Based (Baseline)': baseline_result,
    'ML Model 1: Gradient Boosting Quality Weights': gb_result,
    'ML Model 2: Support Vector Regression Weights': svr_result,
    'Hybrid 1: Content + GB': hybrid1_result,
    'Hybrid 2: Content + SVR + Rating': hybrid2_result
}

print("\n" + "-" * 110)
print(f"{'System':<50} | {'Precision@5':<15} | {'NDCG@5':<15}")
print("-" * 110)

for name, result in results.items():
    print(f"{name:<50} | {result['precision']:<15.4f} | {result['ndcg']:<15.4f}")

best_system = max(results.items(), key=lambda x: x[1]['precision'] * 0.5 + x[1]['ndcg'] * 0.5)
print("\n" + "=" * 110)
print(f"✓ BEST SYSTEM: {best_system[0]}")
print(f"  Precision@5: {best_system[1]['precision']:.4f}")
print(f"  NDCG@5: {best_system[1]['ndcg']:.4f}")
print("=" * 110)

# ============================================================================
# STEP 10: 5-FOLD CROSS-VALIDATION ON BEST SYSTEM
# ============================================================================

print("\n\nSTEP 9: 5-FOLD CROSS-VALIDATION ON BEST SYSTEM")
print("-" * 110)

best_system_name = best_system[0]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_precisions = []
cv_ndcgs = []

for fold_idx, (train_idx_cv, test_idx_cv) in enumerate(kf.split(data)):
    
    if 'GB' in best_system_name and 'SVR' not in best_system_name:
        # GB Model
        X_pairs_cv, y_pairs_cv = create_pairwise_data(
            content_features[train_idx_cv], 
            data['rating'].iloc[train_idx_cv].values
        )
        gb_cv = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        gb_cv.fit(X_pairs_cv, y_pairs_cv)
        
        gb_scores_cv = gb_cv.predict(content_features[train_idx_cv].toarray())
        gb_weights_cv = (gb_scores_cv - gb_scores_cv.min()) / (gb_scores_cv.max() - gb_scores_cv.min() + 1e-8)
        weights_cv = np.power(gb_weights_cv, 0.3)
        
    elif 'SVR' in best_system_name:
        # SVR Model
        sample_cv = np.random.choice(train_idx_cv, min(2000, len(train_idx_cv)), replace=False)
        X_svr_cv = content_features[sample_cv].toarray()
        y_svr_cv = data['rating'].iloc[sample_cv].values
        
        svr_cv = SVR(kernel='rbf', C=10, gamma='scale')
        svr_cv.fit(X_svr_cv, y_svr_cv)
        
        svr_scores_cv = svr_cv.predict(content_features[train_idx_cv].toarray())
        svr_weights_cv = (svr_scores_cv - svr_scores_cv.min()) / (svr_scores_cv.max() - svr_scores_cv.min() + 1e-8)
        
        if 'Rating' in best_system_name:
            # Hybrid 2
            train_ratings_cv = data['rating'].iloc[train_idx_cv].values
            rating_boost_cv = (train_ratings_cv - train_ratings_cv.mean()) / (train_ratings_cv.std() + 1e-8)
            rating_boost_cv = np.clip(rating_boost_cv, -1, 2)
            rating_boost_cv = (rating_boost_cv + 1) / 3
            weights_cv = 0.7 * svr_weights_cv + 0.3 * rating_boost_cv
        else:
            weights_cv = np.power(svr_weights_cv, 0.5)
    else:
        # Baseline
        weights_cv = None
    
    # Evaluate this fold
    result_cv = evaluate_recommendations(train_idx_cv, test_idx_cv, quality_weights=weights_cv)
    cv_precisions.append(result_cv['precision'])
    cv_ndcgs.append(result_cv['ndcg'])
    print(f"Fold {fold_idx + 1}: Precision@5 = {result_cv['precision']:.4f} | NDCG@5 = {result_cv['ndcg']:.4f}")

print(f"\n5-Fold CV Results:")
print(f"Precision@5: {np.mean(cv_precisions):.4f} ± {np.std(cv_precisions):.4f}")
print(f"NDCG@5: {np.mean(cv_ndcgs):.4f} ± {np.std(cv_ndcgs):.4f}")
print(f"Std Dev (Precision): {np.std(cv_precisions):.4f} → {'✓ Excellent' if np.std(cv_precisions) < 0.05 else '✓ Good' if np.std(cv_precisions) < 0.08 else '⚠ Check'}")

# ============================================================================
# STEP 11: SAMPLE RECOMMENDATIONS
# ============================================================================

print("\n\n" + "=" * 110)
print("STEP 10: SAMPLE RECOMMENDATIONS FROM BEST SYSTEM")
print("=" * 110)

# Use best weights for recommendations
if 'GB' in best_system_name and 'SVR' not in best_system_name:
    best_weights = gb_quality_weights
elif 'SVR' in best_system_name and 'Rating' in best_system_name:
    best_weights = hybrid2_weights
elif 'SVR' in best_system_name:
    best_weights = svr_quality_weights
else:
    best_weights = None

for sample_idx in range(min(3, len(test_idx))):
    test_original_idx = test_idx[sample_idx]
    story = data.iloc[test_original_idx]
    
    # Get recommendations for this test story
    test_feature = content_features[test_original_idx]
    similarities = cosine_similarity(test_feature, content_features[train_idx])[0]
    
    if best_weights is not None:
        similarities = similarities * best_weights
    
    top_5 = np.argsort(similarities)[::-1][:5]
    
    print(f"\n\nQuery Story: {story['story_name']} (Rating: {story['rating']:.2f})")
    print("-" * 110)
    print("Top 5 Recommendations:")
    
    for rank, pos in enumerate(top_5, 1):
        rec_story = data.iloc[train_idx[pos]]
        quality = "✓ HIGH" if rec_story['rating'] >= 8.0 else ""
        print(f"  {rank}. {rec_story['story_name'][:60]} | Rating: {rec_story['rating']:.2f} {quality}")

# ============================================================================
# STEP 12: USER INPUT RECOMMENDATIONS
# ============================================================================

def recommend_from_input(user_input, top_k=5):
    """
    Recommend stories based on either:
    - a story title in the dataset
    - or a free-text description
    """
    user_input = user_input.strip().lower()
    
    # Case 1: if it's an existing story title
    match = data[data['story_name'].str.lower().str.contains(user_input, case=False, na=False)]
    
    if not match.empty:
        idx = match.index[0]
        story = data.iloc[idx]
        print(f"\nGiven story: {story['story_name']} (Rating: {story['rating']:.2f})")
        
        # Use best weights for recommendations
        story_feature = content_features[idx]
        sims = cosine_similarity(story_feature, content_features)[0]
        
        if best_weights is not None and idx in train_idx:
            # Apply weights if available
            train_sims = cosine_similarity(story_feature, content_features[train_idx])[0]
            train_sims = train_sims * best_weights
            recs = np.argsort(train_sims)[::-1][1:top_k+1]
            recs = [train_idx[r] for r in recs]
        else:
            recs = np.argsort(sims)[::-1][1:top_k+1]
        
        print("\nTop Recommendations:")
        for rank, rec_idx in enumerate(recs, 1):
            rec_story = data.iloc[rec_idx]
            print(f"  {rank}. {rec_story['story_name']} | Rating: {rec_story['rating']:.2f}")
    
    else:
        # Case 2: free text (new input)
        print("\nNew input text detected.")
        input_vec = tfidf_vec.transform([user_input])
        input_features = hstack([
            input_vec * 0.7,
            csr_matrix(np.zeros((1, tag_matrix.shape[1])) * 0.2),
            csr_matrix(np.zeros((1, cat_matrix.shape[1])) * 0.1)
        ])
        
        sims = cosine_similarity(input_features, content_features[train_idx])[0]
        
        if best_weights is not None:
            sims = sims * best_weights
        
        recs = np.argsort(sims)[::-1][:top_k]
        recs = [train_idx[r] for r in recs]
        
        print("\nTop Recommendations:")
        for rank, rec_idx in enumerate(recs, 1):
            rec_story = data.iloc[rec_idx]
            print(f"  {rank}. {rec_story['story_name']} | Rating: {rec_story['rating']:.2f}")

print("\n\n" + "=" * 110)
print("ANALYSIS COMPLETE")
print("=" * 110)

# ============================================================================
# STEP 13: INTERACTIVE RECOMMENDATION PROMPT
# ============================================================================

while True:
    user_query = input("\nEnter a story title or short description (or 'exit' to quit): ").strip()
    if user_query.lower() == "exit":
        print("Goodbye! 👋")
        break
    if user_query:
        recommend_from_input(user_query)
