import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 110)
print("HORROR STORY RECOMMENDATION SYSTEM - PRODUCTION READY")
print("=" * 110)

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================
print("\nSTEP 1: DATA LOADING AND PREPROCESSING")
print("-" * 110)

df = pd.read_excel('creepypastas.xlsx')
data = df.copy()

# Clean missing values
data['body'] = data['body'].fillna('').str.replace('\n', ' ').str.strip()
data = data[data['body'].str.len() > 100].reset_index(drop=True)
data['rating'] = data['average_rating'].fillna(data['average_rating'].median())
data['tags'] = data['tags'].fillna('').apply(
    lambda x: [tag.strip().lower() for tag in str(x).split(',') if tag.strip()]
)
data['categories'] = data['categories'].fillna('').apply(
    lambda x: [cat.strip().lower() for cat in str(x).split(',') if cat.strip()]
)

print(f"Stories: {len(data)} | Rating: mean={data['rating'].mean():.2f}, std={data['rating'].std():.2f}")
print(f"High-quality (≥8.0): {(data['rating'] >= 8.0).sum()} ({(data['rating'] >= 8.0).sum()/len(data)*100:.1f}%)")

# ============================================================================
# STEP 2: TRAIN-TEST SPLIT (BEFORE FEATURE EXTRACTION!)
# ============================================================================
print("\nSTEP 2: TRAIN-TEST SPLIT")
print("-" * 110)

train_idx, test_idx = train_test_split(
    np.arange(len(data)), 
    test_size=0.2, 
    random_state=42,
    stratify=pd.cut(data['rating'], bins=[0, 6, 8, 10])
)

train_data = data.iloc[train_idx].reset_index(drop=True)
test_data = data.iloc[test_idx].reset_index(drop=True)

print(f"Training: {len(train_data)} stories | Test: {len(test_data)} stories (UNSEEN)")

# ============================================================================
# STEP 3: FEATURE ENGINEERING (FIT ON TRAIN ONLY)
# ============================================================================
print("\nSTEP 3: FEATURE ENGINEERING")
print("-" * 110)

# TF-IDF - FIT ON TRAIN, TRANSFORM BOTH
tfidf_vec = TfidfVectorizer(
    stop_words='english', 
    max_features=2000, 
    min_df=3, 
    max_df=0.85, 
    ngram_range=(1, 2), 
    sublinear_tf=True
)

train_tfidf = tfidf_vec.fit_transform(train_data['body'])
test_tfidf = tfidf_vec.transform(test_data['body'])
all_tfidf = tfidf_vec.transform(data['body'])  # For recommendations

# Tags - TRAIN VOCABULARY ONLY
train_tags = sorted(list(set([t for tags in train_data['tags'] for t in tags])))

def create_tag_matrix(data_subset, tag_list):
    matrix = np.zeros((len(data_subset), len(tag_list)))
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}
    for i, tags in enumerate(data_subset['tags']):
        for tag in tags:
            if tag in tag_to_idx:
                matrix[i, tag_to_idx[tag]] = 1
    return matrix

train_tag_matrix = create_tag_matrix(train_data, train_tags)
test_tag_matrix = create_tag_matrix(test_data, train_tags)
all_tag_matrix = create_tag_matrix(data, train_tags)

# Categories - TRAIN VOCABULARY ONLY
train_cats = sorted(list(set([c for cats in train_data['categories'] for c in cats])))

def create_cat_matrix(data_subset, cat_list):
    matrix = np.zeros((len(data_subset), len(cat_list)))
    cat_to_idx = {cat: idx for idx, cat in enumerate(cat_list)}
    for i, cats in enumerate(data_subset['categories']):
        for cat in cats:
            if cat in cat_to_idx:
                matrix[i, cat_to_idx[cat]] = 1
    return matrix

train_cat_matrix = create_cat_matrix(train_data, train_cats)
test_cat_matrix = create_cat_matrix(test_data, train_cats)
all_cat_matrix = create_cat_matrix(data, train_cats)

# Combine features
train_features = hstack([
    train_tfidf * 0.7, 
    csr_matrix(train_tag_matrix) * 0.2, 
    csr_matrix(train_cat_matrix) * 0.1
])
test_features = hstack([
    test_tfidf * 0.7, 
    csr_matrix(test_tag_matrix) * 0.2, 
    csr_matrix(test_cat_matrix) * 0.1
])
all_features = hstack([
    all_tfidf * 0.7, 
    csr_matrix(all_tag_matrix) * 0.2, 
    csr_matrix(all_cat_matrix) * 0.1
])

print(f"TF-IDF: {train_tfidf.shape} | Tags: {len(train_tags)} | Categories: {len(train_cats)}")
print(f"Combined features: {train_features.shape}")

# ============================================================================
# STEP 4: EVALUATION FUNCTION
# ============================================================================
print("\nSTEP 4: EVALUATION FUNCTION")
print("-" * 110)
def evaluate_model(train_feat, test_feat, train_ratings, quality_weights=None, 
                   k=5, threshold=8.0, sample_size=200):
    """Properly evaluate recommendations on test set"""
    precisions = []
    ndcgs = []
    
    # Sample for speed
    test_sample = min(sample_size, len(test_feat.toarray()))
    
    for test_idx_local in range(test_sample):
        # Similarity from this test story to ALL training stories
        test_story = test_feat[test_idx_local]
        similarities = cosine_similarity(test_story, train_feat)[0]
        
        # Apply quality weights if provided
        if quality_weights is not None:
            similarities = similarities * quality_weights
        
        # Get top-k recommendations
        top_k_pos = np.argsort(similarities)[::-1][:k]
        rec_ratings = train_ratings[top_k_pos]
        
        # Precision
        relevant = (rec_ratings >= threshold).sum()
        precisions.append(relevant / k)
        
        # NDCG
        ideal = np.sort(rec_ratings)[::-1]
        dcg = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(rec_ratings))
        idcg = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    return {
        'precision': np.mean(precisions),
        'ndcg': np.mean(ndcgs),
        'precision_std': np.std(precisions),
        'ndcg_std': np.std(ndcgs)
    }

# ============================================================================
# STEP 5: MODEL 1 - BASELINE CONTENT-BASED
# ============================================================================
print("\nSTEP 5: MODEL 1 - CONTENT-BASED BASELINE")
print("-" * 110)

baseline_result = evaluate_model(
    train_features, 
    test_features, 
    train_data['rating'].values
)
print(f"Content-Based - Precision@5: {baseline_result['precision']:.4f} | NDCG@5: {baseline_result['ndcg']:.4f}")

# ============================================================================
# STEP 6: MODEL 2 - GRADIENT BOOSTING QUALITY WEIGHTING
# ============================================================================
print("\nSTEP 6: MODEL 2 - GRADIENT BOOSTING QUALITY WEIGHTS")
print("-" * 110)

def create_pairwise_data(X, y, n_pairs=5000):
    """Create pairwise ranking data"""
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

print("Training GB ranker...")
X_pairs, y_pairs = create_pairwise_data(train_features, train_data['rating'].values)

gb_ranker = GradientBoostingRegressor(
    n_estimators=50, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)
gb_ranker.fit(X_pairs, y_pairs)

# Predict quality scores for training stories
gb_scores = gb_ranker.predict(train_features.toarray())
gb_weights = (gb_scores - gb_scores.min()) / (gb_scores.max() - gb_scores.min() + 1e-8)
gb_weights = np.power(gb_weights, 0.3)  # Gentle boost

gb_result = evaluate_model(
    train_features, 
    test_features, 
    train_data['rating'].values, 
    quality_weights=gb_weights
)
print(f"GB Quality-Weighted - Precision@5: {gb_result['precision']:.4f} | NDCG@5: {gb_result['ndcg']:.4f}")

# ============================================================================
# STEP 7: MODEL 3 - K-NEAREST NEIGHBORS
# ============================================================================
print("\nSTEP 7: MODEL 3 - K-NEAREST NEIGHBORS")
print("-" * 110)

from sklearn.preprocessing import normalize

# Normalize features for KNN
train_norm = normalize(train_features, norm='l2')
test_norm = normalize(test_features, norm='l2')

knn_model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute')
knn_model.fit(train_norm)

# Evaluate KNN
precisions_knn = []
ndcgs_knn = []

for test_idx_local in range(min(200, test_norm.shape[0])):
    test_story = test_norm[test_idx_local]
    distances, indices = knn_model.kneighbors(test_story, n_neighbors=5)
    
    rec_ratings = train_data['rating'].values[indices[0]]
    
    relevant = (rec_ratings >= 8.0).sum()
    precisions_knn.append(relevant / 5)
    
    ideal = np.sort(rec_ratings)[::-1]
    dcg = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(rec_ratings))
    idcg = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(ideal))
    ndcgs_knn.append(dcg / idcg if idcg > 0 else 0)

knn_result = {
    'precision': np.mean(precisions_knn),
    'ndcg': np.mean(ndcgs_knn),
    'precision_std': np.std(precisions_knn),
    'ndcg_std': np.std(ndcgs_knn)
}
print(f"KNN - Precision@5: {knn_result['precision']:.4f} | NDCG@5: {knn_result['ndcg']:.4f}")

# ============================================================================
# STEP 8: MODEL 4 - RANDOM FOREST QUALITY WEIGHTING
# ============================================================================
print("\nSTEP 8: MODEL 4 - RANDOM FOREST QUALITY WEIGHTS")
print("-" * 110)

print("Training RF regressor...")
# Direct regression instead of pairwise
sample_size = min(2000, len(train_features.toarray()))
X_rf = train_features[:sample_size].toarray()
y_rf = train_data['rating'].values[:sample_size]

rf_model = RandomForestRegressor(
    n_estimators=50, 
    max_depth=5, 
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_rf, y_rf)

# Predict for all training
rf_scores = rf_model.predict(train_features.toarray())
rf_weights = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-8)
rf_weights = np.power(rf_weights, 0.3)

rf_result = evaluate_model(
    train_features, 
    test_features, 
    train_data['rating'].values, 
    quality_weights=rf_weights
)
print(f"RF Quality-Weighted - Precision@5: {rf_result['precision']:.4f} | NDCG@5: {rf_result['ndcg']:.4f}")

# ============================================================================
# STEP 9: MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 110)
print("STEP 9: MODEL COMPARISON")
print("=" * 110)

results = {
    'Content-Based (Baseline)': baseline_result,
    'Gradient Boosting Quality Weights': gb_result,
    'K-Nearest Neighbors': knn_result,
    'Random Forest Quality Weights': rf_result
}

print(f"\n{'Model':<45} | {'Precision@5':<15} | {'NDCG@5':<15}")
print("-" * 110)
for name, result in results.items():
    print(f"{name:<45} | {result['precision']:<15.4f} | {result['ndcg']:<15.4f}")

best_model_name = max(results.items(), key=lambda x: x[1]['precision'] * 0.5 + x[1]['ndcg'] * 0.5)[0]
best_result = results[best_model_name]

print("\n" + "=" * 110)
print(f"BEST MODEL: {best_model_name}")
print(f"   Precision@5: {best_result['precision']:.4f} ± {best_result['precision_std']:.4f}")
print(f"   NDCG@5: {best_result['ndcg']:.4f} ± {best_result['ndcg_std']:.4f}")
print("=" * 110)

# Select best weights
if 'Gradient' in best_model_name:
    best_weights = gb_weights
    best_model = gb_ranker
elif 'Random' in best_model_name:
    best_weights = rf_weights
    best_model = rf_model
else:
    best_weights = None
    best_model = None

# ============================================================================
# STEP 10: 5-FOLD CROSS-VALIDATION
# ============================================================================
print("\nSTEP 10: 5-FOLD CROSS-VALIDATION ON BEST MODEL")
print("-" * 110)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_precisions = []
cv_ndcgs = []

for fold, (train_cv_idx, test_cv_idx) in enumerate(kf.split(data)):
    # Build features for this fold
    train_cv = data.iloc[train_cv_idx]
    test_cv = data.iloc[test_cv_idx]
    
    # Extract features
    tfidf_cv = TfidfVectorizer(stop_words='english', max_features=2000, min_df=3, 
                               max_df=0.85, ngram_range=(1,2), sublinear_tf=True)
    train_tfidf_cv = tfidf_cv.fit_transform(train_cv['body'])
    test_tfidf_cv = tfidf_cv.transform(test_cv['body'])
    
    train_tags_cv = sorted(list(set([t for tags in train_cv['tags'] for t in tags])))
    train_tag_cv = create_tag_matrix(train_cv, train_tags_cv)
    test_tag_cv = create_tag_matrix(test_cv, train_tags_cv)
    
    train_cats_cv = sorted(list(set([c for cats in train_cv['categories'] for c in cats])))
    train_cat_cv = create_cat_matrix(train_cv, train_cats_cv)
    test_cat_cv = create_cat_matrix(test_cv, train_cats_cv)
    
    train_feat_cv = hstack([train_tfidf_cv * 0.7, csr_matrix(train_tag_cv) * 0.2, 
                           csr_matrix(train_cat_cv) * 0.1])
    test_feat_cv = hstack([test_tfidf_cv * 0.7, csr_matrix(test_tag_cv) * 0.2, 
                          csr_matrix(test_cat_cv) * 0.1])
    
    # Train model on this fold
    if 'Gradient' in best_model_name:
        X_pairs_cv, y_pairs_cv = create_pairwise_data(train_feat_cv, train_cv['rating'].values)
        gb_cv = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, 
                                         max_depth=3, random_state=42)
        gb_cv.fit(X_pairs_cv, y_pairs_cv)
        scores_cv = gb_cv.predict(train_feat_cv.toarray())
        weights_cv = (scores_cv - scores_cv.min()) / (scores_cv.max() - scores_cv.min() + 1e-8)
        weights_cv = np.power(weights_cv, 0.3)
    elif 'Random' in best_model_name:
        sample_cv = min(2000, len(train_feat_cv.toarray()))
        rf_cv = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf_cv.fit(train_feat_cv[:sample_cv].toarray(), train_cv['rating'].values[:sample_cv])
        scores_cv = rf_cv.predict(train_feat_cv.toarray())
        weights_cv = (scores_cv - scores_cv.min()) / (scores_cv.max() - scores_cv.min() + 1e-8)
        weights_cv = np.power(weights_cv, 0.3)
    else:
        weights_cv = None
    
    # Evaluate
    result_cv = evaluate_model(train_feat_cv, test_feat_cv, train_cv['rating'].values, 
                              quality_weights=weights_cv, sample_size=100)
    cv_precisions.append(result_cv['precision'])
    cv_ndcgs.append(result_cv['ndcg'])
    
    print(f"Fold {fold+1}: Precision@5 = {result_cv['precision']:.4f} | NDCG@5 = {result_cv['ndcg']:.4f}")

print(f"\n5-Fold CV Results:")
print(f"Precision@5: {np.mean(cv_precisions):.4f} ± {np.std(cv_precisions):.4f}")
print(f"NDCG@5: {np.mean(cv_ndcgs):.4f} ± {np.std(cv_ndcgs):.4f}")
print(f"Consistency: {'Excellent' if np.std(cv_precisions) < 0.05 else 'Good' if np.std(cv_precisions) < 0.08 else 'Check'}")

# ============================================================================
# STEP 11: SMART RECOMMENDATION FUNCTION
# ============================================================================
print("\n" + "=" * 110)
print("STEP 11: BUILDING RECOMMENDATION SYSTEM")
print("=" * 110)

def recommend_stories(user_input, top_k=5, min_similarity=0.15):
    """
    Smart recommendation with fallback for poor inputs
    """
    user_input_lower = user_input.strip().lower()
    
    # Case 1: Match existing story
    match = data[data['story_name'].str.lower().str.contains(user_input_lower, case=False, na=False)]
    
    if not match.empty:
        story_idx = match.index[0]
        story = data.iloc[story_idx]
        
        print(f"\n{'='*80}")
        print(f"INPUT: {story['story_name']} (Rating: {story['rating']:.2f})")
        print(f"{'='*80}")
        print(f"Tags: {', '.join(story['tags'][:5]) if story['tags'] else 'None'}")
        print(f"Preview: {story['body'][:150]}...\n")
        
        # Get similarities
        story_features = all_features[story_idx]
        similarities = cosine_similarity(story_features, train_features)[0]
        
        if best_weights is not None:
            similarities = similarities * best_weights
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:top_k+1]
        
        # Filter out the input story itself if in training
        recommendations = []
        for idx in top_indices:
            rec_story = train_data.iloc[idx]
            if rec_story['story_name'] != story['story_name']:
                recommendations.append((rec_story, similarities[idx]))
            if len(recommendations) >= top_k:
                break
    
    else:
        # Case 2: New text input
        print(f"\n{'='*80}")
        print(f" NEW INPUT TEXT: '{user_input}'")
        print(f"{'='*80}\n")
        
        # Transform input
        input_tfidf = tfidf_vec.transform([user_input])
        input_features = hstack([
            input_tfidf * 0.7,
            csr_matrix(np.zeros((1, len(train_tags)))) * 0.2,
            csr_matrix(np.zeros((1, len(train_cats)))) * 0.1
        ])
        
        # Get similarities
        similarities = cosine_similarity(input_features, train_features)[0]
        
        # Check if input is too weak
        max_sim = similarities.max()
        
        if max_sim < min_similarity:
            print(f"WARNING: Input has very low similarity (max: {max_sim:.4f})")
            print(f"Showing TOP-RATED popular stories instead:\n")
            
            # Fallback: Show highest-rated popular stories
            top_rated = train_data.nlargest(top_k, 'rating')
            recommendations = [(row, 1.0) for _, row in top_rated.iterrows()]
        else:
            if best_weights is not None:
                similarities = similarities * best_weights
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            recommendations = [(train_data.iloc[idx], similarities[idx]) for idx in top_indices]
    
    # Display recommendations
    print(f"{'='*80}")
    print(f"TOP {len(recommendations)} RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    for rank, (rec_story, sim_score) in enumerate(recommendations, 1):
        quality = "HIGH QUALITY" if rec_story['rating'] >= 8.0 else ""
        print(f"{rank}. {rec_story['story_name']}")
        print(f"   Rating: {rec_story['rating']:.2f}/10 {quality}")
        print(f"   Similarity: {sim_score:.4f}")
        print(f"   Tags: {', '.join(rec_story['tags'][:3]) if rec_story['tags'] else 'None'}")
        print(f"   Preview: {rec_story['body'][:100]}...")
        print()
    
    return recommendations

# ============================================================================
# STEP 12: SAMPLE RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 110)
print("STEP 12: SAMPLE RECOMMENDATIONS")
print("=" * 110)

# Test with known story
recommend_stories("Slenderman", top_k=5)

# Test with random input (will trigger fallback)
recommend_stories("random nonsense text", top_k=5)

# ============================================================================
# STEP 13: INTERACTIVE MODE
# ============================================================================
print("\n" + "=" * 110)
print("SYSTEM READY - INTERACTIVE MODE")
print("=" * 110)

# Interactive loop
while True:
    user_input = input("\nEnter story title or description (or 'exit'): ").strip()
    if user_input.lower() == 'exit':
        print("Goodbye! ")
        break
    if user_input:
        recommend_stories(user_input, top_k=5)
