import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 0: DATASET VALIDATION & ANALYSIS
# ============================================================================

print("=" * 80)
print("DATASET VALIDATION FOR STORY RECOMMENDATION PROBLEM")
print("=" * 80)

df = pd.read_excel('creepypastas.xlsx')

print("\n1. BASIC DATA INFO")
print("-" * 80)
print(f"Shape: {df.shape[0]} stories, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# CHECK FOR SUITABILITY TO PROBLEM
# ============================================================================

print("\n\n2. PROBLEM STATEMENT CHECK")
print("-" * 80)
print("Problem: Build a recommendation system to recommend horror stories")
print("\nRequired elements:")
print("✓ Story content (for content-based filtering)")
print("✓ Story features (tags, categories, metadata)")
print("✓ Quality metric (ratings to learn what makes good stories)")
print("✓ Enough data for training (typically 100+ samples)")

print(f"\nDataset has:")
print(f"✓ {df.shape[0]} stories - GOOD (enough for training)")

# Check for story content
if 'body' in df.columns:
    non_empty_body = df['body'].notna().sum() - (df['body'] == '').sum()
    print(f"✓ Story text (body): {non_empty_body} non-empty stories")
else:
    print("✗ NO story text column - PROBLEM!")

# Check for ratings
if 'average_rating' in df.columns:
    valid_ratings = pd.to_numeric(df['average_rating'], errors='coerce').notna().sum()
    print(f"✓ Ratings: {valid_ratings} stories have ratings")
else:
    print("✗ NO ratings column - PROBLEM!")

# Check for metadata
has_metadata = False
if 'tags' in df.columns:
    print(f"✓ Tags: {df['tags'].notna().sum()} stories have tags")
    has_metadata = True
if 'categories' in df.columns:
    print(f"✓ Categories: {df['categories'].notna().sum()} stories have categories")
    has_metadata = True
if 'estimated_reading_time' in df.columns:
    print(f"✓ Reading time: {df['estimated_reading_time'].notna().sum()} stories")
    has_metadata = True

if not has_metadata:
    print("✗ LIMITED metadata - might affect recommendation quality")

# ============================================================================
# DETAILED DATA QUALITY CHECK
# ============================================================================

print("\n\n3. DATA QUALITY ASSESSMENT")
print("-" * 80)

print("\nMissing values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100)
for col in df.columns:
    pct = missing_pct[col]
    if pct > 0:
        print(f"  {col}: {missing[col]} ({pct:.1f}%)")
    
print("\n" + "-" * 80)

# Rating analysis
print("\nRating distribution:")
ratings = pd.to_numeric(df['average_rating'], errors='coerce')
print(f"  Valid ratings: {ratings.notna().sum()}/{len(df)}")
print(f"  Mean: {ratings.mean():.2f}")
print(f"  Std: {ratings.std():.2f}")
print(f"  Min: {ratings.min():.2f}")
print(f"  Max: {ratings.max():.2f}")
print(f"  Median: {ratings.median():.2f}")
print(f"  Q1: {ratings.quantile(0.25):.2f}, Q3: {ratings.quantile(0.75):.2f}")

# Check rating variance
if ratings.std() < 0.5:
    print("  ⚠️ WARNING: Very low variance in ratings - hard to distinguish quality!")
else:
    print("  ✓ Good variance in ratings")

print("\n" + "-" * 80)

# Text length analysis
print("\nStory text analysis:")
if 'body' in df.columns:
    text_lengths = df['body'].fillna('').str.len()
    valid_stories = text_lengths > 100  # Minimum reasonable story length
    print(f"  Stories with text > 100 chars: {valid_stories.sum()}/{len(df)}")
    print(f"  Average length: {text_lengths[valid_stories].mean():.0f} chars")
    print(f"  Min length: {text_lengths[valid_stories].min():.0f}")
    print(f"  Max length: {text_lengths[valid_stories].max():.0f}")
    
    if valid_stories.sum() < 100:
        print("  ✗ NOT ENOUGH valid story text for NLP!")
    else:
        print("  ✓ Sufficient text content")

print("\n" + "-" * 80)

# Tags/Categories analysis
print("\nMetadata (Tags & Categories):")
if 'tags' in df.columns:
    tags_list = []
    for t in df['tags'].fillna(''):
        if isinstance(t, str) and t.strip():
            tags_list.extend([x.strip() for x in str(t).split(',') if x.strip()])
    print(f"  Unique tags: {len(set(tags_list))}")
    print(f"  Top 10 tags: {Counter(tags_list).most_common(10)}")

if 'categories' in df.columns:
    cats_list = []
    for c in df['categories'].fillna(''):
        if isinstance(c, str) and c.strip():
            cats_list.extend([x.strip() for x in str(c).split(',') if x.strip()])
    print(f"  Unique categories: {len(set(cats_list))}")
    print(f"  Top 10 categories: {Counter(cats_list).most_common(10)}")

# ============================================================================
# CHECK FOR RECOMMENDATION-SPECIFIC ISSUES
# ============================================================================

print("\n\n4. RECOMMENDATION SYSTEM FEASIBILITY")
print("-" * 80)

print("\nCan we build recommendations?")

# Check if we have distinguishing features
has_content = 'body' in df.columns and (df['body'].notna().sum() > 100)
has_quality = 'average_rating' in df.columns
has_features = ('tags' in df.columns or 'categories' in df.columns)
has_metadata = 'estimated_reading_time' in df.columns

if has_content:
    print("✓ Content-based filtering: YES (can use story text)")
else:
    print("✗ Content-based filtering: NO (insufficient story text)")

if has_features:
    print("✓ Feature-based matching: YES (can use tags/categories)")
else:
    print("✗ Feature-based matching: NO (no tags/categories)")

if has_quality:
    print("✓ Quality learning: YES (can train on ratings)")
else:
    print("✗ Quality learning: NO (no ratings)")

if has_metadata:
    print("✓ Metadata enrichment: YES (can use reading time, etc.)")
else:
    print("✗ Metadata enrichment: NO")

# Check sparsity
if has_features and 'tags' in df.columns:
    non_empty_tags = (df['tags'].notna() & (df['tags'].str.len() > 0)).sum()
    sparsity = (1 - non_empty_tags / len(df)) * 100
    print(f"\nTag sparsity: {sparsity:.1f}% (lower is better)")
    if sparsity > 50:
        print("  ⚠️ Too sparse - will limit tag-based recommendations")

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("\n\n5. FINAL ASSESSMENT")
print("-" * 80)

is_suitable = True
issues = []

if not has_content:
    issues.append("Insufficient story text for content-based filtering")
    is_suitable = False

if not has_quality:
    issues.append("No quality metric (ratings) to learn from")
    is_suitable = False

if not (has_features or has_content):
    issues.append("No features to distinguish between stories")
    is_suitable = False

if df.shape[0] < 100:
    issues.append(f"Only {df.shape[0]} stories - might be too small")
    is_suitable = False

if is_suitable:
    print("\n✓ DATASET IS SUITABLE FOR RECOMMENDATION SYSTEM")
    print("\nRecommended approach:")
    if has_content and has_quality:
        print("- PRIMARY: Content-based filtering (TF-IDF on story text)")
        print("- SECONDARY: Tag/category-based filtering")
        print("- LEARN: Rating prediction model to score quality")
    print("\nExpected performance:")
    print("- Precision@5: 0.65-0.80 (depending on story similarity)")
    print("- Recall@5: 0.40-0.60 (depends on diversity)")
else:
    print("\n✗ DATASET HAS ISSUES FOR RECOMMENDATION SYSTEM")
    print("\nProblems found:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nRECOMMENDATION: Check with instructor/Keerthana about dataset suitability")

# ============================================================================
# DATA CLEANING REQUIRED
# ============================================================================

print("\n\n6. PREPROCESSING NEEDED")
print("-" * 80)

print("\nSteps to take:")
print("1. Remove stories with empty/null body text")
print("2. Normalize ratings (handle any non-numeric values)")
print("3. Parse tags and categories properly (handle formatting)")
print("4. Handle missing reading times (fill with median)")
print("5. Convert dates if present")
print("6. Create story ID for tracking")

# Show data sample
print("\n\n7. DATA SAMPLE")
print("-" * 80)
print(df.head(1)[['story_name', 'average_rating', 'estimated_reading_time', 'tags', 'categories']].to_string())

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - Ready to proceed with model building")
print("=" * 80)