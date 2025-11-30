# Changelog

## 2025-11-30

### 🚀 Major Algorithm Improvements

#### 1. Database-Level Pre-filtering
- **Before**: Fetched all photos, then filtered in application layer
- **After**: Filter excluded photos and gender at MongoDB query level
- **Impact**: Significantly improved query performance and reduced memory usage

#### 2. Dynamic Learning Rate
- **Before**: Fixed decay rate (0.9) for all users
- **After**: Adaptive decay based on user experience
  - Early stage (0-10 likes): Fast learning (0.5-0.7)
  - Mid stage (10-50 likes): Gradual stabilization (0.7-0.9)
  - Stable stage (50+ likes): High stability (0.9)
- **Impact**: New users adapt faster to preferences, experienced users have stable recommendations

#### 3. Negative Feedback Learning
- **Before**: Only "like" actions updated user preferences
- **After**: "Dislike" actions push preferences away from rejected photos
- **Impact**: More accurate preference modeling through both positive and negative signals

#### 4. Pure Similarity-Based Recommendations
- **Before**: Mixed personalized and random exploration (ε-greedy)
- **After**: 100% similarity-based using FAISS vector search
  - Existing users: Use their preference embedding
  - New users: Use dataset average as neutral starting point
- **Impact**: More consistent and predictable recommendations

### 🔧 Configuration

New settings in `app/core/config.py`:
```python
exploration_epsilon: 0.2  # (Removed - no longer used)
negative_feedback_weight: 0.05
early_learning_decay_min: 0.5
early_learning_decay_max: 0.7
mid_learning_decay: 0.9
early_learning_threshold: 10
stable_learning_threshold: 50
```

### 📝 API Changes

**GET /users/{username}/recommendations**
- Response now includes `recommendation_type`:
  - `personalized`: User has established preferences
  - `new_user_neutral`: New user (uses dataset average)
  - `no_available_photos`: No photos available
  - `no_valid_embeddings`: No valid embeddings found

**POST /swipe**
- "Pass" action now applies negative feedback to user preferences
- Response includes `negative_feedback_applied: true` when applicable

### 🐛 Bug Fixes
- Fixed numpy array handling for empty photo lists
- Added proper error handling for missing embeddings
- Fixed FAISS index creation for filtered photo sets
