# Anomaly Detection Performance Analysis

## Current Performance Issues

Your model achieved only **46.38% accuracy** with severe performance degradation. Here's a deep analysis of the root causes:

## 1. Severe Overfitting 🚨

**Evidence:**
- Training accuracy: 100% by epoch 2
- Training loss: 0.0331
- Validation accuracy: 46.4% (final)
- Validation loss: 1.780 (increasing)

**Root Cause:** The 22.5M parameter transformer is memorizing the training data rather than learning generalizable patterns.

## 2. Extreme Class Imbalance 📊

**Evidence:**
- Most predictions concentrated in classes -1 and 0
- Classes 2, 6, 102, 106, 107 have 0.00 precision/recall
- Weighted avg precision (0.70) >> macro avg (0.15)

**Root Cause:** Normal operations (class 0) dominate the dataset while anomalies are rare, causing the model to predict "normal" for everything.

## 3. Data Leakage in Time Series 🕐

**Current Approach:**
```python
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx]  # Random 80/20 split
```

**Problem:** Time series data requires temporal splits to prevent future information bleeding into training.

## 4. Architecture Mismatch 🏗️

**Issues:**
- Encoder-decoder transformers designed for sequence-to-sequence tasks
- Anomaly detection typically needs simpler architectures
- 128-sample sequences may not capture relevant anomaly patterns
- Attention mechanisms may focus on noise rather than anomalies

## 5. Target Mapping Confusion 🎯

**Evidence:**
```
Unique classes in predictions: [0 1 2 5 7 8 9]
Classes in confusion matrix: ['-1', '0', '2', '6', '102', '106', '107']
```

**Problem:** Inconsistent class encoding between training and evaluation.

## 6. Validation Strategy Issues ✅

**Problems:**
- Early stopping on `val_loss` but validation metrics are erratic
- No stratified sampling ensuring anomaly representation in validation
- Batch size (64) may not capture rare anomaly patterns

## Recommended Solutions

### Immediate Fixes

1. **Reduce Model Complexity**
   - Use simpler architectures (LSTM, 1D CNN, or basic feedforward)
   - Reduce parameters from 22.5M to <1M
   - Consider autoencoder for unsupervised anomaly detection

2. **Fix Data Splits**
   ```python
   # Temporal split
   split_time = df['timestamp'].quantile(0.8)
   train_df = df[df['timestamp'] < split_time]
   test_df = df[df['timestamp'] >= split_time]
   ```

3. **Address Class Imbalance**
   - Use class weights in loss function
   - Apply SMOTE or undersampling
   - Focus on anomaly detection metrics (F1, AUC-ROC)

4. **Simplify Target Encoding**
   - Binary classification: Normal vs Anomaly
   - Use consistent label encoding throughout

### Architecture Alternatives

1. **Isolation Forest** - Simple, effective for anomaly detection
2. **Autoencoder** - Reconstruction error for anomaly scoring
3. **LSTM + Dense** - Simpler sequence modeling
4. **One-Class SVM** - Unsupervised anomaly detection

### Evaluation Improvements

1. **Use Anomaly-Specific Metrics**
   - AUC-ROC, AUC-PR
   - F1-score for minority classes
   - Precision@K for top anomalies

2. **Cross-Validation**
   - Time series cross-validation
   - Ensure temporal ordering

3. **Threshold Tuning**
   - Optimize classification threshold
   - Balance precision/recall for anomalies

## Expected Improvements

Implementing these changes should achieve:
- **Accuracy**: 80-90% (vs current 46%)
- **Anomaly Detection**: F1 > 0.6 for rare events
- **Generalization**: Validation performance close to training
- **Inference Speed**: 10x faster with simpler model

## Next Steps Priority

1. 🔴 **High**: Implement temporal data split
2. 🔴 **High**: Reduce model complexity 
3. 🟡 **Medium**: Address class imbalance
4. 🟡 **Medium**: Binary anomaly classification
5. 🟢 **Low**: Hyperparameter optimization

The current approach treats this as a complex sequence-to-sequence problem when it's fundamentally an anomaly detection task requiring different techniques.