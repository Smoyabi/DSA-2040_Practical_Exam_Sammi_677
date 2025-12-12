# Classification Results

## Part A: Classification on Iris Dataset

### Decision Tree Classifier

**Performance Metrics:**
- Accuracy: 0.9333
- Precision: 0.9333
- Recall: 0.9333
- F1-score: 0.9333

**Confusion Matrix:**
```
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
```

### K-Nearest Neighbors (k=5)

**Performance Metrics:**
- Accuracy: 0.9667
- Precision: 0.9697
- Recall: 0.9667
- F1-score: 0.9666

### Model Comparison

**Winner:** KNN

**Analysis:**
Both models performed well on the Iris dataset. The KNN achieved slightly better accuracy (0.9667 vs 0.9333). The Decision Tree offers better interpretability through its visual representation, making it easier to understand which features drive predictions. KNN is simpler and doesn't require training but can be slower for large datasets. For this task, the KNN is recommended due to higher accuracy on the test set.

**Key Observations:**
- Both models achieved over 95% accuracy
- Very few misclassifications in the test set
- The Iris dataset is well-suited for both algorithms
