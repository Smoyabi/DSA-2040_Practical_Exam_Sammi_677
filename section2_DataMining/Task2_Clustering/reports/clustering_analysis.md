# Clustering Analysis Report

## Task 2: K-Means Clustering on Iris Dataset

### Cluster Quality and Separation

The K-Means clustering with k=3 achieved an Adjusted Rand Index (ARI) of approximately 0.73, indicating good alignment between the predicted clusters and actual Iris species. The elbow curve clearly shows a bend at k=3, confirming this as the optimal number of clusters. The inertia decreases significantly from k=1 to k=3, then levels off, which supports our choice.

### Cluster Overlap and Misclassifications

The scatter plot reveals that one cluster is very well separated (likely Setosa), while the other two clusters show some overlap in the petal length-width space. This overlap explains why the ARI isn't perfect—some Versicolor and Virginica samples are difficult to distinguish based purely on these features. The misclassifications mainly occur in the boundary region between these two species.

### Real-World Applications

This clustering approach demonstrates practical value for customer segmentation in business contexts. For example, an e-commerce company could use K-Means to group customers based on purchasing behavior (amount spent, frequency, product categories). These segments would help personalize marketing campaigns, optimize inventory, and identify high-value customer groups. The ARI metric would measure how well the automated segmentation aligns with business-defined customer categories.

**Total Word Count:** 198 words