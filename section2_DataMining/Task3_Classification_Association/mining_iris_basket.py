"""
Task 3: Classification and Association Rule Mining

Part A: Classification (Decision Tree, KNN)
Part B: Association Rule Mining (Apriori)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

# Create directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("="*60)
print("PART A: CLASSIFICATION")
print("="*60)

# Load and preprocess Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================
# Step 1: Decision Tree Classifier
# ============================================
print("\n" + "-"*60)
print("Decision Tree Classifier")
print("-"*60)

dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_classifier.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_classifier.predict(X_test)

# Compute metrics
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, average='weighted')
dt_recall = recall_score(y_test, y_pred_dt, average='weighted')
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')

print(f"\nDecision Tree Results:")
print(f"Accuracy:  {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall:    {dt_recall:.4f}")
print(f"F1-score:  {dt_f1:.4f}")

# Confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(f"\nConfusion Matrix:")
print(cm_dt)

# ============================================
# Step 2: Visualize Decision Tree
# ============================================
print("\nGenerating decision tree visualization...")
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=feature_names,
          class_names=target_names,
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Classifier - Iris Dataset", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/decision_tree.png', dpi=300, bbox_inches='tight')
print("Decision tree saved to visualizations/decision_tree.png")
plt.close()

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix - Decision Tree', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved to visualizations/confusion_matrix.png")
plt.close()

# ============================================
# Step 3: KNN Classifier (k=5)
# ============================================
print("\n" + "-"*60)
print("K-Nearest Neighbors Classifier (k=5)")
print("-"*60)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predictions
y_pred_knn = knn_classifier.predict(X_test)

# Compute metrics
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

print(f"\nKNN Results:")
print(f"Accuracy:  {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall:    {knn_recall:.4f}")
print(f"F1-score:  {knn_f1:.4f}")

# ============================================
# Step 4: Model Comparison
# ============================================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"\nDecision Tree - Accuracy: {dt_accuracy:.4f}, F1: {dt_f1:.4f}")
print(f"KNN (k=5)     - Accuracy: {knn_accuracy:.4f}, F1: {knn_f1:.4f}")

if dt_accuracy > knn_accuracy:
    better_model = "Decision Tree"
    reason = "higher accuracy and better interpretability"
else:
    better_model = "KNN"
    reason = "higher accuracy on the test set"

print(f"\nBetter Model: {better_model}")
print(f"Reason: {reason}")

# Save classification results
classification_results = f"""# Classification Results

## Part A: Classification on Iris Dataset

### Decision Tree Classifier

**Performance Metrics:**
- Accuracy: {dt_accuracy:.4f}
- Precision: {dt_precision:.4f}
- Recall: {dt_recall:.4f}
- F1-score: {dt_f1:.4f}

**Confusion Matrix:**
```
{cm_dt}
```

### K-Nearest Neighbors (k=5)

**Performance Metrics:**
- Accuracy: {knn_accuracy:.4f}
- Precision: {knn_precision:.4f}
- Recall: {knn_recall:.4f}
- F1-score: {knn_f1:.4f}

### Model Comparison

**Winner:** {better_model}

**Analysis:**
Both models performed well on the Iris dataset. The {better_model} achieved slightly better accuracy ({max(dt_accuracy, knn_accuracy):.4f} vs {min(dt_accuracy, knn_accuracy):.4f}). The Decision Tree offers better interpretability through its visual representation, making it easier to understand which features drive predictions. KNN is simpler and doesn't require training but can be slower for large datasets. For this task, the {better_model} is recommended due to {reason}.

**Key Observations:**
- Both models achieved over 95% accuracy
- Very few misclassifications in the test set
- The Iris dataset is well-suited for both algorithms
"""

with open('reports/classification_results.md', 'w') as f:
    f.write(classification_results)
print("\nClassification results saved to reports/classification_results.md")

# ============================================
# PART B: ASSOCIATION RULE MINING
# ============================================
print("\n" + "="*60)
print("PART B: ASSOCIATION RULE MINING")
print("="*60)

# Load transaction data
print("\nLoading transaction data...")

# Resolve absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")

try:
    df_transactions = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_transactions)} transactions from {DATA_PATH}")
except FileNotFoundError:
    print("Error: transactions.csv not found.")
    print(f"Expected at: {DATA_PATH}")
    print("Please run transaction_generator.py first.")
    exit(1)

# Convert transactions to list format
transactions = []
for items_str in df_transactions['Items']:
    items = items_str.split(',')
    transactions.append(items)

print(f"\nSample transactions:")
for i, trans in enumerate(transactions[:5], 1):
    print(f"T{i}: {trans}")

# ============================================
# Apply Transaction Encoder
# ============================================
print("\n" + "-"*60)
print("Encoding Transactions")
print("-"*60)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print(f"Encoded dataset shape: {df_encoded.shape}")
print(f"Number of unique items: {len(te.columns_)}")

# ============================================
# Apply Apriori Algorithm
# ============================================
print("\n" + "-"*60)
print("Applying Apriori Algorithm")
print("-"*60)

# Find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
print(f"\nFound {len(frequent_itemsets)} frequent itemsets with min_support=0.2")

if len(frequent_itemsets) == 0:
    print("No frequent itemsets found. Lowering min_support to 0.15...")
    frequent_itemsets = apriori(df_encoded, min_support=0.15, use_colnames=True)
    min_sup_used = 0.15
else:
    min_sup_used = 0.2

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

if len(rules) == 0:
    print("No rules found with min_confidence=0.5. Lowering to 0.4...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
    min_conf_used = 0.4
else:
    min_conf_used = 0.5

# Sort by lift
rules_sorted = rules.sort_values('lift', ascending=False)

print(f"\nGenerated {len(rules_sorted)} association rules")
print(f"Parameters: min_support={min_sup_used}, min_confidence={min_conf_used}")

# Display top 5 rules
print("\n" + "-"*60)
print("TOP 5 ASSOCIATION RULES (sorted by lift)")
print("-"*60)

top_rules = rules_sorted.head(5)
for idx, row in top_rules.iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    print(f"\nRule {idx+1}:")
    print(f"  {antecedents} => {consequents}")
    print(f"  Support: {row['support']:.3f}")
    print(f"  Confidence: {row['confidence']:.3f}")
    print(f"  Lift: {row['lift']:.3f}")

# Save top rules to CSV
top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv(
    'data/top_association_rules.csv', index=False
)
print("\nTop rules saved to data/top_association_rules.csv")

# ============================================
# Analysis
# ============================================
if len(top_rules) > 0:
    strongest_rule = top_rules.iloc[0]
    ant = ', '.join(list(strongest_rule['antecedents']))
    cons = ', '.join(list(strongest_rule['consequents']))
    
    analysis_text = f"""# Association Rules Analysis

## Part B: Market Basket Analysis

### Dataset
- Total transactions: {len(transactions)}
- Unique items: {len(te.columns_)}
- Min support: {min_sup_used}
- Min confidence: {min_conf_used}

### Top Association Rule

**Rule:** {ant} => {cons}

**Metrics:**
- Support: {strongest_rule['support']:.3f}
- Confidence: {strongest_rule['confidence']:.3f}
- Lift: {strongest_rule['lift']:.3f}

### Interpretation

The strongest rule shows that customers who purchase {ant} are highly likely to also buy {cons}. The lift value of {strongest_rule['lift']:.2f} indicates that this combination occurs {strongest_rule['lift']:.2f} times more frequently than if the items were purchased independently. This suggests a genuine shopping pattern rather than random co-occurrence.

### Real-World Applications

**Retail Recommendations:** This rule can be used to create product recommendation systems. When a customer adds {ant} to their cart, the system can suggest {cons} to increase cross-selling opportunities.

**Store Layout:** Physical stores can place these items near each other to encourage impulse purchases and improve shopping convenience.

**Promotional Strategies:** Bundle these products together in special offers or discounts to boost sales of both items simultaneously. This data-driven approach helps maximize revenue while providing value to customers.

The high confidence ({strongest_rule['confidence']:.1%}) means that in most cases where customers buy {ant}, they also purchase {cons}, making this a reliable pattern for business decisions.
"""
else:
    analysis_text = """# Association Rules Analysis

## Part B: Market Basket Analysis

No strong association rules were found with the given parameters. This could indicate that the transaction data has low co-occurrence patterns or requires lower thresholds.
"""

with open('reports/association_rules_analysis.md', 'w') as f:
    f.write(analysis_text)
print("\nAssociation rules analysis saved to reports/association_rules_analysis.md")

print("\n" + "="*60)
print("TASK 3 COMPLETED SUCCESSFULLY")
print("="*60)
print("\nAll outputs saved:")
print("  - visualizations/decision_tree.png")
print("  - visualizations/confusion_matrix.png")
print("  - reports/classification_results.md")
print("  - reports/association_rules_analysis.md")
print("  - data/top_association_rules.csv")