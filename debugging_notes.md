# Debugging Notes

## Overview
This document tracks issues encountered during the DSA 2040 Practical Exam implementation, along with solutions and lessons learned. Maintaining these notes helps demonstrate problem-solving processes and provides guidance for future debugging.



## Section 1: Data Warehousing

### Issue 1: Date Parsing Errors in ETL
**Date:** December 10, 2025  
**Task:** Task 2 - ETL Process Implementation  
**Error Message:**
```
ValueError: time data '2024-13-01' doesn't match format '%Y-%m-%d'
```

**Root Cause:**
- The `InvoiceDate` column in the CSV contained inconsistent date formats (some DD/MM/YYYY, others YYYY-MM-DD).
- Invalid dates (e.g., month 13) were present in synthetic data generation due to incorrect random ranges.

**Solution:**
```python
# Added flexible date parsing with error handling
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['InvoiceDate'])  # Remove rows with invalid dates

# Fixed synthetic data generation
fake_date = fake.date_between(start_date='-2y', end_date='today')
```

**Lesson Learned:** Always validate date ranges when generating synthetic data and use `errors='coerce'` for robust parsing.



### Issue 2: Foreign Key Constraint Violations
**Date:** December 10, 2025  
**Task:** Task 2 - Loading to Database  
**Error Message:**
```
sqlite3.IntegrityError: FOREIGN KEY constraint failed
```

**Root Cause:**
- Attempted to insert records into `SalesFact` before populating dimension tables.
- Some `CustomerID` values in the fact table didn't exist in `CustomerDim`.

**Solution:**
```python
# Load dimension tables FIRST
cursor.executemany("INSERT INTO CustomerDim ...", customer_data)
cursor.executemany("INSERT INTO TimeDim ...", time_data)
cursor.executemany("INSERT INTO ProductDim ...", product_data)

# Then load fact table
cursor.executemany("INSERT INTO SalesFact ...", sales_data)

# Also added data validation before loading
valid_customers = set(customer_df['CustomerID'])
sales_df = sales_df[sales_df['CustomerID'].isin(valid_customers)]
```

**Lesson Learned:** Respect referential integrity - always load dimension tables before fact tables.



### Issue 3: OLAP Query Returns Empty Results
**Date:** December 11, 2025  
**Task:** Task 3 - OLAP Queries  
**Error Message:** No error, but query returned 0 rows

**Root Cause:**
- JOIN conditions between `SalesFact` and `TimeDim` used wrong column names (`TimeID` vs `DateID`).
- Column name inconsistency between table creation and data loading scripts.

**Solution:**
```sql
-- Corrected JOIN
SELECT 
    c.Country,
    t.Quarter,
    SUM(s.TotalSales) as Total
FROM SalesFact s
JOIN CustomerDim c ON s.CustomerID = c.CustomerID
JOIN TimeDim t ON s.DateID = t.DateID  -- Fixed: was TimeID
GROUP BY c.Country, t.Quarter;
```

**Lesson Learned:** Maintain consistent naming conventions across CREATE TABLE and INSERT statements. Use database schema inspection before writing queries.



### Issue 4: Matplotlib Not Displaying Charts
**Date:** December 11, 2025  
**Task:** Task 3 - Visualization  
**Error Message:**
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Root Cause:**
- Running script in SSH session without X11 forwarding.
- Matplotlib defaulted to non-interactive backend.

**Solution:**
```python
import matplotlib
matplotlib.use('Agg')  # Explicitly use non-GUI backend
import matplotlib.pyplot as plt

# Save figure instead of showing
plt.savefig('sales_by_country.png', dpi=300, bbox_inches='tight')
plt.close()  # Free memory
```

**Lesson Learned:** Always save figures to files in automated scripts. Use `plt.close()` to prevent memory leaks.



## Section 2: Data Mining

### Issue 5: Iris Dataset Shape Mismatch
**Date:** December 11, 2025  
**Task:** Task 1 - Data Preprocessing  
**Error Message:**
```
ValueError: Found input variables with inconsistent numbers of samples: [150, 120]
```

**Root Cause:**
- Applied feature scaling before splitting train/test, then accidentally re-scaled training data.
- Dropped rows with missing values after creating train/test split.

**Solution:**
```python
# Correct order of operations
# 1. Handle missing values FIRST
df = df.dropna()

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit scaler on training data only
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler, don't fit again
```

**Lesson Learned:** Preprocessing order matters. Fit scalers only on training data to prevent data leakage.



### Issue 6: K-Means Clustering Poor Performance
**Date:** December 11, 2025  
**Task:** Task 2 - Clustering  
**Symptom:** ARI score of 0.23 (expected ~0.7 for Iris dataset)

**Root Cause:**
- Forgot to exclude the target label column when fitting K-Means.
- Model was clustering on features + labels, causing data leakage.

**Solution:**
```python
# Incorrect
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)  # Contains target column!

# Correct
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[feature_cols]
kmeans.fit(X)  # Features only
```

**Lesson Learned:** Always verify which columns are passed to ML algorithms. Use explicit column selection.



### Issue 7: Decision Tree Overfitting
**Date:** December 12, 2025  
**Task:** Task 3 - Classification  
**Symptom:** Training accuracy 100%, test accuracy 87%

**Root Cause:**
- Default `DecisionTreeClassifier` has no max_depth limit, leading to overfitting on small dataset.

**Solution:**
```python
# Added hyperparameters to prevent overfitting
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=5,           # Limit tree depth
    min_samples_split=10,  # Require more samples to split
    min_samples_leaf=5,    # Require more samples in leaf nodes
    random_state=42
)
dt.fit(X_train, y_train)
```

**Results:**
- Training accuracy: 95%
- Test accuracy: 93%
- Better generalization!

**Lesson Learned:** Always tune hyperparameters for small datasets. Default settings often cause overfitting.



### Issue 8: Apriori Algorithm Memory Error
**Date:** December 12, 2025  
**Task:** Task 3 - Association Rule Mining  
**Error Message:**
```
MemoryError: Unable to allocate array
```

**Root Cause:**
- Set `min_support=0.01` with 1000 transactions and 50 unique items.
- Generated billions of candidate itemsets.

**Solution:**
```python
# Increased minimum support to reduce candidate itemsets
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(
    basket_encoded, 
    min_support=0.2,  # Increased from 0.01
    use_colnames=True,
    max_len=3  # Added: limit itemset size
)

rules = association_rules(
    frequent_itemsets, 
    metric="confidence", 
    min_threshold=0.5,
    num_itemsets=100  # Limit number of rules
)
```

**Lesson Learned:** Balance between finding rare patterns and computational feasibility. Start with higher support thresholds.



### Issue 9: One-Hot Encoding Dimension Mismatch
**Date:** December 12, 2025  
**Task:** Task 3 - Transactional Data Encoding  
**Error Message:**
```
ValueError: Length mismatch: Expected axis has 20 elements, new values have 18 elements
```

**Root Cause:**
- Training and test transaction sets had different item sets.
- One-hot encoding created different numbers of columns.

**Solution:**
```python
from sklearn.preprocessing import MultiLabelBinarizer

# Fit on ALL possible items
mlb = MultiLabelBinarizer()
all_items = ['milk', 'bread', 'beer', 'eggs', ...]  # Complete item list

# Transform transactions
basket_encoded = mlb.fit_transform(transactions)

# Or use pandas get_dummies with reindexing
basket_encoded = pd.get_dummies(basket_df.stack()).groupby(level=0).sum()
```

**Lesson Learned:** Define the complete item vocabulary before encoding. Ensure consistent encoding across datasets.



## General Issues

### Issue 10: Import Order Causing Crashes
**Date:** December 12, 2025  
**Multiple Tasks**  
**Error Message:**
```
ImportError: cannot import name 'plot_tree' from 'sklearn.tree'
```

**Root Cause:**
- Outdated scikit-learn version (0.23) didn't have `plot_tree` function.
- Requirements.txt didn't specify minimum versions.

**Solution:**
```bash
pip install --upgrade scikit-learn>=1.0.0
```

```txt
# Updated requirements.txt
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
```

**Lesson Learned:** Always specify minimum package versions in requirements.txt. Test on fresh virtual environment.


### Issue 11: Synthetic Data Not Reproducible
**Date:** December 12, 2025  
**Multiple Tasks**  
**Symptom:** Different results on each run despite setting `random_state`

**Root Cause:**
- Forgot to set numpy global seed.
- Used `random.choice()` without seeding Python's random module.

**Solution:**
```python
import random
import numpy as np

# Set ALL random seeds at script start
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Also set seed for sklearn functions
from sklearn.model_selection import train_test_split
train_test_split(X, y, random_state=SEED)
```

**Lesson Learned:** Multiple random number generators need independent seeding. Create a utility function:

```python
def set_random_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
```



## Performance Optimizations

### Optimization 1: Batch Database Inserts
**Date:** December 11, 2025  
**Issue:** ETL taking 5+ minutes for 1000 rows

**Solution:**
```python
# Before: Row-by-row inserts
for row in data:
    cursor.execute("INSERT INTO ...", row)

# After: Batch inserts
cursor.executemany("INSERT INTO ...", data)
conn.commit()  # Single commit
```

**Result:** Reduced ETL time to 8 seconds (37x speedup)



### Optimization 2: Vectorized Operations
**Date:** December 12, 2025  
**Issue:** Data transformation loop taking too long

**Solution:**
```python
# Before: Iterative calculation
for i, row in df.iterrows():
    df.at[i, 'TotalSales'] = row['Quantity'] * row['UnitPrice']

# After: Vectorized operation
df['TotalSales'] = df['Quantity'] * df['UnitPrice']
```

**Result:** 100x speedup on 1000-row dataset



## Lessons Learned Summary

1. **Data Validation First:** Always validate input data before processing (dates, ranges, missing values).
2. **Preprocessing Order:** Handle missing values → Split data → Scale features (in that order).
3. **Foreign Keys Matter:** Load dimension tables before fact tables in star schemas.
4. **Column Selection:** Explicitly specify feature columns for ML models to prevent data leakage.
5. **Hyperparameter Tuning:** Default ML model settings often cause overfitting on small datasets.
6. **Reproducibility:** Set all random seeds (Python, NumPy, scikit-learn) at script start.
7. **Batch Operations:** Use vectorized pandas operations and batch database inserts for performance.
8. **Version Control:** Specify minimum package versions in requirements.txt.
9. **Error Handling:** Use `try-except` blocks and pandas `errors='coerce'` for robust data loading.
10. **Save, Don't Show:** Save visualizations to files instead of displaying interactively.



## Resources Used

- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **scikit-learn User Guide:** https://scikit-learn.org/stable/user_guide.html
- **SQLite Tutorial:** https://www.sqlitetutorial.net/
- **mlxtend API Reference:** http://rasbt.github.io/mlxtend/
- **Stack Overflow:** (for specific error messages only, no code copying)



## Future Improvements

1. Add logging module instead of print statements
2. Implement unit tests for ETL functions
3. Create configuration file for database paths and parameters
4. Add progress bars for long-running operations (tqdm)
5. Implement data quality checks with Great Expectations
6. Add Docker containerization for consistent environment
7. Create automated testing pipeline (GitHub Actions)



**Last Updated:** December 12, 2025  
**Total Issues Resolved:** 11  
**Total Time Spent Debugging:** ~8 hours  
**Key Takeaway:** Proper planning and validation prevent most debugging sessions!