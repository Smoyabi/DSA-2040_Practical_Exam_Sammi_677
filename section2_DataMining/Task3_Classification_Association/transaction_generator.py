"""
Transaction Data Generator
Generates synthetic retail transaction data for association rule mining
"""

import pandas as pd
import random
import os

# Set random seed for reproducibility
random.seed(42)

# Create data directory
os.makedirs('data', exist_ok=True)

# Item pool
items_pool = [
    'milk', 'bread', 'eggs', 'beer', 'diapers', 
    'butter', 'chips', 'cheese', 'coffee', 'tea',
    'juice', 'cereal', 'yogurt', 'chicken', 'pasta',
    'rice', 'tomatoes', 'onions', 'apples', 'bananas'
]

# Define frequent patterns to simulate realistic shopping behavior
patterns = [
    ['milk', 'bread', 'eggs'],           # Breakfast combo
    ['beer', 'chips'],                    # Snack combo
    ['diapers', 'milk', 'baby_wipes'],   # Baby products (adding baby_wipes)
    ['pasta', 'tomatoes', 'cheese'],     # Italian meal
    ['coffee', 'milk'],                   # Coffee drinkers
    ['chicken', 'rice', 'onions'],       # Dinner combo
    ['cereal', 'milk', 'bananas'],       # Cereal breakfast
    ['bread', 'butter', 'cheese']        # Sandwich items
]

# Add baby_wipes to item pool
if 'baby_wipes' not in items_pool:
    items_pool.append('baby_wipes')

def generate_transaction():
    """
    Generate a single transaction with 3-8 items
    70% chance of including a pattern, 30% random items
    """
    transaction_size = random.randint(3, 8)
    
    # 70% chance to start with a common pattern
    if random.random() < 0.7 and patterns:
        pattern = random.choice(patterns)
        items = pattern.copy()
        
        # Add additional random items to reach transaction size
        remaining = transaction_size - len(items)
        if remaining > 0:
            additional = random.sample(items_pool, min(remaining, len(items_pool)))
            items.extend(additional)
    else:
        # Pure random transaction
        items = random.sample(items_pool, min(transaction_size, len(items_pool)))
    
    # Remove duplicates and return
    return list(set(items))

# Generate 40 transactions
transactions = []
for i in range(40):
    trans = generate_transaction()
    transactions.append(trans)

# Convert to DataFrame format
# Each row is a transaction, columns are items (comma-separated)
df_transactions = pd.DataFrame({
    'TransactionID': [f'T{str(i+1).zfill(3)}' for i in range(len(transactions))],
    'Items': [','.join(sorted(trans)) for trans in transactions]
})

# Save to CSV
output_path = 'data/transactions.csv'
df_transactions.to_csv(output_path, index=False)

print(f"Generated {len(transactions)} transactions")
print(f"Saved to: {output_path}")
print(f"\nSample transactions:")
print(df_transactions.head(10))
print(f"\nTransaction size range: {min([len(t) for t in transactions])} to {max([len(t) for t in transactions])} items")