# Task 2: ETL Process Implementation

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# DATASET GENERATION (Synthetic Data)
# Generating ~1000 rows of synthetic retail data

def generate_synthetic_data(n_rows=1000):
    """
    Generate synthetic online retail data matching UCI dataset structure
    """
    print("Generating synthetic retail data...")
    
    # Define possible values
    countries = ['UK', 'Germany', 'France', 'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Portugal']
    products = ['MUG', 'PLATE', 'CANDLE', 'BASKET', 'BAG', 'FRAME', 'LAMP', 'CLOCK', 'CUSHION', 'VASE']
    descriptions = {
        'MUG': 'CERAMIC COFFEE MUG',
        'PLATE': 'DINNER PLATE SET',
        'CANDLE': 'SCENTED CANDLE',
        'BASKET': 'WICKER BASKET',
        'BAG': 'SHOPPING BAG',
        'FRAME': 'PHOTO FRAME',
        'LAMP': 'TABLE LAMP',
        'CLOCK': 'WALL CLOCK',
        'CUSHION': 'DECORATIVE CUSHION',
        'VASE': 'GLASS VASE'
    }
    
    # Generate dates over 2 years (Aug 2023 - Aug 2025)
    start_date = datetime(2023, 8, 1)
    end_date = datetime(2025, 8, 12)
    date_range = (end_date - start_date).days
    
    # Generate data
    data = {
        'InvoiceNo': [f'INV{100000 + i//3}' for i in range(n_rows)],  # Multiple items per invoice
        'StockCode': np.random.choice(products, n_rows),
        'Description': [descriptions[code] for code in np.random.choice(products, n_rows)],
        'Quantity': np.random.randint(1, 51, n_rows),
        'InvoiceDate': [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_rows)],
        'UnitPrice': np.round(np.random.uniform(1.0, 100.0, n_rows), 2),
        'CustomerID': np.random.randint(10000, 10100, n_rows),  # 100 unique customers
        'Country': np.random.choice(countries, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values (realistic)
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_rows), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'CustomerID'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'Description'] = np.nan
    
    # Introduce some outliers (negative quantities, zero prices)
    outlier_indices = np.random.choice(df.index, size=int(0.03 * n_rows), replace=False)
    df.loc[outlier_indices[:len(outlier_indices)//2], 'Quantity'] = -np.random.randint(1, 10)
    df.loc[outlier_indices[len(outlier_indices)//2:], 'UnitPrice'] = 0
    
    print(f"Generated {len(df)} rows of synthetic data")
    return df


# EXTRACTION


def extract_data():
    """
    Extract: Load data and handle basic data type conversions
    """
    print("\n=== EXTRACTION PHASE ===")
    
    
    # df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')
    
    # Use generated synthetic data
    df = generate_synthetic_data(1000)
    
    print(f"Initial data loaded: {len(df)} rows")
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    print("Converted InvoiceDate to datetime format")
    
    # Handle missing values
    print(f"Missing values before handling:\n{df.isnull().sum()}")
    
    # Drop rows with missing CustomerID (can't track customer without ID)
    df = df.dropna(subset=['CustomerID'])
    
    # Fill missing descriptions with 'UNKNOWN'
    df['Description'].fillna('UNKNOWN', inplace=True)
    
    # Drop rows with missing dates
    df = df.dropna(subset=['InvoiceDate'])
    
    print(f"Missing values after handling:\n{df.isnull().sum()}")
    print(f"Rows after handling missing values: {len(df)}")
    
    # Ensure correct data types
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['Quantity'] = df['Quantity'].astype(int)
    df['UnitPrice'] = df['UnitPrice'].astype(float)
    
    print("Data types corrected")
    
    return df


# TRANSFORMATION


def transform_data(df):
    """
    Transform: Calculate TotalSales, create summaries, filter dates, remove outliers
    """
    print("\n=== TRANSFORMATION PHASE ===")
    
    # A. Calculate TotalSales column
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    print("Created TotalSales column")
    
    # B. Remove outliers (Quantity < 0 or UnitPrice <= 0)
    rows_before_outlier_removal = len(df)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print(f"Removed outliers: {rows_before_outlier_removal - len(df)} rows removed")
    print(f"Rows after outlier removal: {len(df)}")
    
    # C. Filter for last year (Aug 12, 2024 to Aug 12, 2025)
    current_date = pd.to_datetime('2025-08-12')
    one_year_ago = current_date - pd.DateOffset(years=1)
    
    rows_before_date_filter = len(df)
    df = df[(df['InvoiceDate'] >= one_year_ago) & (df['InvoiceDate'] <= current_date)]
    print(f"Filtered for last year (Aug 12, 2024 - Aug 12, 2025): {len(df)} rows retained")
    
    # D. Create Customer Dimension (summary by CustomerID)
    customer_dim = df.groupby('CustomerID').agg({
        'TotalSales': 'sum',
        'InvoiceNo': 'count',
        'Country': 'first'  # Take first country for each customer
    }).reset_index()
    
    customer_dim.columns = ['CustomerID', 'TotalPurchases', 'TransactionCount', 'Country']
    print(f"Created CustomerDim: {len(customer_dim)} unique customers")
    
    # E. Create Time Dimension
    time_dim = df[['InvoiceDate']].drop_duplicates().copy()
    time_dim['Date'] = time_dim['InvoiceDate'].dt.date
    time_dim['Year'] = time_dim['InvoiceDate'].dt.year
    time_dim['Month'] = time_dim['InvoiceDate'].dt.month
    time_dim['Quarter'] = time_dim['InvoiceDate'].dt.quarter
    time_dim['DayOfWeek'] = time_dim['InvoiceDate'].dt.dayofweek
    time_dim = time_dim.reset_index(drop=True)
    time_dim['TimeID'] = time_dim.index + 1
    print(f"Created TimeDim: {len(time_dim)} unique dates")
    
    # F. Prepare Sales Fact table
    # Merge with TimeDim to get TimeID
    df_merged = df.merge(
        time_dim[['InvoiceDate', 'TimeID']], 
        on='InvoiceDate', 
        how='left'
    )
    
    sales_fact = df_merged[[
        'InvoiceNo', 
        'StockCode', 
        'CustomerID', 
        'TimeID', 
        'Quantity', 
        'UnitPrice', 
        'TotalSales'
    ]].copy()
    
    print(f"Created SalesFact: {len(sales_fact)} transaction records")
    
    return sales_fact, customer_dim, time_dim


# LOADING TO SQLite


def load_to_database(sales_fact, customer_dim, time_dim, db_name='retail_dw.db'):
    """
    Load: Create SQLite database and load transformed data into tables
    """
    print("\n=== LOADING PHASE ===")
    
    # Connect to SQLite database (creates file if doesn't exist)
    conn = sqlite3.connect(db_name)
    print(f"Connected to database: {db_name}")
    
    try:
        # Load CustomerDim
        customer_dim.to_sql('CustomerDim', conn, if_exists='replace', index=False)
        print(f"Loaded CustomerDim: {len(customer_dim)} rows")
        
        # Load TimeDim
        time_dim.to_sql('TimeDim', conn, if_exists='replace', index=False)
        print(f"Loaded TimeDim: {len(time_dim)} rows")
        
        # Load SalesFact
        sales_fact.to_sql('SalesFact', conn, if_exists='replace', index=False)
        print(f"Loaded SalesFact: {len(sales_fact)} rows")
        
        # Verify tables were created
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"\nTables in database: {[table[0] for table in tables]}")
        
        print("\nData successfully loaded to database!")
        
    except Exception as e:
        print(f"Error during loading: {e}")
        
    finally:
        conn.close()
        print("Database connection closed")


# ETL FUNCTION


def run_etl():
    """
    Main ETL function that orchestrates the entire process with logging
    """
    print("="*60)
    print("STARTING ETL PROCESS - RETAIL DATA WAREHOUSE")
    print("="*60)
    
    # EXTRACT
    df_raw = extract_data()
    print(f"\n[LOG] Original dataset size: {len(df_raw)} rows")
    
    # TRANSFORM
    sales_fact, customer_dim, time_dim = transform_data(df_raw)
    
    print(f"\n[LOG] Transformation Summary:")
    print(f"  - SalesFact rows: {len(sales_fact)}")
    print(f"  - CustomerDim rows: {len(customer_dim)}")
    print(f"  - TimeDim rows: {len(time_dim)}")
    
    # LOAD
    load_to_database(sales_fact, customer_dim, time_dim)
    
    print("\n" + "="*60)
    print("ETL PROCESS COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nDatabase file 'retail_dw.db' has been created.")
    print("Tables created: SalesFact, CustomerDim, TimeDim")
    
    return sales_fact, customer_dim, time_dim


# MAIN EXECUTION


if __name__ == "__main__":
    # Run the complete ETL process
    sales_fact, customer_dim, time_dim = run_etl()
    
    # Display sample data
    print("\n=== SAMPLE DATA ===")
    print("\nSalesFact (first 5 rows):")
    print(sales_fact.head())
    print("\nCustomerDim (first 5 rows):")
    print(customer_dim.head())
    print("\nTimeDim (first 5 rows):")
    print(time_dim.head())