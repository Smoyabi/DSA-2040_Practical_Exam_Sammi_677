# Task 3: OLAP Queries and Analysis

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories for outputs if they don't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# DATABASE CONNECTION


def get_connection(db_path=None):
    """
    Connect to SQLite database
    Automatically finds database in Task2_ETL/database/ directory
    """
    if db_path is None:
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build path to database (go up to section_1_datawarehousing, then to Task2_ETL)
        db_path = os.path.join(current_dir, '..', 'Task2_ETL', 'database', 'retail_dw.db')
        
        # Normalize path
        db_path = os.path.normpath(db_path)
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at: {db_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        raise FileNotFoundError(f"Database not found at: {db_path}")
    
    print(f"Connecting to database: {db_path}")
    return sqlite3.connect(db_path)


# QUERY 1: ROLL-UP - Total Sales by Country and Quarter


def query_rollup():
    """
    Roll-up query: Aggregate sales by country and quarter
    Provides high-level strategic view of regional performance
    """
    print("\n=== QUERY 1: ROLL-UP - Sales by Country and Quarter ===")
    
    conn = get_connection()
    
    query = """
    SELECT 
        c.Country,
        t.Year,
        t.Quarter,
        COUNT(DISTINCT s.InvoiceNo) AS TransactionCount,
        SUM(s.Quantity) AS TotalQuantitySold,
        ROUND(SUM(s.TotalSales), 2) AS TotalSalesAmount
    FROM 
        SalesFact s
    JOIN 
        CustomerDim c ON s.CustomerID = c.CustomerID
    JOIN 
        TimeDim t ON s.TimeID = t.TimeID
    GROUP BY 
        c.Country, t.Year, t.Quarter
    ORDER BY 
        c.Country, t.Year, t.Quarter
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nResults: {len(df)} rows")
    print(df.head(10))
    
    return df


# QUERY 2: DRILL-DOWN - Sales Details for UK by Month


def query_drilldown(country='UK'):
    """
    Drill-down query: Detailed monthly breakdown for specific country
    Allows deeper analysis of sales patterns
    """
    print(f"\n=== QUERY 2: DRILL-DOWN - {country} Sales by Month ===")
    
    conn = get_connection()
    
    query = f"""
    SELECT 
        c.Country,
        t.Year,
        t.Month,
        COUNT(DISTINCT s.InvoiceNo) AS TransactionCount,
        COUNT(s.StockCode) AS ItemsSold,
        SUM(s.Quantity) AS TotalQuantity,
        ROUND(SUM(s.TotalSales), 2) AS TotalSalesAmount,
        ROUND(AVG(s.TotalSales), 2) AS AvgSalesPerTransaction
    FROM 
        SalesFact s
    JOIN 
        CustomerDim c ON s.CustomerID = c.CustomerID
    JOIN 
        TimeDim t ON s.TimeID = t.TimeID
    WHERE 
        c.Country = '{country}'
    GROUP BY 
        c.Country, t.Year, t.Month
    ORDER BY 
        t.Year, t.Month
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nResults: {len(df)} rows")
    print(df.head(10))
    
    return df


# QUERY 3: SLICE - Total Sales by Product Category


def query_slice():
    """
    Slice query: Filter by product category dimension
    Focus on specific product segments
    """
    print("\n=== QUERY 3: SLICE - Sales by Product Category ===")
    
    conn = get_connection()
    
    query = """
    SELECT 
        CASE 
            WHEN s.StockCode IN ('MUG', 'PLATE', 'VASE', 'FRAME') THEN 'Home Decor'
            WHEN s.StockCode IN ('LAMP', 'CLOCK') THEN 'Electronics'
            WHEN s.StockCode IN ('BASKET', 'BAG', 'CUSHION') THEN 'Textiles'
            WHEN s.StockCode = 'CANDLE' THEN 'Fragrances'
            ELSE 'Other'
        END AS ProductCategory,
        COUNT(DISTINCT s.InvoiceNo) AS TransactionCount,
        SUM(s.Quantity) AS TotalQuantitySold,
        ROUND(SUM(s.TotalSales), 2) AS TotalSalesAmount,
        ROUND(AVG(s.UnitPrice), 2) AS AvgUnitPrice
    FROM 
        SalesFact s
    GROUP BY 
        ProductCategory
    ORDER BY 
        TotalSalesAmount DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nResults: {len(df)} rows")
    print(df)
    
    return df


# VISUALIZATION: Sales by Country


def visualize_sales_by_country():
    """
    Create bar chart visualization of total sales by country
    Saves as PNG image
    """
    print("\n=== CREATING VISUALIZATION: Sales by Country ===")
    
    conn = get_connection()
    
    # Query to get total sales by country
    query = """
    SELECT 
        c.Country,
        COUNT(DISTINCT s.InvoiceNo) AS TransactionCount,
        SUM(s.Quantity) AS TotalQuantity,
        ROUND(SUM(s.TotalSales), 2) AS TotalSalesAmount
    FROM 
        SalesFact s
    JOIN 
        CustomerDim c ON s.CustomerID = c.CustomerID
    GROUP BY 
        c.Country
    ORDER BY 
        TotalSalesAmount DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Total Sales Amount by Country
    colors = sns.color_palette("husl", len(df))
    bars1 = ax1.bar(df['Country'], df['TotalSalesAmount'], color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Country', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Sales Amount', fontsize=12, fontweight='bold')
    ax1.set_title('Total Sales Amount by Country', fontsize=14, fontweight='bold', pad=20)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Transaction Count by Country
    bars2 = ax2.bar(df['Country'], df['TransactionCount'], color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Country', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
    ax2.set_title('Transaction Count by Country', fontsize=14, fontweight='bold', pad=20)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'visualizations/sales_by_country.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_path}")
    
    plt.show()
    
    return df


# ADDITIONAL VISUALIZATION: Quarterly Trends


def visualize_quarterly_trends():
    """
    Create line chart showing sales trends over quarters
    """
    print("\n=== CREATING VISUALIZATION: Quarterly Sales Trends ===")
    
    conn = get_connection()
    
    query = """
    SELECT 
        t.Year,
        t.Quarter,
        ROUND(SUM(s.TotalSales), 2) AS TotalSalesAmount
    FROM 
        SalesFact s
    JOIN 
        TimeDim t ON s.TimeID = t.TimeID
    GROUP BY 
        t.Year, t.Quarter
    ORDER BY 
        t.Year, t.Quarter
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Create quarter label
    df['Quarter_Label'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['Quarter_Label'], df['TotalSalesAmount'], 
             marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    plt.xlabel('Quarter', fontsize=12, fontweight='bold')
    plt.ylabel('Total Sales Amount', fontsize=12, fontweight='bold')
    plt.title('Sales Trends by Quarter (Last Year)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_path = 'visualizations/quarterly_trends.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_path}")
    
    plt.show()

# GENERATE ANALYSIS REPORT DATA


def generate_report_data():
    """
    Generate summary statistics for analysis report
    """
    print("\n=== GENERATING REPORT DATA ===")
    
    conn = get_connection()
    
    # Overall statistics
    overall_query = """
    SELECT 
        COUNT(DISTINCT s.InvoiceNo) AS TotalTransactions,
        COUNT(DISTINCT s.CustomerID) AS TotalCustomers,
        SUM(s.Quantity) AS TotalItemsSold,
        ROUND(SUM(s.TotalSales), 2) AS TotalRevenue,
        ROUND(AVG(s.TotalSales), 2) AS AvgTransactionValue
    FROM SalesFact s
    """
    
    overall_stats = pd.read_sql_query(overall_query, conn)
    
    # Top 3 countries
    top_countries_query = """
    SELECT 
        c.Country,
        ROUND(SUM(s.TotalSales), 2) AS TotalSales
    FROM SalesFact s
    JOIN CustomerDim c ON s.CustomerID = c.CustomerID
    GROUP BY c.Country
    ORDER BY TotalSales DESC
    LIMIT 3
    """
    
    top_countries = pd.read_sql_query(top_countries_query, conn)
    
    # Top category
    top_category_query = """
    SELECT 
        CASE 
            WHEN s.StockCode IN ('MUG', 'PLATE', 'VASE', 'FRAME') THEN 'Home Decor'
            WHEN s.StockCode IN ('LAMP', 'CLOCK') THEN 'Electronics'
            WHEN s.StockCode IN ('BASKET', 'BAG', 'CUSHION') THEN 'Textiles'
            WHEN s.StockCode = 'CANDLE' THEN 'Fragrances'
            ELSE 'Other'
        END AS ProductCategory,
        ROUND(SUM(s.TotalSales), 2) AS TotalSales
    FROM SalesFact s
    GROUP BY ProductCategory
    ORDER BY TotalSales DESC
    LIMIT 1
    """
    
    top_category = pd.read_sql_query(top_category_query, conn)
    
    conn.close()
    
    print("\n=== SUMMARY STATISTICS ===")
    print("\nOverall Statistics:")
    print(overall_stats.to_string(index=False))
    print("\nTop 3 Countries by Sales:")
    print(top_countries.to_string(index=False))
    print("\nTop Product Category:")
    print(top_category.to_string(index=False))
    
    return overall_stats, top_countries, top_category


# MAIN EXECUTION


def main():
    """
    Main function to execute all OLAP queries and generate visualizations
    """
    print("="*70)
    print("OLAP QUERIES AND ANALYSIS - RETAIL DATA WAREHOUSE")
    print("="*70)
    
    # Execute all queries
    rollup_df = query_rollup()
    drilldown_df = query_drilldown('UK')
    slice_df = query_slice()
    
    # Create visualizations
    sales_by_country = visualize_sales_by_country()
    visualize_quarterly_trends()
    
    # Generate report data
    overall_stats, top_countries, top_category = generate_report_data()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - visualizations/sales_by_country.png")
    print("  - visualizations/quarterly_trends.png")
    print("\nNext step: Write analysis report based on these insights")
    print("Report should be 200-300 words discussing:")
    print("  - Top-selling countries and trends")
    print("  - Product category performance")
    print("  - How the data warehouse supports decision-making")
    print("  - Impact of synthetic data on realism")

if __name__ == "__main__":
    main()