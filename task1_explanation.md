# Task 1: Data Warehouse Design - Explanation

## Schema Design Overview

This data warehouse is designed for a retail company that sells products across multiple categories (electronics, clothing, etc.). The warehouse tracks sales transactions, customer information, product details, and time dimensions to support analytical queries.

## Star Schema Components

### Fact Table: SalesFact
The central fact table contains measurable business metrics:
- **sales_amount**: Total revenue from each transaction
- **quantity**: Number of units sold
- **Foreign Keys**: Links to CustomerDim, ProductDim, and TimeDim

### Dimension Tables

**CustomerDim**: Stores customer demographic information
- customer_id (Primary Key)
- customer_name
- gender
- age
- country

**ProductDim**: Contains product catalog details
- product_id (Primary Key)
- product_name
- category
- unit_price

**TimeDim**: Provides time-based analysis capabilities
- time_id (Primary Key)
- date
- month
- quarter
- year

## Why Star Schema Over Snowflake Schema?

Star schema is simpler and requires fewer joins when running queries. This makes analytical queries faster because the database engine does not need to navigate through multiple levels of normalized tables. It is also easier to understand and use for reporting.

## Query Support

This design directly supports the required business queries:

1. **Total sales by product category per quarter**: Join SalesFact with ProductDim (for category) and TimeDim (for quarter), then aggregate sales_amount.

2. **Customer demographics analysis**: CustomerDim contains all demographic fields (age, gender, country) that can be analyzed with sales data.

3. **Inventory trends**: The quantity measure in SalesFact combined with TimeDim allows tracking of sales volume over different time periods.

## Design Decisions

- **Denormalized dimensions**: All dimension attributes are kept in single tables to minimize joins.
- **No surrogate keys complexity**: Simple integer primary keys for easy reference.
- **Time dimension**: Pre-calculated time attributes (month, quarter, year) eliminate the need for date calculations in queries.
- **Category in ProductDim**: Placing category directly in the product dimension avoids creating a separate category dimension (snowflaking).

