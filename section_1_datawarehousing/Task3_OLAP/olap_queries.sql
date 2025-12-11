--  Task 3: OLAP Queries and Analysis



-- Query 1: ROLL-UP - Total Sales by Country and Quarter

-- This query aggregates sales data at a higher level (country and quarter)
-- to provide strategic insights into regional performance over time

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
    c.Country, t.Year, t.Quarter;


-- Query 2: DRILL-DOWN - Sales Details for UK by Month

-- This query provides detailed monthly breakdown for a specific country (UK)
-- allowing deeper analysis of sales patterns within that market

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
    c.Country = 'UK'
GROUP BY 
    c.Country, t.Year, t.Month
ORDER BY 
    t.Year, t.Month;



-- Query 3: SLICE - Total Sales for Electronics Category

-- This query filters data for a specific dimension (product category)
-- Note: Assumes product categorization was added during ETL or generation
-- For synthetic data, we categorize based on StockCode patterns

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
    TotalSalesAmount DESC;



-- Additional Query: Electronics Category Only (Slice Focus)

-- Detailed view of only Electronics category sales

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
WHERE 
    s.StockCode IN ('LAMP', 'CLOCK')  -- Electronics category
GROUP BY 
    c.Country, t.Year, t.Quarter
ORDER BY 
    TotalSalesAmount DESC;