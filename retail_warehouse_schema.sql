-- Retail Data Warehouse Schema

-- Dimension Table: Customer
CREATE TABLE CustomerDim (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    gender TEXT,
    age INTEGER,
    country TEXT
);

-- Dimension Table: Product
CREATE TABLE ProductDim (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL
);

-- Dimension Table: Time
CREATE TABLE TimeDim (
    time_id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    month INTEGER,
    quarter INTEGER,
    year INTEGER
);

-- Fact Table: Sales
CREATE TABLE SalesFact (
    sales_amount REAL NOT NULL,
    quantity INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    time_id INTEGER NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES CustomerDim(customer_id),
    FOREIGN KEY (product_id) REFERENCES ProductDim(product_id),
    FOREIGN KEY (time_id) REFERENCES TimeDim(time_id)
);