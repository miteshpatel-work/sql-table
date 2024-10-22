-- Create Customers Table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    date_of_birth DATE,
    email VARCHAR(100),
    phone_number VARCHAR(20),
    address VARCHAR(200),
    city VARCHAR(50),
    state VARCHAR(50),
    zip_code VARCHAR(10),
    country VARCHAR(50),
    date_joined DATE
);

-- Create Accounts Table
CREATE TABLE accounts (
    account_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    account_type VARCHAR(20), -- e.g. Savings, Checking, etc.
    balance DECIMAL(15, 2),
    interest_rate DECIMAL(5, 2),
    account_status VARCHAR(20), -- e.g. Active, Closed, etc.
    date_opened DATE,
    date_closed DATE
);

-- Create Transactions Table
CREATE TABLE transactions (
    transaction_id INT PRIMARY KEY,
    account_id INT REFERENCES accounts(account_id),
    transaction_date DATE,
    transaction_type VARCHAR(20), -- e.g. Deposit, Withdrawal, Transfer, etc.
    amount DECIMAL(15, 2),
    description VARCHAR(200)
);

-- Create Loans Table
CREATE TABLE loans (
    loan_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    loan_type VARCHAR(20), -- e.g. Personal, Mortgage, Auto, etc.
    loan_amount DECIMAL(15, 2),
    interest_rate DECIMAL(5, 2),
    loan_term INT, -- in months
    loan_status VARCHAR(20), -- e.g. Active, Closed, Defaulted
    start_date DATE,
    end_date DATE
);

-- Create Loan Payments Table
CREATE TABLE loan_payments (
    payment_id INT PRIMARY KEY,
    loan_id INT REFERENCES loans(loan_id),
    payment_date DATE,
    payment_amount DECIMAL(15, 2),
    payment_method VARCHAR(20) -- e.g. Bank Transfer, Credit Card, etc.
);

-- Create Credit Cards Table
CREATE TABLE credit_cards (
    card_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    card_number VARCHAR(16),
    card_type VARCHAR(20), -- e.g. Visa, MasterCard, etc.
    credit_limit DECIMAL(15, 2),
    interest_rate DECIMAL(5, 2),
    card_status VARCHAR(20), -- e.g. Active, Blocked
    issue_date DATE,
    expiration_date DATE
);

-- Create Credit Card Transactions Table
CREATE TABLE credit_card_transactions (
    transaction_id INT PRIMARY KEY,
    card_id INT REFERENCES credit_cards(card_id),
    transaction_date DATE,
    merchant VARCHAR(100),
    transaction_amount DECIMAL(15, 2),
    transaction_type VARCHAR(20), -- e.g. Purchase, Cash Withdrawal
    description VARCHAR(200)
);

-- Create Investments Table
CREATE TABLE investments (
    investment_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    investment_type VARCHAR(20), -- e.g. Stocks, Bonds, Mutual Funds
    amount_invested DECIMAL(15, 2),
    date_invested DATE,
    current_value DECIMAL(15, 2)
);

-- Create Dividends Table
CREATE TABLE dividends (
    dividend_id INT PRIMARY KEY,
    investment_id INT REFERENCES investments(investment_id),
    dividend_date DATE,
    dividend_amount DECIMAL(15, 2)
);

-- Create Bill Payments Table
CREATE TABLE bill_payments (
    payment_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    bill_type VARCHAR(50), -- e.g. Electricity, Water, Rent
    payment_date DATE,
    payment_amount DECIMAL(15, 2),
    payment_method VARCHAR(20), -- e.g. Direct Debit, Credit Card
    description VARCHAR(200)
);

-- Insert sample data into customers
INSERT INTO customers VALUES
    (1, 'John', 'Doe', '1985-04-23', 'john.doe@example.com', '555-1234', '123 Elm St', 'New York', 'NY', '10001', 'USA', '2020-01-15'),
    (2, 'Jane', 'Smith', '1990-07-14', 'jane.smith@example.com', '555-5678', '456 Oak St', 'Los Angeles', 'CA', '90001', 'USA', '2019-06-20'),
    (3, 'Bob', 'Johnson', '1982-11-30', 'bob.j@example.com', '555-9876', '789 Maple St', 'Chicago', 'IL', '60601', 'USA', '2018-03-25'),
    (4, 'Alice', 'Williams', '1978-05-10', 'alice.w@example.com', '555-8765', '321 Pine St', 'Boston', 'MA', '02108', 'USA', '2021-07-12'),
    (5, 'Charlie', 'Brown', '1995-12-01', 'charlie.b@example.com', '555-6543', '654 Cedar St', 'San Francisco', 'CA', '94102', 'USA', '2020-11-20'),
    (6, 'David', 'Clark', '1987-03-17', 'david.c@example.com', '555-4321', '987 Birch St', 'Houston', 'TX', '77002', 'USA', '2019-08-19'),
    (7, 'Emma', 'Davis', '1992-09-25', 'emma.d@example.com', '555-3214', '654 Willow St', 'Miami', 'FL', '33101', 'USA', '2020-06-11'),
    (8, 'Fiona', 'Garcia', '1983-06-08', 'fiona.g@example.com', '555-6789', '111 Palm St', 'Phoenix', 'AZ', '85001', 'USA', '2021-01-05'),
    (9, 'George', 'Martinez', '1980-12-15', 'george.m@example.com', '555-9873', '222 Redwood St', 'Dallas', 'TX', '75201', 'USA', '2018-11-18'),
    (10, 'Helen', 'Lopez', '1988-07-22', 'helen.l@example.com', '555-5432', '333 Spruce St', 'Philadelphia', 'PA', '19102', 'USA', '2022-05-03');

-- Insert sample data into accounts
INSERT INTO accounts VALUES
    (101, 1, 'Checking', 5000.00, 0.01, 'Active', '2020-01-15', NULL),
    (102, 1, 'Savings', 15000.00, 0.05, 'Active', '2020-01-15', NULL),
    (103, 2, 'Checking', 7000.00, 0.01, 'Active', '2019-06-20', NULL),
    (104, 2, 'Savings', 10000.00, 0.04, 'Active', '2019-06-20', NULL),
    (105, 3, 'Checking', 8000.00, 0.02, 'Active', '2018-03-25', NULL),
    (106, 3, 'Savings', 20000.00, 0.05, 'Active', '2018-03-25', NULL),
    (107, 4, 'Checking', 12000.00, 0.02, 'Active', '2021-07-12', NULL),
    (108, 5, 'Checking', 9000.00, 0.01, 'Active', '2020-11-20', NULL),
    (109, 6, 'Checking', 15000.00, 0.02, 'Active', '2019-08-19', NULL),
    (110, 7, 'Savings', 25000.00, 0.06, 'Active', '2020-06-11', NULL);

-- Insert sample data into transactions
INSERT INTO transactions VALUES
    (1001, 101, '2023-10-01', 'Deposit', 1000.00, 'Salary Deposit'),
    (1002, 101, '2023-10-05', 'Withdrawal', 200.00, 'ATM Withdrawal'),
    (1003, 102, '2023-10-10', 'Transfer', 500.00, 'Transfer to Checking'),
    (1004, 103, '2023-09-15', 'Deposit', 2000.00, 'Bonus'),
    (1005, 104, '2023-09-20', 'Withdrawal', 300.00, 'Online Shopping'),
    (1006, 105, '2023-10-01', 'Deposit', 1500.00, 'Salary Deposit'),
    (1007, 106, '2023-10-05', 'Transfer', 400.00, 'Transfer to Checking'),
    (1008, 107, '2023-10-10', 'Deposit', 3000.00, 'Freelance Income'),
    (1009, 108, '2023-09-15', 'Deposit', 1000.00, 'Salary Deposit'),
    (1010, 109, '2023-09-20', 'Withdrawal', 100.00, 'Gas Station');

-- Insert sample data into loans
INSERT INTO loans VALUES
    (2001, 1, 'Mortgage', 300000.00, 3.50, 360, 'Active', '2020-02-01', '2050-01-31'),
    (2002, 2, 'Auto', 25000.00, 4.00, 60, 'Active', '2021-01-15', '2026-01-14'),
    (2003, 3, 'Personal', 10000.00, 7.00, 24, 'Closed', '2019-03-20', '2021-03-19'),
    (2004, 4, 'Mortgage', 350000.00, 3.75, 360, 'Active', '2021-08-01', '2051-07-31'),
    (2005, 5, 'Auto', 30000.00, 4.50, 72, 'Active', '2020-12-10', '2026-12-09'),
    (2006, 6, 'Personal', 20000.00, 6.50, 36, 'Active', '2019-09-25', '2022-09-24'),
    (2007, 7, 'Mortgage', 400000.00, 3.25, 360, 'Active', '2020-06-15', '2050-06-14'),
    (2008, 8, 'Auto', 18000.00, 4.25, 48, 'Active', '2021-04-01', '2025-04-01'),
    (2009, 9, 'Personal', 15000.00, 7.25, 36, 'Closed', '2018-08-05', '2021-08-04'),
    (2010, 10, 'Mortgage', 250000.00, 3.60, 360, 'Active', '2022-05-01', '2052-04-30');

-- Insert sample data into loan payments
INSERT INTO loan_payments VALUES
    (3001, 2001, '2023-10-01', 1500.00, 'Bank Transfer'),
    (3002, 2002, '2023-10-05', 450.00, 'Credit Card'),
    (3003, 2003, '2021-03-19', 500.00, 'Bank Transfer'),
    (3004, 2004, '2023-10-10', 1700.00, 'Direct Debit'),
    (3005, 2005, '2023-09-15', 600.00, 'Bank Transfer'),
    (3006, 2006, '2022-09-24', 700.00, 'Credit Card'),
    (3007, 2007, '2023-10-01', 1800.00, 'Bank Transfer'),
    (3008, 2008, '2023-09-20', 450.00, 'Direct Debit'),
    (3009, 2009, '2021-08-04', 500.00, 'Credit Card'),
    (3010, 2010, '2023-10-15', 1600.00, 'Bank Transfer');

-- Insert sample data into credit_cards
INSERT INTO credit_cards VALUES
    (4001, 1, '1234567812345678', 'Visa', 10000.00, 19.99, 'Active', '2020-01-15', '2025-01-15'),
    (4002, 2, '2345678923456789', 'MasterCard', 8000.00, 18.50, 'Active', '2019-06-20', '2024-06-20'),
    (4003, 3, '3456789034567890', 'Visa', 12000.00, 20.00, 'Blocked', '2018-03-25', '2023-03-25'),
    (4004, 4, '4567890145678901', 'Amex', 15000.00, 16.99, 'Active', '2021-07-12', '2026-07-12'),
    (4005, 5, '5678901256789012', 'Discover', 9000.00, 17.50, 'Active', '2020-11-20', '2025-11-20'),
    (4006, 6, '6789012367890123', 'Visa', 11000.00, 19.75, 'Active', '2019-08-19', '2024-08-19'),
    (4007, 7, '7890123478901234', 'MasterCard', 14000.00, 18.25, 'Active', '2020-06-11', '2025-06-11'),
    (4008, 8, '8901234589012345', 'Visa', 10000.00, 19.00, 'Blocked', '2021-01-05', '2026-01-05'),
    (4009, 9, '9012345690123456', 'Amex', 16000.00, 16.75, 'Active', '2018-11-18', '2023-11-18'),
    (4010, 10, '0123456701234567', 'Discover', 13000.00, 17.00, 'Active', '2022-05-03', '2027-05-03');

-- Insert sample data into credit_card_transactions
INSERT INTO credit_card_transactions VALUES
    (5001, 4001, '2023-09-01', 'Amazon', 250.00, 'Purchase', 'Bought Electronics'),
    (5002, 4002, '2023-09-05', 'Walmart', 100.00, 'Purchase', 'Groceries'),
    (5003, 4003, '2023-09-10', 'Best Buy', 500.00, 'Purchase', 'Bought Laptop'),
    (5004, 4004, '2023-09-15', 'Gas Station', 75.00, 'Purchase', 'Gas Refill'),
    (5005, 4005, '2023-09-20', 'Target', 200.00, 'Purchase', 'Household Items'),
    (5006, 4006, '2023-09-25', 'Costco', 300.00, 'Purchase', 'Groceries'),
    (5007, 4007, '2023-09-30', 'Nike', 150.00, 'Purchase', 'Bought Shoes'),
    (5008, 4008, '2023-10-01', 'Apple Store', 600.00, 'Purchase', 'Bought iPhone'),
    (5009, 4009, '2023-10-05', 'Gas Station', 80.00, 'Purchase', 'Gas Refill'),
    (5010, 4010, '2023-10-10', 'Walmart', 120.00, 'Purchase', 'Groceries');

-- Insert sample data into investments
INSERT INTO investments VALUES
    (6001, 1, 'Stocks', 10000.00, '2021-01-15', 12000.00),
    (6002, 2, 'Bonds', 15000.00, '2020-06-20', 16000.00),
    (6003, 3, 'Mutual Funds', 20000.00, '2019-03-25', 22000.00),
    (6004, 4, 'Stocks', 25000.00, '2021-07-12', 28000.00),
    (6005, 5, 'Bonds', 18000.00, '2020-11-20', 20000.00),
    (6006, 6, 'Mutual Funds', 22000.00, '2019-08-19', 25000.00),
    (6007, 7, 'Stocks', 30000.00, '2020-06-11', 34000.00),
    (6008, 8, 'Bonds', 12000.00, '2021-01-05', 14000.00),
    (6009, 9, 'Mutual Funds', 16000.00, '2018-11-18', 18000.00),
    (6010, 10, 'Stocks', 20000.00, '2022-05-03', 22000.00);

-- Insert sample data into dividends
INSERT INTO dividends VALUES
    (7001, 6001, '2023-10-01', 500.00),
    (7002, 6002, '2023-09-15', 300.00),
    (7003, 6003, '2023-08-10', 400.00),
    (7004, 6004, '2023-07-05', 600.00),
    (7005, 6005, '2023-06-20', 200.00),
    (7006, 6006, '2023-05-15', 350.00),
    (7007, 6007, '2023-04-10', 450.00),
    (7008, 6008, '2023-03-05', 250.00),
    (7009, 6009, '2023-02-15', 300.00),
    (7010, 6010, '2023-01-01', 400.00);

-- Insert sample data into bill_payments
INSERT INTO bill_payments VALUES
    (8001, 1, 'Electricity', '2023-10-01', 150.00, 'Direct Debit', 'October electricity bill'),
    (8002, 2, 'Water', '2023-10-05', 100.00, 'Credit Card', 'October water bill'),
    (8003, 3, 'Rent', '2023-10-10', 2000.00, 'Bank Transfer', 'October rent payment'),
    (8004, 4, 'Internet', '2023-09-15', 75.00, 'Direct Debit', 'September internet bill'),
    (8005, 5, 'Gas', '2023-09-20', 50.00, 'Bank Transfer', 'September gas bill'),
    (8006, 6, 'Phone', '2023-09-25', 120.00, 'Credit Card', 'September phone bill'),
    (8007, 7, 'Insurance', '2023-09-30', 300.00, 'Direct Debit', 'September insurance premium'),
    (8008, 8, 'Cable', '2023-10-01', 90.00, 'Credit Card', 'October cable bill'),
    (8009, 9, 'Electricity', '2023-10-05', 160.00, 'Bank Transfer', 'October electricity bill'),
    (8010, 10, 'Water', '2023-10-10', 110.00, 'Direct Debit', 'October water bill');
