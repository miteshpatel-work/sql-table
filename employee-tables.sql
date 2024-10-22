-- Create departments table
CREATE OR REPLACE TABLE departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(50),
    location VARCHAR(100),
    budget DECIMAL(15,2)
);

-- Create employees table with department reference
CREATE OR REPLACE TABLE employees (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department_id INT REFERENCES departments(department_id),
    salary DECIMAL(10,2),
    hire_date DATE,
    manager_id INT REFERENCES employees(employee_id)
);

-- Create projects table
CREATE OR REPLACE TABLE projects (
    project_id INT PRIMARY KEY,
    project_name VARCHAR(100),
    department_id INT REFERENCES departments(department_id),
    start_date DATE,
    end_date DATE,
    budget DECIMAL(12,2),
    status VARCHAR(20)
);

-- Create employee_projects (junction table for many-to-many relationship)
CREATE OR REPLACE TABLE employee_projects (
    employee_id INT REFERENCES employees(employee_id),
    project_id INT REFERENCES projects(project_id),
    role VARCHAR(50),
    hours_allocated INT,
    PRIMARY KEY (employee_id, project_id)
);

-- Create salaries_history table
CREATE OR REPLACE TABLE salary_history (
    history_id INT PRIMARY KEY,
    employee_id INT REFERENCES employees(employee_id),
    salary_amount DECIMAL(10,2),
    effective_date DATE,
    end_date DATE
);

-- Insert sample data into departments
INSERT INTO departments VALUES
    (1, 'Engineering', 'New York', 1000000.00),
    (2, 'Marketing', 'Los Angeles', 750000.00),
    (3, 'HR', 'Chicago', 500000.00),
    (4, 'Sales', 'Boston', 850000.00),
    (5, 'Research', 'Seattle', 900000.00);

-- Insert sample data into employees
INSERT INTO employees VALUES
    (1, 'John', 'Doe', 1, 85000.00, '2022-01-15', NULL),
    (2, 'Jane', 'Smith', 2, 75000.00, '2021-06-20', 1),
    (3, 'Bob', 'Johnson', 1, 95000.00, '2020-03-10', 1),
    (4, 'Alice', 'Brown', 3, 65000.00, '2023-02-01', 2),
    (5, 'Charlie', 'Wilson', 2, 70000.00, '2022-11-30', 2),
    (6, 'Eva', 'Davis', 4, 80000.00, '2021-09-15', 3),
    (7, 'Frank', 'Miller', 5, 90000.00, '2022-07-22', 3);

-- Insert sample data into projects
INSERT INTO projects VALUES
    (1, 'Mobile App Development', 1, '2023-01-01', '2023-12-31', 500000.00, 'In Progress'),
    (2, 'Marketing Campaign', 2, '2023-03-15', '2023-09-30', 250000.00, 'Completed'),
    (3, 'HR System Update', 3, '2023-06-01', '2024-01-31', 150000.00, 'In Progress'),
    (4, 'Sales Analytics', 4, '2023-04-01', '2023-12-31', 300000.00, 'In Progress'),
    (5, 'Research Initiative', 5, '2023-07-01', '2024-06-30', 450000.00, 'Planning');

-- Insert sample data into employee_projects
INSERT INTO employee_projects VALUES
    (1, 1, 'Project Lead', 40),
    (2, 2, 'Manager', 30),
    (3, 1, 'Developer', 40),
    (4, 3, 'Coordinator', 20),
    (5, 2, 'Designer', 25),
    (6, 4, 'Analyst', 35),
    (7, 5, 'Researcher', 40);

-- Insert sample data into salary_history
INSERT INTO salary_history VALUES
    (1, 1, 80000.00, '2022-01-15', '2022-12-31'),
    (2, 1, 85000.00, '2023-01-01', NULL),
    (3, 2, 70000.00, '2021-06-20', '2022-06-30'),
    (4, 2, 75000.00, '2022-07-01', NULL),
    (5, 3, 90000.00, '2020-03-10', '2021-03-31'),
    (6, 3, 95000.00, '2021-04-01', NULL);
