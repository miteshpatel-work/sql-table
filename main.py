from typing import Dict, Any, List, Tuple
import snowflake.connector
import google.generativeai as genai
import os
from dotenv import load_dotenv

class NLToSQLConverter:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini model
        print("Initializing Gemini model...")
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize Snowflake connection
        self.snow_conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse='PERSONAL_WH',
            database='PERSONAL_DB',
            schema='PUBLIC'
        )
        self.cursor = self.snow_conn.cursor()
        
        # Set schema explicitly
        try:
            self.cursor.execute("USE SCHEMA PUBLIC")
        except Exception as e:
            print(f"Warning: Could not set schema: {e}")
        
        # Get all tables in the schema
        self.tables = self._get_all_tables()
        
        # Get schema and relationship information
        self.schema_info = self._get_schema_info()
        self.table_relationships = self._discover_relationships()
        
        # Store detected primary and foreign keys
        self.primary_keys = {}
        self.foreign_keys = {}
        self._discover_keys()

    def _get_all_tables(self) -> List[str]:
        """Get all table names in the current schema"""
        try:
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'PUBLIC' 
            AND table_type = 'BASE TABLE';
            """
            self.cursor.execute(query)
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"Error getting tables: {str(e)}")
            return []

    def _get_schema_info(self) -> str:
        """Retrieve schema information for all tables in the database"""
        try:
            schema_query = """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'PUBLIC'
            ORDER BY table_name, ordinal_position;
            """
            self.cursor.execute(schema_query)
            schema_info = ""
            current_table = ""
            
            for row in self.cursor.fetchall():
                if row[0] != current_table:
                    current_table = row[0]
                    schema_info += f"\nTable: {current_table}\n"
                nullable = "NULL" if row[3] == "YES" else "NOT NULL"
                schema_info += f"- {row[1]} ({row[2]}) {nullable}\n"
            
            return schema_info
        except Exception as e:
            return f"Error getting schema info: {str(e)}"

    def _discover_keys(self):
        """Discover primary and foreign keys for all tables using Snowflake's SHOW commands"""
        try:
            # Get primary keys using SHOW PRIMARY KEYS
            for table in self.tables:
                try:
                    self.cursor.execute(f"SHOW PRIMARY KEYS IN {table}")
                    pk_results = self.cursor.fetchall()
                    if pk_results:
                        # "column_name" is at index 4 in Snowflake's SHOW PRIMARY KEYS output
                        self.primary_keys[table] = pk_results[0][4]
                except Exception as e:
                    print(f"Warning: Could not get primary key for table {table}: {str(e)}")

            # Get foreign keys using SHOW IMPORTED KEYS
            for table in self.tables:
                try:
                    self.cursor.execute(f"SHOW IMPORTED KEYS IN TABLE {table}")
                    fk_results = self.cursor.fetchall()
                    
                    if fk_results:
                        if table not in self.foreign_keys:
                            self.foreign_keys[table] = []
                        
                        for row in fk_results:
                            # Snowflake SHOW IMPORTED KEYS column indices:
                            # PK_TABLE_NAME: 2, PK_COLUMN_NAME: 3
                            # FK_TABLE_NAME: 6, FK_COLUMN_NAME: 7
                            self.foreign_keys[table].append({
                                'column': row[7],
                                'references': {
                                    'table': row[2],
                                    'column': row[3]
                                }
                            })
                except Exception as e:
                    print(f"Warning: Could not get foreign keys for table {table}: {str(e)}")
                    
        except Exception as e:
            print(f"Warning: Could not discover all keys: {str(e)}")
            print("Continuing with limited key information...")

    def _discover_relationships(self) -> str:
        """Discover and format table relationships using Snowflake-specific commands"""
        try:
            relationships = "\nTable Relationships:\n"
            
            # Use SHOW IMPORTED KEYS for each table
            for table in self.tables:
                try:
                    self.cursor.execute(f"SHOW IMPORTED KEYS IN TABLE {table};")
                    fk_results = self.cursor.fetchall()
                    
                    if fk_results:
                        for row in fk_results:
                            pk_table = row[2]  # PK_TABLE_NAME
                            pk_column = row[3]  # PK_COLUMN_NAME
                            fk_table = row[6]   # FK_TABLE_NAME
                            fk_column = row[7]  # FK_COLUMN_NAME
                            relationships += f"- {fk_table}.{fk_column} references {pk_table}.{pk_column}\n"
                except Exception as e:
                    print(f"Warning: Could not get relationships for table {table}: {str(e)}")
            
            # Additional relationship discovery through naming conventions
            for table in self.tables:
                columns_query = f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'PUBLIC' 
                AND table_name = '{table}'
                """
                self.cursor.execute(columns_query)
                columns = [row[0] for row in self.cursor.fetchall()]
                
                for column in columns:
                    # Look for potential relationships through common naming patterns
                    if column.lower().endswith('_id'):
                        potential_table = column[:-3]  # Remove '_id'
                        if potential_table.upper() in self.tables:
                            relationships += f"- Potential (by naming): {table}.{column} might reference {potential_table.upper()}.{column}\n"
            
            return relationships if relationships != "\nTable Relationships:\n" else "\nTable Relationships: No explicit relationships found.\n"
            
        except Exception as e:
            print(f"Warning: Could not discover all relationships: {str(e)}")
            return "\nTable Relationships: Could not determine relationships.\n"

    def _suggest_joins(self, tables: List[str]) -> List[str]:
        """Suggest JOIN clauses based on discovered relationships"""
        joins = []
        processed_tables = set()
        
        def add_joins(current_table: str):
            if current_table in processed_tables:
                return
            processed_tables.add(current_table)
            
            if current_table in self.foreign_keys:
                for fk in self.foreign_keys[current_table]:
                    ref_table = fk['references']['table']
                    if ref_table in tables and ref_table not in processed_tables:
                        joins.append(
                            f"JOIN {ref_table} ON {current_table}.{fk['column']} = "
                            f"{ref_table}.{fk['references']['column']}"
                        )
                        add_joins(ref_table)
        
        for table in tables:
            add_joins(table)
        
        return joins

    def generate_sql(self, user_query: str) -> str:
        """Convert natural language query to SQL using Gemini"""
        prompt = f"""
        You are a SQL query generator. Convert the following question to a SQL query based on the given database schema and relationships.
        Only return the SQL query itself, without any explanations or additional text.
        
        Database schema:
        {self.schema_info}
        
        Known relationships:
        {self.table_relationships}
        
        Primary Keys:
        {self.primary_keys}
        
        Foreign Keys:
        {self.foreign_keys}
        
        Important notes:
        - Use appropriate JOIN clauses based on the discovered relationships
        - Include clear column aliases when joining tables
        - Use table aliases to make the query more readable
        - Consider using appropriate aggregate functions when needed
        - Ensure proper join conditions based on the foreign key relationships
        
        Question: {user_query}
        """
        
        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            if not sql_query.upper().startswith('SELECT'):
                sql_query = f"SELECT {sql_query}"
            
            return sql_query
        except Exception as e:
            return f"Error generating SQL: {str(e)}"

    def execute_query(self, sql_query: str) -> Dict:
        """Execute the SQL query and return results"""
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            return {'columns': column_names, 'data': results}
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def process_natural_language_query(self, user_query: str) -> Dict[str, Any]:
        """Process natural language query end-to-end"""
        try:
            print("\nAnalyzing your question...")
            sql_query = self.generate_sql(user_query)
            
            print("\nGenerated SQL:", sql_query)
            print("\nExecuting query...")
            results = self.execute_query(sql_query)
            
            return {
                "user_query": user_query,
                "generated_sql": sql_query,
                "results": results
            }
        except Exception as e:
            return {
                "error": str(e)
            }

    def close(self):
        """Close all connections"""
        self.cursor.close()
        self.snow_conn.close()

def display_results(results: Dict) -> None:
    """Display query results in a formatted table"""
    if isinstance(results, str):  # Error message
        print(f"\nError: {results}")
        return
        
    if not isinstance(results, dict) or 'columns' not in results or 'data' not in results:
        print("\nInvalid results format")
        return
        
    columns = results['columns']
    data = results['data']
    
    if not data:
        print("\nNo results found.")
        return
    
    # Calculate column widths (minimum 10 characters, maximum 50 characters)
    col_widths = []
    for i, col in enumerate(columns):
        max_width = max(
            len(str(col)),
            max(len(str(row[i])) for row in data)
        )
        col_widths.append(min(max(max_width, 10), 50))
    
    # Print header
    header = " | ".join(str(col).ljust(width) for col, width in zip(columns, col_widths))
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    
    # Print data rows
    for row in data:
        # Truncate long values and add ellipsis
        formatted_row = []
        for i, value in enumerate(row):
            str_value = str(value)
            if len(str_value) > col_widths[i]:
                str_value = str_value[:col_widths[i]-3] + "..."
            formatted_row.append(str_value.ljust(col_widths[i]))
        print(" | ".join(formatted_row))
    
    print(separator + "\n")

def main():
    try:
        # Initialize converter
        print("Initializing the NL to SQL Converter...")
        converter = NLToSQLConverter()
        
        # Print discovered information
        print("\nDiscovered table relationships:")
        print(converter.table_relationships)
        
        print("\nPrimary Keys:")
        print(converter.primary_keys)
        
        print("\nForeign Keys:")
        print(converter.foreign_keys)
        
        print("\nExample questions you can ask:")
        print("1. Show me all employees in the Sales department")
        print("2. What is the average salary by department?")
        print("3. Who are the project managers and their projects?")
        print("4. List all employees and their current salaries")
        print("5. Show me departments with more than 5 employees")
        
        # Main loop
        while True:
            try:
                user_query = input("\nEnter your question (or 'exit' to quit): ").strip()
                if not user_query:
                    continue
                    
                if user_query.lower() == 'exit':
                    break
                    
                result = converter.process_natural_language_query(user_query)
                
                if "error" in result:
                    print(f"\nError: {result['error']}")
                else:
                    display_results(result["results"])
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user. Type 'exit' to quit.")
                continue
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                continue
        
        print("\nClosing connections...")
        converter.close()
        print("Done. Goodbye!")
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("\nPlease verify your environment variables and database connection settings.")
        
if __name__ == "__main__":
    main()