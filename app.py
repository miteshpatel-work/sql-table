from typing import Annotated, Dict, List, Tuple, TypedDict, Optional
import snowflake.connector
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter
import re
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Custom types for state management
class AgentState(TypedDict):
    messages: List[Dict]
    current_sql: str
    attempts: int
    final_result: Optional[Dict]
    error: Optional[str]

class SnowflakeDB:
    def __init__(self):
        self.conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse='PERSONAL_WH',
            database='PERSONAL_DB',
            schema='FINANCE'
        )
        self.cursor = self.conn.cursor()
        self.schema_info = self._get_schema_info()
        self.relationships = self._discover_relationships()
        self.primary_keys = {}
        self.foreign_keys = {}
        self._discover_keys()

    def _get_schema_info(self) -> str:
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

    def _discover_relationships(self) -> str:
        try:
            relationships = "\nTable Relationships:\n"
            tables = self._get_tables()
            
            for table in tables:
                try:
                    self.cursor.execute(f"SHOW IMPORTED KEYS IN TABLE {table};")
                    fk_results = self.cursor.fetchall()
                    
                    if fk_results:
                        for row in fk_results:
                            pk_table = row[2]
                            pk_column = row[3]
                            fk_table = row[6]
                            fk_column = row[7]
                            relationships += f"- {fk_table}.{fk_column} references {pk_table}.{pk_column}\n"
                except Exception as e:
                    print(f"Warning: Could not get relationships for table {table}: {str(e)}")
            
            return relationships
        except Exception as e:
            return f"Error discovering relationships: {str(e)}"

    def _get_tables(self) -> List[str]:
        self.cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'PUBLIC' 
            AND table_type = 'BASE TABLE';
        """)
        return [row[0] for row in self.cursor.fetchall()]

    def _discover_keys(self):
        """Discover primary and foreign keys using Snowflake's SHOW commands"""
        try:
            tables = self._get_tables()
            
            # Get primary keys
            for table in tables:
                try:
                    self.cursor.execute(f"SHOW PRIMARY KEYS IN {table}")
                    pk_results = self.cursor.fetchall()
                    if pk_results:
                        self.primary_keys[table] = pk_results[0][4]
                except Exception as e:
                    print(f"Warning: Could not get primary key for table {table}: {str(e)}")

            # Get foreign keys
            for table in tables:
                try:
                    self.cursor.execute(f"SHOW IMPORTED KEYS IN TABLE {table}")
                    fk_results = self.cursor.fetchall()
                    
                    if fk_results:
                        if table not in self.foreign_keys:
                            self.foreign_keys[table] = []
                        
                        for row in fk_results:
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

    def execute_query(self, sql: str) -> Dict:
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return {
                'success': True,
                'columns': columns,
                'data': results
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def close(self):
        self.cursor.close()
        self.conn.close()

class SQLAgent:
    def __init__(self):
        self.db = SnowflakeDB()
        self.workflow = self._create_workflow()

    def _create_sql_query(self, state: AgentState) -> AgentState:
        """Generate or modify SQL query based on the current state"""
        messages = state['messages']
        current_sql = state['current_sql']
        attempts = state['attempts']

        # Prepare the prompt based on whether this is the first attempt or a retry
        if attempts == 0:
            prompt = f"""
            Convert this question to a SQL query. Use the following schema information:
            
            {self.db.schema_info}
            
            Relationships:
            {self.db.relationships}
            
            Question: {messages[-1]['content']}
            
            Return only the SQL query, no explanations.
            """
        else:
            error = state['error']
            prompt = f"""
            The following SQL query failed:
            {current_sql}
            
            Error: {error}
            
            Please modify the query to fix this error. Use the schema:
            {self.db.schema_info}
            
            Return only the modified SQL query, no explanations.
            """

        # Generate SQL using Gemini
        response = model.generate_content(prompt)
        sql_query = response.text.strip()
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()

        # Update state
        state['current_sql'] = sql_query
        return state

    def _execute_query(self, state: AgentState) -> AgentState:
        """Execute the current SQL query"""
        result = self.db.execute_query(state['current_sql'])
        
        if result['success']:
            state['final_result'] = result
            state['error'] = None
        else:
            state['error'] = result['error']
            state['attempts'] += 1
        
        return state

    def _should_continue(self, state: AgentState) -> bool:
        """Determine if we should continue trying"""
        return (
            state['error'] is not None and 
            state['attempts'] < 5 and 
            state['final_result'] is None
        )

    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("create_sql", self._create_sql_query)
        workflow.add_node("execute_query", self._execute_query)

        # Add edges
        workflow.add_edge("create_sql", "execute_query")
        workflow.add_conditional_edges(
            "execute_query",
            self._should_continue,
            {
                True: "create_sql",
                False: END
            }
        )

        # Set entry point
        workflow.set_entry_point("create_sql")

        return workflow.compile()

    def process_query(self, user_query: str) -> Dict:
        """Process a natural language query and return results"""
        # Initialize state
        state = {
            "messages": [{"role": "user", "content": user_query}],
            "current_sql": "",
            "attempts": 0,
            "final_result": None,
            "error": None
        }

        # Run the workflow
        final_state = self.workflow.invoke(state)

        if final_state['final_result']:
            return {
                'success': True,
                'sql': final_state['current_sql'],
                'results': final_state['final_result'],
                'attempts': final_state['attempts'] + 1
            }
        else:
            return {
                'success': False,
                'sql': final_state['current_sql'],
                'error': final_state['error'],
                'attempts': final_state['attempts']
            }

def display_results(result: Dict) -> None:
    """Display query results in a formatted way"""
    print(f"\nAttempts: {result['attempts']}")
    print(f"Final SQL: {result['sql']}")
    
    if result['success']:
        data = result['results']
        columns = data['columns']
        rows = data['data']
        
        if not rows:
            print("\nNo results found.")
            return
            
        # Calculate column widths
        col_widths = []
        for i, col in enumerate(columns):
            max_width = max(
                len(str(col)),
                max(len(str(row[i])) for row in rows)
            )
            col_widths.append(min(max(max_width, 10), 50))
        
        # Print header
        header = " | ".join(str(col).ljust(width) for col, width in zip(columns, col_widths))
        separator = "-" * len(header)
        print("\n" + separator)
        print(header)
        print(separator)
        
        # Print rows
        for row in rows:
            formatted_row = []
            for i, value in enumerate(row):
                str_value = str(value)
                if len(str_value) > col_widths[i]:
                    str_value = str_value[:col_widths[i]-3] + "..."
                formatted_row.append(str_value.ljust(col_widths[i]))
            print(" | ".join(formatted_row))
        
        print(separator)
    else:
        print(f"\nError: {result['error']}")

def main():
    try:
        print("Initializing SQL Agent...")
        agent = SQLAgent()
        
        print("\nExample questions you can ask:")
        print("1. Show me all employees in the Sales department")
        print("2. What is the average salary by department?")
        print("3. Who are the project managers and their projects?")
        print("4. List all employees hired in 2023")
        print("5. Show me departments with more than 5 employees")
        
        while True:
            try:
                query = input("\nEnter your question (or 'exit' to quit): ").strip()
                
                if not query:
                    continue
                    
                if query.lower() == 'exit':
                    break
                    
                result = agent.process_query(query)
                display_results(result)
                
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                continue
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
        
        print("\nClosing connections...")
        agent.db.close()
        print("Goodbye!")
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please verify your environment variables and database connection settings.")

if __name__ == "__main__":
    main()