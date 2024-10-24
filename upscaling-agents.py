import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from typing import Annotated, TypedDict, Dict, Any, List
from streamlit_mermaid import st_mermaid
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
import snowflake.connector
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-pro')

# Initialize Snowflake connection
@st.cache_resource
def init_snowflake():
    return snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse='COMPUTE_WH',
        database='SNOWFLAKE_SAMPLE_DATA',
        schema='TPCDS_SF100TCL'
    )

snow_conn = init_snowflake()
cursor = snow_conn.cursor()

# Enhanced State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    schema_metadata: Dict[str, Any]
    relevant_tables: Dict[str, Any]
    sql_query: str
    results: Dict[str, Any]
    error_count: int
    visualization: Dict[str, Any]
    last_error: str
    query_limit: int

# Schema Metadata Collection Agent
def schema_metadata_collector(state: State):
    """Agent responsible for collecting and organizing schema metadata"""
    try:
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'TPCDS_SF100TCL' 
            AND table_type = 'BASE TABLE';
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_metadata = {
            'tables': {},
            'relationships': [],
            'last_updated': datetime.now().isoformat()
        }
        
        # Collect detailed information for each table
        for table in tables:
            # Get columns and their properties
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable, 
                       (SELECT 'YES' FROM information_schema.key_column_usage 
                        WHERE table_name = columns.table_name 
                        AND column_name = columns.column_name 
                        AND constraint_name LIKE 'PK%'
                        LIMIT 1) as is_primary_key
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            
            schema_metadata['tables'][table] = {
                'columns': [{
                    'name': col[0],
                    'type': col[1],
                    'nullable': col[2] == 'YES',
                    'is_primary_key': col[3] == 'YES'
                } for col in columns],
                'relationships': []
            }
            
            # Get foreign key relationships
            try:
                cursor.execute(f"SHOW IMPORTED KEYS IN TABLE {table};")
                fk_results = cursor.fetchall()
                
                for fk in fk_results:
                    relationship = {
                        'from_table': fk[6],
                        'from_column': fk[7],
                        'to_table': fk[2],
                        'to_column': fk[3]
                    }
                    schema_metadata['tables'][table]['relationships'].append(relationship)
                    schema_metadata['relationships'].append(relationship)
            except Exception as e:
                print(f"Warning: Could not get relationships for {table}: {str(e)}")
        
        return {
            **state,
            "schema_metadata": schema_metadata,
            "messages": state["messages"] + [SystemMessage(content="Schema metadata collected successfully")]
        }
    except Exception as e:
        error_msg = f"Error collecting schema metadata: {str(e)}"
        return {
            **state,
            "schema_metadata": {},
            "messages": state["messages"] + [SystemMessage(content=error_msg)],
            "last_error": error_msg
        }

# Query Analyzer Agent
def query_analyzer(state: State):
    """Agent to analyze the query and identify relevant tables"""
    user_query = state["messages"][-1].content
    schema_metadata = state["schema_metadata"]
    
    prompt = f"""
    You are a database query analyzer. Given a user question and database schema, identify the most relevant tables and their relationships needed to answer the question.
    
    Database Schema Summary:
    {json.dumps(schema_metadata, indent=2)}
    
    User Question: {user_query}
    
    Analyze the question and:
    1. Identify the main tables needed
    2. Include any related tables necessary for joins
    3. Specify the relevant columns
    4. Note any important relationships
    
    Return ONLY a JSON object with this structure:
    {{
        "main_tables": ["table1", "table2"],
        "related_tables": ["table3"],
        "relevant_columns": {{"table1": ["col1", "col2"], "table2": ["col1"]}},
        "relationships": [
            {{"from_table": "table1", "from_column": "col1", "to_table": "table2", "to_column": "col2"}}
        ],
        "reasoning": "Brief explanation of why these tables were chosen"
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        analysis = json.loads(response.text)
        
        # Validate the selected tables exist in the schema
        all_tables = set(schema_metadata['tables'].keys())
        selected_tables = set(analysis['main_tables'] + analysis.get('related_tables', []))
        
        if not selected_tables.issubset(all_tables):
            invalid_tables = selected_tables - all_tables
            raise ValueError(f"Invalid tables selected: {invalid_tables}")
            
        return {
            **state,
            "relevant_tables": analysis,
            "messages": state["messages"] + [SystemMessage(content=f"Query analysis completed: {analysis['reasoning']}")],
        }
    except Exception as e:
        error_msg = f"Error in query analysis: {str(e)}"
        return {
            **state,
            "relevant_tables": {},
            "messages": state["messages"] + [SystemMessage(content=error_msg)],
            "last_error": error_msg
        }

# SQL Generator Agent
def sql_generator(state: State):
    """Agent to generate SQL from analyzed query with optimization for large datasets"""
    user_query = state["messages"][-1].content
    relevant_tables = state["relevant_tables"]
    last_error = state.get("last_error", "")
    
    optimization_hints = """
    Important Optimization Requirements:
    1. Always include WHERE clauses to filter data
    2. Use appropriate indexes when available
    3. Avoid SELECT * 
    4. Use COUNT(*) instead of COUNT(column) when possible
    5. Add LIMIT clause for large result sets
    6. Consider using aggregations to reduce data volume
    7. Use table sample when appropriate (TABLESAMPLE(1))
    """
    
    # Adjust prompt based on previous errors
    if "timeout" in last_error.lower():
        optimization_hints += """
        8. Query timed out - add more specific filters
        9. Consider using SAMPLE() for approximate results
        10. Break complex joins into simpler queries
        """
    
    prompt = f"""
    Generate an optimized SQL query for Snowflake based on this analysis:
    
    User Question: {user_query}
    
    Relevant Tables and Relationships:
    {json.dumps(relevant_tables, indent=2)}
    
    {optimization_hints}
    
    Return ONLY the SQL query without any explanations.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        sql_query = response.text.strip().replace('```sql', '').replace('```', '').strip()
        
        # Add TABLESAMPLE if dealing with very large tables
        if any(table in sql_query for table in ["store_sales", "web_sales", "inventory"]):
            sql_query = sql_query.replace(
                "FROM store_sales",
                "FROM store_sales TABLESAMPLE(1)"
            ).replace(
                "FROM web_sales",
                "FROM web_sales TABLESAMPLE(1)"
            ).replace(
                "FROM inventory",
                "FROM inventory TABLESAMPLE(1)"
            )
        
        return {
            **state,
            "sql_query": sql_query,
            "messages": state["messages"] + [
                FunctionMessage(
                    content="Optimized SQL query generated",
                    name="sql_generator"
                )
            ]
        }
    except Exception as e:
        error_msg = f"Error generating SQL: {str(e)}"
        return {
            **state,
            "sql_query": "",
            "messages": state["messages"] + [
                FunctionMessage(
                    content=error_msg,
                    name="sql_generator"
                )
            ],
            "last_error": error_msg
        }

# Query Executor Agent with Feedback Loop
def query_executor(state: State):
    """Agent to execute SQL query with error handling and query limits"""
    try:
        # Add LIMIT clause if not present
        sql_query = state["sql_query"]
        if "LIMIT" not in sql_query.upper():
            sql_query = f"{sql_query} LIMIT 10000"  # Default limit

        # Add timeout to query
        sql_query = f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 30; {sql_query}"
        
        # Execute with cursor
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        formatted_results = {
            'columns': column_names,
            'data': results,
            'row_count': len(results)
        }
        
        return {
            **state,
            "results": formatted_results,
            "error_count": 0,
            "messages": state["messages"] + [
                FunctionMessage(
                    content="Query executed successfully",
                    name="query_executor"
                )
            ]
        }
        
    except Exception as e:
        error_msg = str(e)
        state["error_count"] = state.get("error_count", 0) + 1
        
        if state["error_count"] <= 3:  # Reduced max retries
            # Add specific error handling for common large data issues
            if "timeout" in error_msg.lower() or "resource limit" in error_msg.lower():
                return {
                    **state,
                    "last_error": "Query timeout - needs optimization",
                    "messages": state["messages"] + [
                        FunctionMessage(
                            content="Query timeout - attempting to optimize",
                            name="query_executor"
                        )
                    ]
                }
            else:
                return {
                    **state,
                    "last_error": error_msg,
                    "messages": state["messages"] + [
                        FunctionMessage(
                            content=f"Query error: {error_msg}",
                            name="query_executor"
                        )
                    ]
                }
        else:
            return {
                **state,
                "results": {"error": "Max retry attempts reached"},
                "last_error": error_msg,
                "messages": state["messages"] + [
                    FunctionMessage(
                        content="Query failed after maximum retry attempts",
                        name="query_executor"
                    )
                ]
            }


# Enhanced Visualization Agent
def visualization_analyzer(state: State):
    """Agent to analyze results and create appropriate visualizations"""
    user_query = state["messages"][-1].content.lower()
    results = state["results"]
    
    # First, determine if visualization is needed
    visualization_keywords = [
        'trend', 'compare', 'distribution', 'relationship', 'pattern',
        'visualization', 'chart', 'graph', 'plot', 'show', 'display'
    ]
    
    needs_visualization = any(keyword in user_query for keyword in visualization_keywords)
    
    if not needs_visualization or 'error' in results:
        return {
            **state,
            "visualization": {
                "type": "none",
                "reason": "Visualization not required or error in results"
            }
        }
    
    try:
        # Convert results to DataFrame
        df = pd.DataFrame(results['data'], columns=results['columns'])
        
        # Analyze data types and structure
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Determine appropriate visualization
        if len(df) == 1 and len(numeric_cols) == 1:
            # Single numeric value - use indicator
            fig = go.Figure(go.Indicator(
                mode="number",
                value=df[numeric_cols[0]].iloc[0],
                title={"text": numeric_cols[0]}
            ))
            viz_type = "indicator"
            
        elif len(df.columns) == 2 and len(numeric_cols) == 1 and len(categorical_cols) == 1:
            # Category vs. numeric - use bar chart
            fig = px.bar(
                df,
                x=categorical_cols[0],
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} by {categorical_cols[0]}"
            )
            viz_type = "bar"
            
        elif len(numeric_cols) >= 2:
            # Multiple numeric columns - use scatter plot
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
            )
            viz_type = "scatter"
            
        elif len(categorical_cols) == 1 and len(df) > 1:
            # Single categorical column - use pie chart
            value_counts = df[categorical_cols[0]].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {categorical_cols[0]}"
            )
            viz_type = "pie"
            
        else:
            return {
                **state,
                "visualization": {
                    "type": "table",
                    "reason": "Data structure not suitable for visualization"
                }
            }
        
        # Update layout
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            title_x=0.5
        )
        
        return {
            **state,
            "visualization": {
                "type": viz_type,
                "figure": fig,
                "reason": "Visualization created based on data structure and query type"
            }
        }
        
    except Exception as e:
        return {
            **state,
            "visualization": {
                "type": "error",
                "reason": f"Error creating visualization: {str(e)}"
            }
        }

def create_streamlit_app():
    st.title("Enhanced NLP to SQL Multi-Agent System")
    
    # Display workflow diagram
    st.subheader("Agent Workflow Structure")
    mermaid_code = """
    flowchart LR
        START(Start) --> meta[Schema Metadata Collector]
        meta --> analyzer[Query Analyzer]
        analyzer --> sql[SQL Generator]
        sql --> exec[Query Executor]
        exec -->|Error| analyzer
        exec --> viz[Visualization Analyzer]
        viz --> END(End)
        
        style START fill:#e9ecef,stroke:#343a40
        style END fill:#e9ecef,stroke:#343a40
        style meta fill:#bbdefb,stroke:#1976d2
        style analyzer fill:#c8e6c9,stroke:#388e3c
        style sql fill:#dcedc8,stroke:#689f38
        style exec fill:#f8bbd0,stroke:#c2185b
        style viz fill:#b3e5fc,stroke:#0288d1
    """
    st_mermaid(mermaid_code, height="200px")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'schema_metadata' not in st.session_state:
        st.session_state.schema_metadata = None
    
    # Build the graph with increased recursion limit
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("schema_metadata_collector", schema_metadata_collector)
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("sql_generator", sql_generator)
    workflow.add_node("query_executor", query_executor)
    workflow.add_node("visualization_analyzer", visualization_analyzer)
    
    # Add edges
    workflow.add_edge(START, "schema_metadata_collector")
    workflow.add_edge("schema_metadata_collector", "query_analyzer")
    workflow.add_edge("query_analyzer", "sql_generator")
    workflow.add_edge("sql_generator", "query_executor")
    
    def determine_next_step(state: State) -> str:
        """Determine the next node based on error state"""
        if (state.get("last_error") and 
            state.get("error_count", 0) <= 3 and  # Reduced max retries
            "Max retry attempts reached" not in state.get("last_error", "")):
            
            # If timeout error, go back to SQL generator for optimization
            if "timeout" in state.get("last_error", "").lower():
                return "sql_generator"
            # For other errors, go back to query analyzer
            return "query_analyzer"
        return "visualization_analyzer"
    
    # Add conditional edge with proper path function
    workflow.add_conditional_edges(
        source="query_executor",
        path=determine_next_step
    )
    
    workflow.add_edge("visualization_analyzer", END)
    
    # Compile the graph with increased recursion limit
    graph = workflow.compile()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'schema_metadata' not in st.session_state:
        st.session_state.schema_metadata = None
    
    # Initialize schema metadata with query limits
    if not st.session_state.schema_metadata:
        initial_state = {
            "messages": [SystemMessage(content="Initializing schema metadata collection")],
            "schema_metadata": {},
            "relevant_tables": {},
            "sql_query": "",
            "results": {},
            "error_count": 0,
            "visualization": {},
            "last_error": "",
            "query_limit": 10000  # Default query limit
        }
        
        with st.spinner("Collecting database schema information..."):
            for state in graph.stream(initial_state, recursion_limit=10):  # Lower recursion limit
                if state and "schema_metadata" in state and state["schema_metadata"]:
                    st.session_state.schema_metadata = state["schema_metadata"]
                    break
    
    # Query input
    user_query = st.text_input("Enter your question:", key="query_input")
    
    if st.button("Submit"):
        if user_query:
            # Initialize state with pre-collected schema metadata
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "schema_metadata": st.session_state.schema_metadata,
                "relevant_tables": {},
                "sql_query": "",
                "results": {},
                "error_count": 0,
                "visualization": {},
                "last_error": ""
            }
            
            # Process through the graph with progress tracking
            with st.spinner("Processing your query..."):
                progress_bar = st.progress(0)
                stages = ["Analyzing Query", "Generating SQL", "Executing Query", "Creating Visualization"]
                current_stage = 0
                
                final_state = None
                for state in graph.stream(initial_state):
                    final_state = state
                    
                    # Update progress
                    if state.get("messages"):
                        last_message = state["messages"][-1].content
                        if "analysis completed" in last_message:
                            current_stage = 1
                        elif "SQL query generated" in last_message:
                            current_stage = 2
                        elif "Query executed successfully" in last_message:
                            current_stage = 3
                        
                        progress_bar.progress((current_stage + 1) / len(stages))
                        st.write(f"Stage: {stages[current_stage]}")
            
            if final_state:
                # Display SQL Query
                st.subheader("Generated SQL Query")
                st.code(final_state["sql_query"], language="sql")
                
                # Display Results
                st.subheader("Query Results")
                if 'results' in final_state and 'data' in final_state['results']:
                    df = pd.DataFrame(
                        final_state['results']['data'],
                        columns=final_state['results']['columns']
                    )
                    st.dataframe(df)
                    
                    # Display error if any
                    if final_state.get("last_error"):
                        st.error(f"Error encountered: {final_state['last_error']}")
                
                # Display Visualization if available
                if final_state.get("visualization", {}).get("type") not in ["none", "error", "table"]:
                    st.subheader("Visualization")
                    viz_config = final_state["visualization"]
                    st.plotly_chart(viz_config["figure"], use_container_width=True)
                    st.info(f"Visualization type: {viz_config['type']}")
                    if viz_config.get("reason"):
                        st.caption(viz_config["reason"])
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": user_query,
                    "sql": final_state["sql_query"],
                    "results": final_state["results"],
                    "error": final_state.get("last_error", ""),
                    "timestamp": datetime.now().isoformat()
                })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Query History")
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Query {len(st.session_state.chat_history) - i}: {item['query'][:50]}..."):
                st.code(item['sql'], language="sql")
                if item['error']:
                    st.error(f"Error: {item['error']}")
                st.caption(f"Executed at: {item['timestamp']}")

if __name__ == "__main__":
    create_streamlit_app()
