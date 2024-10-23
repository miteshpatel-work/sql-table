import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import re
from typing import Annotated, TypedDict, Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import snowflake.connector
import google.generativeai as genai
from dotenv import load_dotenv
import os

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
        warehouse='PERSONAL_WH',
        database='PERSONAL_DB',
        schema='FINANCE'
    )

snow_conn = init_snowflake()
cursor = snow_conn.cursor()

# State definition for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    sql_query: str
    results: Dict[str, Any]
    schema_info: str
    table_relationships: str
    primary_keys: Dict
    foreign_keys: Dict
    visualization: Dict[str, Any]

# Helper functions
def _get_all_tables() -> List[str]:
    try:
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'FINANCE' 
        AND table_type = 'BASE TABLE';
        """
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error getting tables: {str(e)}")
        return []

def _get_schema_info() -> str:
    try:
        schema_query = """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_schema = 'FINANCE'
        ORDER BY table_name, ordinal_position;
        """
        cursor.execute(schema_query)
        schema_info = ""
        current_table = ""
        
        for row in cursor.fetchall():
            if row[0] != current_table:
                current_table = row[0]
                schema_info += f"\nTable: {current_table}\n"
            nullable = "NULL" if row[3] == "YES" else "NOT NULL"
            schema_info += f"- {row[1]} ({row[2]}) {nullable}\n"
        
        return schema_info
    except Exception as e:
        return f"Error getting schema info: {str(e)}"

def _discover_relationships(tables) -> str:
    try:
        relationships = "\nTable Relationships:\n"
        
        for table in tables:
            try:
                cursor.execute(f"SHOW IMPORTED KEYS IN TABLE {table};")
                fk_results = cursor.fetchall()
                
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
        return "\nTable Relationships: Could not determine relationships.\n"

def _discover_keys(tables):
    primary_keys = {}
    foreign_keys = {}
    
    try:
        for table in tables:
            try:
                cursor.execute(f"SHOW PRIMARY KEYS IN {table}")
                pk_results = cursor.fetchall()
                if pk_results:
                    primary_keys[table] = pk_results[0][4]
            except Exception as e:
                print(f"Warning: Could not get primary key for table {table}: {str(e)}")

            try:
                cursor.execute(f"SHOW IMPORTED KEYS IN TABLE {table}")
                fk_results = cursor.fetchall()
                
                if fk_results:
                    if table not in foreign_keys:
                        foreign_keys[table] = []
                    
                    for row in fk_results:
                        foreign_keys[table].append({
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
    
    return primary_keys, foreign_keys

# Initialize database metadata
tables = _get_all_tables()
schema_info = _get_schema_info()
table_relationships = _discover_relationships(tables)
primary_keys, foreign_keys = _discover_keys(tables)

# Agent nodes
def sql_generator(state: State):
    """Node to generate SQL from natural language query"""
    # Get the last user message content
    user_query = state["messages"][-1].content
    
    prompt = f"""
    You are a SQL query generator specifically for Snowflake database. Convert the following question to a SQL query based on the given database schema and relationships.
    Only return the SQL query itself, without any explanations or additional text.
    
    Database schema:
    {state['schema_info']}
    
    Known relationships:
    {state['table_relationships']}
    
    Primary Keys:
    {state['primary_keys']}
    
    Foreign Keys:
    {state['foreign_keys']}
    
    Important notes:
    - Use appropriate JOIN clauses based on the discovered relationships
    - Include clear column aliases when joining tables
    - Use table aliases to make the query more readable
    - Consider using appropriate aggregate functions when needed
    - Ensure proper join conditions based on the foreign key relationships
    
    Question: {user_query}
    """
    
    response = gemini_model.generate_content(prompt)
    sql_query = response.text.strip().replace('```sql', '').replace('```', '').strip()
    
    if not sql_query.upper().startswith('SELECT'):
        sql_query = f"SELECT {sql_query}"
    
    return {
        "messages": state["messages"],
        "sql_query": sql_query,
        "results": state["results"],
        "schema_info": state["schema_info"],
        "table_relationships": state["table_relationships"],
        "primary_keys": state["primary_keys"],
        "foreign_keys": state["foreign_keys"]
    }

def query_executor(state: State):
    """Node to execute SQL query and format results"""
    try:
        cursor.execute(state["sql_query"])
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        formatted_results = {'columns': column_names, 'data': results}
        result_text = f"Here are the results for your query:\n\nSQL Query:\n{state['sql_query']}\n\nResults:\n{_format_results(formatted_results)}"
        
        return {
            "messages": state["messages"] + [AIMessage(content=result_text)],
            "sql_query": state["sql_query"],
            "results": formatted_results,
            "schema_info": state["schema_info"],
            "table_relationships": state["table_relationships"],
            "primary_keys": state["primary_keys"],
            "foreign_keys": state["foreign_keys"]
        }
    except Exception as e:
        error_message = f"Error executing query: {str(e)}"
        return {
            "messages": state["messages"] + [AIMessage(content=error_message)],
            "sql_query": state["sql_query"],
            "results": {"error": error_message},
            "schema_info": state["schema_info"],
            "table_relationships": state["table_relationships"],
            "primary_keys": state["primary_keys"],
            "foreign_keys": state["foreign_keys"]
        }

def visualization_analyzer(state: State):
    """Node to analyze query and results to determine appropriate visualization"""
    user_query = state["messages"][0].content.lower()
    results = state["results"]
    
    # First, determine if visualization is appropriate based on the query type
    visualization_keywords = [
        'distribution', 'compare', 'trend', 'overview', 'analyze', 'correlation',
        'pattern', 'breakdown', 'percentage', 'proportion', 'average', 'mean',
        'visualization', 'graph', 'chart', 'plot'
    ]
    
    list_keywords = [
        'list', 'show', 'display', 'details', 'all', 'information', 'find', 'get',
        'select', 'fetch', 'retrieve'
    ]
    
    # Check if query explicitly asks for a list or details
    is_list_query = any(keyword in user_query for keyword in list_keywords) and not any(keyword in user_query for keyword in visualization_keywords)
    
    if isinstance(results, dict) and 'columns' in results and 'data' in results:
        df = pd.DataFrame(results['data'], columns=results['columns'])
        
        # If it's a list query or has too many columns (detailed info), skip visualization
        if is_list_query or len(df.columns) > 5:
            return {
                **state,
                "visualization": {
                    "chart_type": "table",
                    "description": "Detailed information displayed in table format",
                    "dataframe": df
                }
            }
        
        # Handle single-value results
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            if isinstance(value, (int, float)):  # Only create indicator for numeric values
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=value,
                    title={"text": df.columns[0]},
                    domain={'row': 0, 'column': 0}
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                return {
                    **state,
                    "visualization": {
                        "chart_type": "indicator",
                        "title": f"{df.columns[0]}",
                        "description": "Single value indicator display",
                        "figure": fig,
                        "dataframe": df
                    }
                }
            else:
                return {
                    **state,
                    "visualization": {
                        "chart_type": "table",
                        "description": "Single value displayed in table format",
                        "dataframe": df
                    }
                }
        
        # Handle single column with multiple rows
        if len(df.columns) == 1:
            col_name = df.columns[0]
            # Only visualize if it makes sense (e.g., frequency distribution)
            if len(df) > 1:  # More than one row
                try:
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        # For numeric data, create a histogram
                        fig = px.histogram(
                            df,
                            x=col_name,
                            title=f"Distribution of {col_name}",
                            nbins=min(20, len(df))
                        )
                        viz_type = "histogram"
                    else:
                        # For categorical data, create a bar chart of value counts
                        value_counts = df[col_name].value_counts()
                        if len(value_counts) <= 20:  # Only visualize if there aren't too many categories
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Frequency of {col_name}",
                                labels={'x': col_name, 'y': 'Count'}
                            )
                            viz_type = "bar"
                        else:
                            return {
                                **state,
                                "visualization": {
                                    "chart_type": "table",
                                    "description": "Too many categories for meaningful visualization",
                                    "dataframe": df
                                }
                            }
                    
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        title_x=0.5,
                        height=500,
                        template='plotly_white'
                    )
                    
                    if viz_type == "bar":
                        fig.update_layout(xaxis_tickangle=-45)
                    
                    return {
                        **state,
                        "visualization": {
                            "chart_type": viz_type,
                            "title": fig.layout.title.text,
                            "description": f"Showing distribution of {col_name}",
                            "figure": fig,
                            "dataframe": df
                        }
                    }
                except Exception as e:
                    return {
                        **state,
                        "visualization": {
                            "chart_type": "table",
                            "description": f"Error in visualization: {str(e)}",
                            "dataframe": df
                        }
                    }
        
        # For multiple columns, only proceed with visualization if appropriate
        if not is_list_query and len(df.columns) <= 5:
            try:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                    # Create bar chart for category comparison
                    fig = px.bar(
                        df,
                        x=categorical_cols[0],
                        y=numeric_cols[0],
                        title=f"{numeric_cols[0]} by {categorical_cols[0]}"
                    )
                elif len(numeric_cols) >= 2:
                    # Create scatter plot for numeric correlations
                    fig = px.scatter(
                        df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
                    )
                else:
                    return {
                        **state,
                        "visualization": {
                            "chart_type": "table",
                            "description": "Data not suitable for visualization",
                            "dataframe": df
                        }
                    }
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    title_x=0.5,
                    height=500,
                    template='plotly_white',
                    xaxis_tickangle=-45 if len(df) > 5 else 0
                )
                
                return {
                    **state,
                    "visualization": {
                        "chart_type": "auto",
                        "title": fig.layout.title.text,
                        "description": "Automatic visualization based on data types",
                        "figure": fig,
                        "dataframe": df
                    }
                }
            except Exception as e:
                return {
                    **state,
                    "visualization": {
                        "chart_type": "table",
                        "description": f"Error in visualization: {str(e)}",
                        "dataframe": df
                    }
                }
    
    return {
        **state,
        "visualization": {
            "chart_type": "table",
            "description": "Data displayed in table format",
            "dataframe": df if 'df' in locals() else None
        }
    }

# Add this helper function to detect if a string contains any of the given terms
def contains(string, terms):
    return any(term in string for term in terms)

def _format_results(results: Dict) -> str:
    """
    Helper function to format query results as a well-aligned markdown table
    with proper spacing and formatting.
    """
    if isinstance(results, str):
        return f"\nError: {results}"
        
    if not isinstance(results, dict) or 'columns' not in results or 'data' not in results:
        return "\nInvalid results format"
        
    columns = results['columns']
    data = results['data']
    
    if not data:
        return "\nNo results found."
    
    # Convert all values to strings and find maximum width for each column
    str_data = [[str(val) for val in row] for row in data]
    col_widths = [len(col) for col in columns]  # Start with header widths
    
    # Update column widths based on data
    for row in str_data:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
    
    # Add padding
    padding = 2  # Space on each side of the content
    col_widths = [width + (padding * 2) for width in col_widths]
    
    # Format header
    header = "│"
    for col, width in zip(columns, col_widths):
        header += f" {col:<{width}} │"
    
    # Format separator
    separator = "├"
    for width in col_widths:
        separator += "─" * width + "┤"
    
    # Format top and bottom borders
    top_border = "┌" + "".join("─" * width + "┐" for width in col_widths)
    bottom_border = "└" + "".join("─" * width + "┘" for width in col_widths)
    
    # Build the table
    output = ["\nResults:", top_border, header, separator]
    
    # Add data rows
    for row in str_data:
        data_row = "│"
        for val, width in zip(row, col_widths):
            data_row += f" {val:<{width}} │"
        output.append(data_row)
    
    output.append(bottom_border)
    
    return "\n".join(output)

def create_streamlit_app():
    st.title("Natural Language to SQL Visualizer")
    st.write("Ask questions about your data in plain English!")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Build the graph
    workflow = StateGraph(State)
    workflow.add_node("sql_generator", sql_generator)
    workflow.add_node("query_executor", query_executor)
    workflow.add_node("visualization_analyzer", visualization_analyzer)
    
    # Add edges
    workflow.add_edge(START, "sql_generator")
    workflow.add_edge("sql_generator", "query_executor")
    workflow.add_edge("query_executor", "visualization_analyzer")
    workflow.add_edge("visualization_analyzer", END)
    
    # Compile the graph
    graph = workflow.compile()
    
    # Query input
    user_query = st.text_input("Enter your question:", key="query_input")
    
    if st.button("Submit"):
        if user_query:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "sql_query": "",
                "results": {},
                "schema_info": schema_info,
                "table_relationships": table_relationships,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
                "visualization": {}
            }
            
            # Process through the graph
            final_state = None
            for state in graph.stream(initial_state, stream_mode="values"):
                final_state = state
            
            if final_state:
                # Display SQL Query
                st.subheader("SQL Query")
                st.code(final_state["sql_query"], language="sql")
                
                # Display Results
                st.subheader("Query Results")
                if 'visualization' in final_state and 'dataframe' in final_state['visualization']:
                    st.dataframe(final_state['visualization']['dataframe'])
                
                # Display Visualization
                st.subheader("Visualization")
                if 'visualization' in final_state and 'figure' in final_state['visualization']:
                    viz_config = final_state['visualization']
                    st.plotly_chart(viz_config['figure'])
                    
                    # Display visualization explanation
                    if 'description' in viz_config:
                        st.info(f"Visualization Choice: {viz_config['description']}")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": user_query,
                    "sql": final_state["sql_query"],
                    "results": final_state["results"]
                })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Previous Queries")
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Query {len(st.session_state.chat_history) - i}: {item['query'][:50]}..."):
                st.code(item['sql'], language="sql")

if __name__ == "__main__":
    create_streamlit_app()