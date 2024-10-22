from typing import Dict, Any, List, Tuple, Annotated
import snowflake.connector
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Sequence
import operator
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

# Define the state structure
class AgentState(TypedDict):
    messages: Sequence[Any]
    query: str
    sql: str
    results: Any
    explanation: str
    graph_data: Dict

class DatabaseConfig:
    def __init__(self):
        load_dotenv()
        self.snow_conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse='PERSONAL_WH',
            database='PERSONAL_DB',
            schema='PUBLIC'
        )
        self.cursor = self.snow_conn.cursor()
        self.schema_info = self._get_schema_info()
        self.relationships = self._get_relationships()

    def _get_schema_info(self):
        # Implementation remains same as before
        pass

    def _get_relationships(self):
        # Implementation remains same as before
        pass

    def close(self):
        self.cursor.close()
        self.snow_conn.close()

class QueryUnderstandingAgent:
    """Agent responsible for understanding the user's query and identifying required tables/columns"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    def analyze(self, state: AgentState) -> AgentState:
        messages = [
            HumanMessage(content=f"""
            Analyze this query and identify:
            1. The main intent
            2. Required tables and columns
            3. Any conditions or filters
            4. Required aggregations
            
            Query: {state['query']}
            """)
        ]
        
        response = self.llm.invoke(messages)
        state['messages'].append(response)
        return state

class SQLGenerationAgent:
    """Agent responsible for generating SQL query based on the analysis"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.db_config = db_config
    
    def generate(self, state: AgentState) -> AgentState:
        previous_analysis = state['messages'][-1].content
        
        messages = [
            HumanMessage(content=f"""
            Generate a Snowflake SQL query based on this analysis and schema:
            
            Analysis: {previous_analysis}
            
            Schema:
            {self.db_config.schema_info}
            
            Relationships:
            {self.db_config.relationships}
            
            Use Snowflake's date functions for any date operations:
            - YEAR(date_column) for year extraction
            - DATEADD(year/month/day, number, date) for date arithmetic
            - CURRENT_DATE() for current date
            """)
        ]
        
        response = self.llm.invoke(messages)
        state['sql'] = response.content.strip()
        state['messages'].append(response)
        return state

class QueryExecutionAgent:
    """Agent responsible for executing the SQL query and handling results"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
    
    def execute(self, state: AgentState) -> AgentState:
        try:
            self.db_config.cursor.execute(state['sql'])
            results = self.db_config.cursor.fetchall()
            column_names = [desc[0] for desc in self.db_config.cursor.description]
            state['results'] = {
                'columns': column_names,
                'data': results
            }
            return state
        except Exception as e:
            state['results'] = {'error': str(e)}
            return state

class ResultAnalysisAgent:
    """Agent responsible for analyzing results and preparing visualization"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    def analyze(self, state: AgentState) -> AgentState:
        if 'error' in state['results']:
            state['explanation'] = f"Error occurred: {state['results']['error']}"
            return state
            
        results = state['results']
        messages = [
            HumanMessage(content=f"""
            Analyze these query results and suggest the best way to visualize them:
            
            Columns: {results['columns']}
            Sample Data: {results['data'][:5]}
            
            Provide:
            1. A brief explanation of the results
            2. Recommended visualization type (bar, line, pie, etc.)
            3. Key insights from the data
            """)
        ]
        
        response = self.llm.invoke(messages)
        state['explanation'] = response.content
        
        # Prepare graph data
        state['graph_data'] = {
            'columns': results['columns'],
            'data': results['data'],
            'visualization_type': 'bar'  # Default, can be modified based on analysis
        }
        
        return state

class VisualizationAgent:
    """Agent responsible for creating visualizations"""
    
    def visualize(self, state: AgentState) -> AgentState:
        if 'error' in state['results']:
            return state
            
        data = state['graph_data']
        plt.figure(figsize=(10, 6))
        
        if len(data['data']) > 0:
            # Basic bar chart - can be extended based on data type
            x = range(len(data['data']))
            y = [row[1] if len(row) > 1 else row[0] for row in data['data']]
            plt.bar(x, y)
            plt.xticks(x, [row[0] for row in data['data']], rotation=45)
            plt.title(f"Results Visualization for: {state['query']}")
            plt.tight_layout()
            
            # Save plot to file or convert to base64 for web display
            plt.savefig('query_results.png')
            plt.close()
        
        return state

class MultiAgentNLtoSQL:
    def __init__(self):
        self.db_config = DatabaseConfig()
        
        # Initialize agents
        self.query_understanding = QueryUnderstandingAgent()
        self.sql_generation = SQLGenerationAgent(self.db_config)
        self.query_execution = QueryExecutionAgent(self.db_config)
        self.result_analysis = ResultAnalysisAgent()
        self.visualization = VisualizationAgent()
        
        # Create workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("understand", self.query_understanding.analyze)
        workflow.add_node("generate", self.sql_generation.generate)
        workflow.add_node("execute", self.query_execution.execute)
        workflow.add_node("analyze", self.result_analysis.analyze)
        workflow.add_node("visualize", self.visualization.visualize)
        
        # Define edges
        workflow.add_edge("understand", "generate")
        workflow.add_edge("generate", "execute")
        workflow.add_edge("execute", "analyze")
        workflow.add_edge("analyze", "visualize")
        workflow.add_edge("visualize", END)
        
        # Set entry point
        workflow.set_entry_point("understand")
        
        return workflow
    
    def process_query(self, query: str) -> Dict[str, Any]:
        # Initialize state
        state = AgentState(
            messages=[],
            query=query,
            sql="",
            results={},
            explanation="",
            graph_data={}
        )
        
        # Execute workflow
        final_state = self.workflow.invoke(state)
        
        return {
            "query": final_state["query"],
            "sql": final_state["sql"],
            "results": final_state["results"],
            "explanation": final_state["explanation"],
            "visualization_path": "query_results.png"
        }
    
    def close(self):
        self.db_config.close()

def main():
    try:
        system = MultiAgentNLtoSQL()
        
        while True:
            query = input("\nEnter your question (or 'exit' to quit): ").strip()
            
            if query.lower() == 'exit':
                break
            
            if not query:
                continue
            
            print("\nProcessing your query...")
            result = system.process_query(query)
            
            print("\nGenerated SQL:")
            print(result["sql"])
            
            print("\nResults Analysis:")
            print(result["explanation"])
            
            print("\nVisualization saved as 'query_results.png'")
            
        system.close()
        print("\nThank you for using the system!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
