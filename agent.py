from typing import Dict, Any, List, Tuple, Annotated, Sequence
import snowflake.connector
import google.generativeai as genai
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import Graph, MessageGraph
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QueryAnalyzerAgent:
    """Agent responsible for analyzing and understanding the user's query"""
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        
    def analyze_query(self, query: str) -> Dict:
        """Analyze the query to identify required tables, conditions, and aggregations"""
        prompt = f"""
        Analyze the following query and break it down into components:
        Query: {query}
        
        Identify:
        1. Required tables
        2. Conditions/filters
        3. Any aggregations needed
        4. Time-related requirements
        5. Sorting requirements
        
        Return the analysis in JSON format.
        """
        
        response = self.llm.predict(prompt)
        return json.loads(response)

class SQLGeneratorAgent:
    """Agent responsible for generating SQL queries"""
    def __init__(self, schema_info: str, relationships: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.schema_info = schema_info
        self.relationships = relationships
        
    def generate_sql(self, query_analysis: Dict) -> str:
        prompt = f"""
        Generate a Snowflake SQL query based on this analysis and schema:
        
        Analysis: {json.dumps(query_analysis, indent=2)}
        
        Schema:
        {self.schema_info}
        
        Relationships:
        {self.relationships}
        
        Use Snowflake's date functions where needed:
        - YEAR(date_column) for year extraction
        - DATEADD(year/month/day, number, date) for date arithmetic
        """
        
        return self.llm.predict(prompt)

class DataVisualizerAgent:
    """Agent responsible for creating visualizations of query results"""
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        
    def create_visualization(self, results: Dict, query_type: str) -> None:
        if not results['data']:
            print("No data to visualize")
            return
            
        # Convert results to format suitable for visualization
        columns = results['columns']
        data = results['data']
        
        if query_type == 'time_series':
            self._create_time_series(data, columns)
        elif query_type == 'comparison':
            self._create_bar_chart(data, columns)
        elif query_type == 'distribution':
            self._create_histogram(data, columns)
        else:
            self._create_default_visualization(data, columns)
    
    def _create_time_series(self, data: List, columns: List):
        plt.figure(figsize=(12, 6))
        x = [row[0] for row in data]  # Assuming first column is time
        y = [row[1] for row in data]  # Assuming second column is value
        plt.plot(x, y, marker='o')
        plt.title('Time Series Analysis')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('time_series.png')
        plt.close()
    
    def _create_bar_chart(self, data: List, columns: List):
        plt.figure(figsize=(12, 6))
        x = [str(row[0]) for row in data]
        y = [row[1] for row in data]
        plt.bar(x, y)
        plt.title('Comparison Analysis')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('bar_chart.png')
        plt.close()
    
    def _create_histogram(self, data: List, columns: List):
        plt.figure(figsize=(12, 6))
        values = [row[0] for row in data]
        plt.hist(values, bins=20)
        plt.title('Distribution Analysis')
        plt.xlabel(columns[0])
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('histogram.png')
        plt.close()
    
    def _create_default_visualization(self, data: List, columns: List):
        plt.figure(figsize=(12, 6))
        if len(data[0]) >= 2:
            x = [str(row[0]) for row in data]
            y = [row[1] for row in data]
            plt.bar(x, y)
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
        else:
            x = range(len(data))
            y = [row[0] for row in data]
            plt.bar(x, y)
            plt.xlabel('Index')
            plt.ylabel(columns[0])
        plt.title('Data Visualization')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('default_viz.png')
        plt.close()

class MultiAgentQueryProcessor:
    def __init__(self):
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
        
        # Get schema info and relationships
        self.schema_info = self._get_schema_info()
        self.relationships = self._get_relationships()
        
        # Initialize agents
        self.query_analyzer = QueryAnalyzerAgent()
        self.sql_generator = SQLGeneratorAgent(self.schema_info, self.relationships)
        self.visualizer = DataVisualizerAgent()
        
        # Create the agent graph
        self.graph = self._create_graph()
        
    def _get_schema_info(self) -> str:
        # Implementation same as before
        pass
        
    def _get_relationships(self) -> str:
        # Implementation same as before
        pass
    
    def _create_graph(self) -> Graph:
        def query_analyzer_node(state):
            query = state['query']
            analysis = self.query_analyzer.analyze_query(query)
            return {'analysis': analysis}
            
        def sql_generator_node(state):
            analysis = state['analysis']
            sql = self.sql_generator.generate_sql(analysis)
            return {'sql': sql}
            
        def database_executor_node(state):
            sql = state['sql']
            try:
                self.cursor.execute(sql)
                results = {
                    'columns': [desc[0] for desc in self.cursor.description],
                    'data': self.cursor.fetchall()
                }
                return {'results': results}
            except Exception as e:
                return {'error': str(e)}
                
        def visualizer_node(state):
            if 'error' in state:
                return {'error': state['error']}
            results = state['results']
            analysis = state['analysis']
            self.visualizer.create_visualization(
                results, 
                analysis.get('visualization_type', 'default')
            )
            return {'visualization': 'created'}
        
        # Create the graph
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("analyzer", query_analyzer_node)
        workflow.add_node("sql_generator", sql_generator_node)
        workflow.add_node("executor", database_executor_node)
        workflow.add_node("visualizer", visualizer_node)
        
        # Add edges
        workflow.add_edge("analyzer", "sql_generator")
        workflow.add_edge("sql_generator", "executor")
        workflow.add_edge("executor", "visualizer")
        
        return workflow.compile()
    
    def process_query(self, query: str) -> Dict:
        """Process a natural language query through the agent workflow"""
        try:
            # Initialize the state with the query
            initial_state = {"query": query}
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return final_state
        except Exception as e:
            return {"error": str(e)}
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.snow_conn.close()

def main():
    try:
        processor = MultiAgentQueryProcessor()
        
        while True:
            query = input("\nEnter your question (or 'exit' to quit): ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                break
            
            result = processor.process_query(query)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("\nQuery Analysis:", json.dumps(result['analysis'], indent=2))
                print("\nGenerated SQL:", result['sql'])
                print("\nResults:", result['results'])
                print("\nVisualization has been saved to file.")
        
        processor.close()
        print("Done. Goodbye!")
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
