import os
import re
from typing import List, Dict, Any
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from sqlalchemy import create_engine, text
import pandas as pd
from tabulate import tabulate
import sqlite3
from dotenv import load_dotenv

class SQLChatAgent:
    def __init__(self, database_url: str, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the SQL chat agent
        
        Args:
            database_url: SQLAlchemy database URL
            openai_api_key: OpenAI API key
            model_name: OpenAI model name
        """
        self.database_url = database_url
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize database connection
        self.engine = create_engine(database_url)
        self.db = SQLDatabase(self.engine)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            verbose=False
        )
        
        # Initialize SQL toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create custom tools
        self.tools = self._create_tools()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_history_length=10
        )
        
        # Initialize agent
        self.agent = self._initialize_agent()
        
        # Store chat history
        self.chat_history = []
        
        # Store the last executed query
        self.last_query = None
    
    def _create_tools(self) -> List[Tool]:
        """Create the tools for the agent"""
        # Get tools from SQL toolkit
        sql_tools = self.toolkit.get_tools()
        
        # Add custom tools
        custom_tools = [
            Tool(
                name="Format-Results",
                func=self._format_results,
                description="Useful for formatting SQL query results into readable tables. Input should be the query result."
            ),
            Tool(
                name="Get-Database-Schema",
                func=self._get_database_schema,
                description="Useful for getting information about the database schema, tables, and columns."
            )
        ]
        
        return sql_tools + custom_tools
    
    def _initialize_agent(self):
        """Initialize the agent with appropriate configuration"""
        system_message = SystemMessage(content="""
        You are a helpful AI assistant that can convert natural language questions to SQL queries.
        You have access to a SQL database and can execute queries to fetch data.
        
        Guidelines:
        1. Always generate valid SQL queries that work with the connected database
        2. Be careful with SQL injection - sanitize inputs
        3. If you're unsure about the database schema, use the Get-Database-Schema tool
        4. Format the results in a clean, readable tabular format using the Format-Results tool
        5. If a query returns no results, explain this to the user
        6. Handle errors gracefully and provide helpful error messages
        7. Be conversational and friendly in your responses
        8. If the user asks a general question not related to the database, answer politely
        """)
        
        agent_kwargs = {
            "extra_prompts_messages": [MessagesPlaceholder(variable_name="chat_history")],
            "system_message": system_message,
        }
        
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
            memory=self.memory,
            agent_kwargs=agent_kwargs,
            handle_parsing_errors=True
        )
    
    def _format_results(self, query_result: str) -> str:
        """Format SQL query results into a readable table"""
        if not query_result or "no results" in query_result.lower():
            return "No results found for your query."
        
        try:
            # Try to parse the result as a list of dictionaries
            if isinstance(query_result, str) and "|" in query_result:
                # Parse the string result into a list of rows
                lines = query_result.strip().split('\n')
                if len(lines) < 3:
                    return query_result
                
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                data = []
                
                for line in lines[2:]:
                    if line.strip() and not line.startswith('---'):
                        row = [cell.strip() for cell in line.split('|') if cell.strip()]
                        if len(row) == len(headers):
                            data.append(row)
                
                if data:
                    # Format as table
                    table = tabulate(data, headers=headers, tablefmt='grid')
                    return f"Query Results:\n{table}"
            
            return str(query_result)
        except Exception as e:
            return f"Formatted result: {query_result}"
    
    def _get_database_schema(self, _: Any = None) -> str:
        """Get the database schema information"""
        return self.db.get_table_info()
    
    def extract_table_names(self) -> List[str]:
        """Extract table names from the database schema"""
        schema_info = self._get_database_schema()
        # Use regex to find table names in the schema info
        table_pattern = r'CREATE TABLE "(\w+)"'
        tables = re.findall(table_pattern, schema_info)
        return sorted(tables)
    
    def execute_query(self, query: str) -> List[Dict]:
        """Execute a SQL query and return results"""
        try:
            # Store the query for display
            self.last_query = query
            
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
    
    def get_last_query(self) -> str:
        """Get the last executed SQL query"""
        return self.last_query
    
    def chat(self, message: str) -> str:
        """
        Send a message to the chat agent and get a response
        
        Args:
            message: User's message/question
            
        Returns:
            Agent's response
        """
        try:
            # Add user message to chat history
            self.chat_history.append({"role": "user", "content": message})
            
            # Use the agent to handle the question
            response = self.agent.run(message)
            
            # Add agent response to chat history
            self.chat_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            error_msg = f"I encountered an error processing your request: {str(e)}"
            self.chat_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def get_chat_history(self) -> List[Dict]:
        """Get the chat history"""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
        self.memory.clear()
        self.last_query = None
    
    def close(self):
        """Close the database connection"""
        self.engine.dispose()

# Load SQL statements from files
def load_sql_files(sql_folder="sql_statements"):
    """Load SQL statements from files in the specified folder"""
    sql_files = {}
    
    # Create the SQL folder if it doesn't exist
    if not os.path.exists(sql_folder):
        os.makedirs(sql_folder)
        print(f"Created {sql_folder} directory. Please add your SQL files there.")
        return sql_files
    
    # Read all .sql files in the directory
    for filename in os.listdir(sql_folder):
        if filename.endswith('.sql'):
            filepath = os.path.join(sql_folder, filename)
            with open(filepath, 'r') as file:
                sql_files[filename] = file.read()
    
    return sql_files

# Example database setup using SQL files
def create_database_from_sql(db_folder="db", db_name="example.db", sql_folder="sql_statements"):
    """Create a SQLite database from SQL files in the specified folder"""
    
    # Create the db directory if it doesn't exist
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)
    
    db_path = os.path.join(db_folder, db_name)
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Load SQL files
    sql_files = load_sql_files(sql_folder)
    
    if not sql_files:
        raise ValueError("No SQL files found.")
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Execute all SQL statements
    for filename, sql_content in sql_files.items():
        print(f"Executing SQL from {filename}...")
        
        # Split SQL content into individual statements
        statements = sql_content.split(';')
        
        for statement in statements:
            statement = statement.strip()
            if statement:  # Only execute non-empty statements
                try:
                    print(f"Executing statement: {statement}")
                    cursor.execute(statement)
                except sqlite3.Error as e:
                    print(f"Error executing statement from {filename}: {e}")
                    print(f"Statement: {statement}")
    
    conn.commit()
    conn.close()
    
    return db_path

# Load environment variables
def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. "
                       "Please create a .env file with your OpenAI API key.")
    
    return api_key

# Interactive chat function
def interactive_chat():
    """Run an interactive chat with the SQL agent"""
    
    # Load environment variables
    try:
        OPENAI_API_KEY = load_environment()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Configuration
    DB_FOLDER = "db"
    DB_NAME = "example.db"
    SQL_FOLDER = "sql_statements"
    
    # Create the database from SQL files
    print(f"Creating database from SQL files in '{SQL_FOLDER}' folder...")
    db_path = create_database_from_sql(DB_FOLDER, DB_NAME, SQL_FOLDER)
    
    # Create SQLAlchemy URL
    DATABASE_URL = f"sqlite:///{db_path}"
    
    # Initialize the agent
    print("Initializing SQL chat agent...")
    agent = SQLChatAgent(DATABASE_URL, OPENAI_API_KEY)
    
    # Get schema information and extract table names
    schema_info = agent._get_database_schema()
    table_names = agent.extract_table_names()
    
    print("\n" + "="*60)
    print("SQL Chat Agent is ready!")
    print(f"Database location: {db_path}")
    print(f"Available tables: {', '.join(table_names)}")
    print("You can ask questions about the database in natural language.")
    print("Type 'quit' to exit, 'history' to see chat history, or 'clear' to clear history.")
    print("="*60 + "\n")
    
    print("Database schema loaded successfully!")
    print("Please ask your question in the chat:")

    # Chat loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        elif user_input.lower() in ['history', 'h']:
            history = agent.get_chat_history()
            if not history:
                print("No chat history yet.")
            else:
                print("\nChat History:")
                for i, msg in enumerate(history, 1):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    print(f"{i}. {role}: {msg['content']}")
                print()
        elif user_input.lower() in ['clear', 'c']:
            agent.clear_chat_history()
            print("Chat history cleared.")
        elif user_input.lower() in ['schema', 'tables']:
            # Show schema information
            print(f"\nAvailable tables: {', '.join(table_names)}")
            print("\nSchema preview:")
            # Show a brief preview of each table structure
            for table in table_names:
                print(f"\n{table} table structure:")
                table_info = re.search(f'CREATE TABLE "{table}" \((.*?)\)', schema_info, re.DOTALL)
                if table_info:
                    columns = table_info.group(1).split('\n')
                    for col in columns[:4]:  # Show first few columns
                        col = col.strip().strip(',')
                        if col:
                            print(f"  - {col}")
                if table == table_names[-1]:
                    print("  - ... (more columns)")
            print()
        elif user_input.lower() in ['query', 'lastquery']:
            # Show the last executed query
            last_query = agent.get_last_query()
            if last_query:
                print(f"\nLast executed query:\n{last_query}\n")
            else:
                print("\nNo query has been executed yet.\n")
        else:
            # Get response from agent
            response = agent.chat(user_input)
            
            # Show the query that was used
            last_query = agent.get_last_query()
            if last_query:
                print(f"\nExecuted query:\n{last_query}\n")
            
            print(f"Assistant: {response}\n")
    
    # Clean up
    agent.close()

if __name__ == "__main__":
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Add your OpenAI API key here\n")
            f.write("OPENAI_API_KEY=your-openai-api-key-here\n")
        print("Created .env file. Please add your OpenAI API key to it.")
    
    # Check if we're running in an environment that supports input()
    try:
        input("Press Enter to continue (make sure you've added your API key to .env file)...")
        interactive_chat()
    except Exception as e:
        print(f"Error: {e}")