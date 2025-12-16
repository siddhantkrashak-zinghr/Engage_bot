"""
Engage Bot - Main Entry Point

This is the main file to run the Engage Bot application.
It demonstrates how to use the EngageBotChain with proper configuration.

Usage:
    python main.py

Environment:
    Configure via .env file (see .env.example)
"""

import sys
import logging
from typing import Dict, Any, List

from config import EngageBotConfig
from engage_bot import EngageBotChain
from mcp_client import create_mcp_client, MCPClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('engage_bot.log')
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                    ENGAGE BOT v1.1                           ║
    ║              LangChain LCEL Implementation                   ║
    ║                                                               ║
    ║      Employee Engagement Assistant (MCP + RAG Enabled)       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_available_tools(mcp_client: MCPClient, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch and print a limited list of available tools."""
    logger.info("Fetching available MCP tools...")
    
    # MCPClient's list_tools is synchronous and caches the result
    available_tools = mcp_client.list_tools()
    
    print("\n🔍 Available MCP Tools (First 10):")
    if not available_tools:
        print("    - No tools found on MCP server. Check connection.")
        return available_tools
    
    for i, tool in enumerate(available_tools):
        if i >= limit:
            print(f"    ... and {len(available_tools) - limit} more tools.")
            break
        print(f"    - **{tool['name']}**: {tool.get('description', 'No description')}")
        
    print("─" * 70)
    return available_tools


def get_user_jwt_token() -> str:
    """Ask the user for their JWT token."""
    # Use the example token from get_sample_queries as a prompt hint
    example_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.your-actual-jwt-token"
    
    print(f"\n🔑 Authentication Required.")
    jwt_token = input(f"   Enter your JWT token: ").strip()
    
    if not jwt_token:
        logger.warning("No JWT token provided. Tool execution may fail.")
    return jwt_token


def run_interactive_mode(bot: EngageBotChain, jwt_token: str):
    """
    Run bot in interactive CLI mode.
    """
    print("\n🤖 Interactive Mode - Type 'exit' or 'quit' to stop\n")
    print("─" * 70)
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            print("\n🔄 Processing...\n")
            response = bot.invoke(query=query, jwt_token=jwt_token)
            
            print(f"🤖 Bot:\n{response}\n")
            print("─" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")


def run_batch_mode(bot: EngageBotChain, queries: list[Dict[str, str]]):
    """
    Run bot in batch mode with predefined queries.
    """
    print("\n📋 Batch Mode - Processing predefined queries\n")
    print("═" * 70)
    
    for i, item in enumerate(queries, 1):
        query = item.get("query", "")
        # Use the master JWT token fetched at startup for batch execution
        jwt = item.get("jwt", "")
        
        print(f"\n[Query {i}/{len(queries)}]")
        print(f"💬 Question: {query}")
        print("─" * 70)
        
        try:
            response = bot.invoke(query=query, jwt_token=jwt)
            print(f"\n🤖 Response:\n{response}\n")
            
        except Exception as e:
            logger.error(f"Error processing query {i}: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")
        
        print("═" * 70)


def run_single_query_mode(bot: EngageBotChain, jwt_token: str):
    """Run bot for a single query."""
    query = input("\n💬 Enter your query: ").strip()
    
    print("\n🔄 Processing...\n")
    response = bot.invoke(query=query, jwt_token=jwt_token)
    print(f"🤖 Response:\n{response}\n")
    print("─" * 70)


def get_sample_queries(jwt_token: str) -> list[Dict[str, str]]:
    """
    Get sample queries for testing, using the provided JWT token.
    Updated to include RAG (Policy/Knowledge) queries.
    """
    return [
        {
            "query": "Who all are celebrating their birthday today?",
            "jwt": jwt_token
        },
        {
            "query": "What are my pending surveys?",
            "jwt": jwt_token
        },
        {
            "query": "What are the leave policy rules?",  # RAG Query
            "jwt": jwt_token
        },
        {
            "query": "Show me the learning courses available",
            "jwt": jwt_token
        },
        {
            "query": "How do I apply for reimbursement?",  # RAG Query
            "jwt": jwt_token
        },
        {
            "query": "Which employees received badges last month?",
            "jwt": jwt_token
        }
    ]


def main():
    """Main application entry point"""
    print_banner()
    
    mcp_client = None # Initialize client to None for cleanup
    
    try:
        # 1. Configuration and Client Setup
        logger.info("Loading configuration...")
        config = EngageBotConfig.from_env()
        config.validate()
        
        # Create MCP client
        mcp_client = create_mcp_client(
            base_url=config.mcp.base_url,
            mock_mode=config.enable_mock_mode,
            timeout=config.mcp.timeout,
            max_retries=config.mcp.max_retries
        )
        
        # 2. Dynamic Tool Discovery & Display
        available_tools = print_available_tools(mcp_client)
        
        # 3. Initialize Bot with Dynamic Tools
        # Note: RAG Client is initialized internally by EngageBotChain using config
        logger.info("Initializing Engage Bot with dynamic tools and RAG...")
        bot = EngageBotChain(
            config=config, 
            mcp_client=mcp_client, 
            available_tools=available_tools
        )
        logger.info("✅ Engage Bot initialized successfully!")
        
        # 4. Mode Selection
        print("\nSelect Mode:")
        print("1. Interactive Mode (CLI)")
        print("2. Batch Mode (predefined queries)")
        print("3. Single Query Test")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        # 5. Get JWT Token (Required for all execution modes)
        jwt_token = ""
        queries = []
        if choice in ["1", "3"]:
            jwt_token = get_user_jwt_token()
        elif choice == "2":
            # For batch mode, ask for a master token, then generate queries
            jwt_token = get_user_jwt_token()
            queries = get_sample_queries(jwt_token)
            
        # 6. Execute Mode
        if not jwt_token:
            print("\n🚨 Cannot proceed without a JWT token. Exiting.")
            return

        print(f"\n🔐 Using JWT token: {jwt_token[:10]}...{jwt_token[-4:]}")
        
        if choice == "1":
            run_interactive_mode(bot, jwt_token)
            
        elif choice == "2":
            run_batch_mode(bot, queries)
            
        elif choice == "3":
            run_single_query_mode(bot, jwt_token)
            
        else:
            print("Invalid choice. Exiting.")
            return
        
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n❌ Fatal Error: {e}\n")
        sys.exit(1)
    finally:
        # 7. Cleanup
        if mcp_client:
            mcp_client.close()
            logger.info("MCP Client closed")


if __name__ == "__main__":
    main()