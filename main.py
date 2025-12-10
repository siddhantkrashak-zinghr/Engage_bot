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
from typing import Dict, Any

from config import EngageBotConfig
from engage_bot import EngageBotChain
from mcp_client import create_mcp_client

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
    ║                    ENGAGE BOT v1.0                           ║
    ║              LangChain LCEL Implementation                   ║
    ║                                                               ║
    ║           Employee Engagement Query Assistant                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_interactive_mode(bot: EngageBotChain, jwt_token: str):
    """
    Run bot in interactive CLI mode.
    
    Args:
        bot: EngageBotChain instance
        jwt_token: JWT authentication token
    """
    print("\n🤖 Interactive Mode - Type 'exit' or 'quit' to stop\n")
    print("─" * 70)
    
    while True:
        try:
            # Get user input
            query = input("\n💬 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            # Process query
            print("\n🔄 Processing...\n")
            response = bot.invoke(query=query, jwt_token=jwt_token)
            
            # Display response
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
    
    Args:
        bot: EngageBotChain instance
        queries: List of query dictionaries with 'query' and 'jwt' keys
    """
    print("\n📋 Batch Mode - Processing predefined queries\n")
    print("═" * 70)
    
    for i, item in enumerate(queries, 1):
        query = item.get("query", "")
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


def get_sample_queries() -> list[Dict[str, str]]:
    """
    Get sample queries for testing.
    
    Returns:
        List of query dictionaries
    """
    # TODO: Replace with your actual JWT token
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.your-actual-jwt-token"
    
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
            "query": "Show me the learning courses available",
            "jwt": jwt_token
        },
        {
            "query": "Which employees received badges last month?",
            "jwt": jwt_token
        },
        {
            "query": "What languages does the platform support?",
            "jwt": jwt_token
        },
        {
            "query": "Who's celebrating work anniversary this week?",
            "jwt": jwt_token
        }
    ]


def main():
    """Main application entry point"""
    print_banner()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = EngageBotConfig.from_env()
        config.validate()
        
        logger.info(f"LLM Model: {config.llm.model}")
        logger.info(f"MCP URL: {config.mcp.base_url}")
        logger.info(f"Mock Mode: {config.enable_mock_mode}")
        
        # Create MCP client
        mcp_client = create_mcp_client(
            base_url=config.mcp.base_url,
            mock_mode=config.enable_mock_mode,
            timeout=config.mcp.timeout,
            max_retries=config.mcp.max_retries
        )
        
        # Initialize bot
        logger.info("Initializing Engage Bot...")
        bot = EngageBotChain(config=config, mcp_client=mcp_client)
        logger.info("✅ Engage Bot initialized successfully!")
        
        # Choose mode
        print("\nSelect Mode:")
        print("1. Interactive Mode (CLI)")
        print("2. Batch Mode (predefined queries)")
        print("3. Single Query Test")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            # Interactive mode
            jwt_token = input("\nEnter JWT token: ").strip()
            if not jwt_token:
                logger.warning("No JWT token provided. Using empty token.")
            run_interactive_mode(bot, jwt_token)
            
        elif choice == "2":
            # Batch mode
            queries = get_sample_queries()
            run_batch_mode(bot, queries)
            
        elif choice == "3":
            # Single query test
            query = input("\nEnter your query: ").strip()
            jwt_token = input("Enter JWT token: ").strip()
            
            print("\n🔄 Processing...\n")
            response = bot.invoke(query=query, jwt_token=jwt_token)
            print(f"🤖 Response:\n{response}\n")
            
        else:
            print("Invalid choice. Exiting.")
            return
        
        # Cleanup
        mcp_client.close()
        logger.info("Application completed successfully")
        
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n❌ Fatal Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()