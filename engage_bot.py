"""
Engage Bot - LangChain LCEL Implementation
Replicates Dify workflow for employee engagement queries using MCP tools.

File: engage_bot.py
Main implementation file containing the complete LCEL chain logic.
"""

import os
import json
import re
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from mcp_client import create_mcp_client, MCPToolRequest

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel
)
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_current_time(config) -> str:
    """Get current time in configured timezone and format."""
    try:
        tz = ZoneInfo(config.time.timezone)
        return datetime.now(tz).strftime(config.time.time_format)
    except Exception as e:
        logger.warning(f"Timezone error: {e}. Using local time.")
        return datetime.now().strftime(config.time.time_format)


def clean_llm_response(text: str) -> str:
    """Remove <think>...</think> tags from LLM response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_tool_parameters(json_str: Union[str, dict, list]) -> Dict[str, Any]:
    """
    Parse LLM output to extract tool names and date parameters.
    
    Expected format:
    {
        "tool": ["tool_name1", "tool_name2"] or "tool_name",
        "from_date": "yyyy-mm-dd" or null,
        "to_date": "yyyy-mm-dd" or null
    }
    
    Args:
        json_str: Can be a string, dict, or list from LLM
        
    Returns:
        Dictionary with mcp_tools, from_date, to_date
    """
    default_output = {
        "mcp_tools": [],
        "from_date": None,
        "to_date": None
    }
    
    # Handle empty input
    if not json_str:
        return default_output
    
    try:
        # If already a dict, use it directly
        if isinstance(json_str, dict):
            data = json_str
        # If it's a list, LLM might have returned array directly
        elif isinstance(json_str, list):
            logger.warning(f"LLM returned list instead of dict: {json_str}")
            # Treat list items as tool names
            return {
                "mcp_tools": [str(item) for item in json_str if item],
                "from_date": None,
                "to_date": None
            }
        # If it's a string, parse it
        elif isinstance(json_str, str):
            # Remove markdown code fences if present
            cleaned = json_str.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(
                    line for line in lines 
                    if not line.strip().startswith("```")
                )
            
            # Try to parse as JSON
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # If parsing fails, try to extract tool names with regex
                logger.warning(f"Failed to parse JSON, attempting regex extraction: {cleaned[:200]}")
                tool_pattern = r'"tool"\s*:\s*\[([^\]]+)\]|"tool"\s*:\s*"([^"]+)"'
                matches = re.findall(tool_pattern, cleaned)
                
                if matches:
                    tools = []
                    for match in matches:
                        if match[0]:  # Array format
                            tools.extend([t.strip(' "\'') for t in match[0].split(',')])
                        elif match[1]:  # String format
                            tools.append(match[1])
                    
                    return {
                        "mcp_tools": tools,
                        "from_date": None,
                        "to_date": None
                    }
                
                return default_output
        else:
            logger.warning(f"Unexpected type for json_str: {type(json_str)}")
            return default_output
        
        # Handle dict data
        if isinstance(data, dict):
            # Extract and normalize tool parameter
            tool_value = data.get("tool", [])
            if isinstance(tool_value, list):
                mcp_tools = [str(t) for t in tool_value if t]
            elif isinstance(tool_value, str) and tool_value:
                mcp_tools = [tool_value]
            else:
                mcp_tools = []
            
            return {
                "mcp_tools": mcp_tools,
                "from_date": data.get("from_date"),
                "to_date": data.get("to_date")
            }
        # Handle list data (shouldn't happen but just in case)
        elif isinstance(data, list):
            return {
                "mcp_tools": [str(item) for item in data if item],
                "from_date": None,
                "to_date": None
            }
        else:
            logger.warning(f"Parsed data is not dict or list: {type(data)}")
            return default_output
        
    except Exception as e:
        logger.error(f"Unexpected error in parse_tool_parameters: {e}", exc_info=True)
        logger.error(f"Input was: {json_str}")
        return default_output


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

ROUTER_SYSTEM_PROMPT = """You are an Engage Bot router. Analyze user queries and identify the necessary MCP tools to fetch data.

**Current Time Source**: {current_time}

**Available MCP Tools**:
1. **get_wishes_data**: Fetch birthday, anniversary, new joinees list
   - Keywords: birthday, bday, anniversary, new employees, new joinees, newly onboarded
   
2. **get_learning_courses**: Fetch learning courses
   - Keywords: learning, lms, training, courses, skill development
   
3. **get_survey_data**: Fetch survey data, status, or pending responses
   - Keywords: survey, poll, feedback, questionnaire, status, pending, response, completion
   
4. **get_badges_with_reaction**: Fetch badges/recognition for entire organization
   - Keywords: badges, awards, recognition, organizational badges, company recognition
   
5. **get_my_badges_with_reaction**: Fetch badges/recognition for user and team
   - Keywords: my badges, my awards, team badges, my recognition
   
6. **get_preference_data**: Fetch list of languages supported on platform
   - Keywords: language, languages, preference, supported languages
7. get_leaves_balance: Fetch leave balance information for the employee
   - Keywords: leave balance, leaves left, remaining leaves, my leaves, leave quota, leave summary, available leaves, balance of leaves

**IMPORTANT**: If the query is about leave, holidays, attendance, payroll, salary, or any topic NOT covered by the tools above, return an empty tool array.

**Date Rules**:
- For relative dates (e.g., "next Friday"), calculate using current_time_source
- If no day mentioned, use 1st of that month for from_date (e.g., "last month", "august 2025")
- Use yyyy-mm-dd format
- If no date mentioned, use null

**Output Format** (JSON only - no explanation, no markdown):
{{
  "tool": ["<tool_name1>", "<tool_name2>"] or "<tool_name>",
  "from_date": "yyyy-mm-dd" or null,
  "to_date": "yyyy-mm-dd" or null
}}

**Rules**:
1. Return ONLY valid JSON - no additional text
2. Map query keywords to appropriate tools
3. Return empty array [] if query is unrelated to Engage tools
4. Single tool returns string, multiple tools return array
5. Do NOT add tools for leave, attendance, or payroll queries

**Examples**:
User: "Show me birthdays today"
Output: {{"tool": "get_wishes_data", "from_date": null, "to_date": null}}

User: "What surveys are pending?"
Output: {{"tool": "get_survey_data", "from_date": null, "to_date": null}}

User: "Show birthdays and surveys"
Output: {{"tool": ["get_wishes_data", "get_survey_data"], "from_date": null, "to_date": null}}

User: "What's my leave balance?"
Output: {{"tool": [get_leave_balance], "from_date": null, "to_date": null}}"""


SYNTHESIS_SYSTEM_PROMPT = """You are a professional assistant helping users with employee engagement data.

**Tasks**:
1. Understand user query: {query}
2. Use provided tool data: {tool_context}
3. Answer concisely and professionally

**Rules**:
- **Language**: Detect and respond in the SAME language as the query
- **Formatting**: Use clean markdown
- **Detail Control**:
  - Default: Provide detailed, well-structured responses
  - Simple queries: Brief, direct answers
  - Complex queries: Comprehensive explanations
- **Time Handling**: 
  - Current time (internal use): {current_time}
  - Only mention time if query requires it
  - Use date range: {from_date} to {to_date}
- **Error Handling**:
  - Authentication errors: "Please check if you are logged in."
  - No tools executed: Provide a helpful message directing user to appropriate resources
  - Other errors: Professional, user-friendly message
- **Tone**: Direct, professional, avoid redundancy
- **Out of Scope**: If the query is not about engagement data (wishes, surveys, learning, badges, preferences), politely inform the user that this is outside your scope and suggest they check the appropriate system or contact HR.

**Output**: Markdown-formatted answer (summary optional)"""


# ============================================================================
# LANGCHAIN LCEL CHAIN CONSTRUCTION
# ============================================================================

class EngageBotChain:
    """Main Engage Bot chain using LangChain LCEL."""
    
    def __init__(self, config, mcp_client=None):
        """
        Initialize Engage Bot Chain.
        
        Args:
            config: EngageBotConfig instance
            mcp_client: MCPClient instance (optional, for dependency injection)
        """
        self.config = config
        
        # Initialize LLM
        llm_kwargs = {
            "model": config.llm.model,
            "api_key": config.llm.api_key,
            "temperature": config.llm.temperature
        }
        if config.llm.base_url:
            llm_kwargs["base_url"] = config.llm.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Store MCP client (injected or will be created per request)
        self.mcp_client = mcp_client
        
        # Build chain
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the complete LCEL chain."""
        
        # Step 1: Add current time
        add_time = RunnablePassthrough.assign(
            current_time=lambda x: get_current_time(self.config)
        )
        
        # Step 2: Router LLM - extract tools and dates
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("user", "{query}")
        ])
        
        router_chain = (
            router_prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(clean_llm_response)
            | RunnableLambda(parse_tool_parameters)
        )
        
        add_parsed = RunnablePassthrough.assign(parsed=router_chain)
        
        # Step 3: Execute MCP tools (iteration)
        def execute_tools(state: Dict[str, Any]) -> List[str]:
            """Execute all MCP tools and collect results."""
            tools = state["parsed"]["mcp_tools"]
            jwt = state.get("jwt", "")
            from_date = state["parsed"]["from_date"]
            to_date = state["parsed"]["to_date"]
            
            if not tools:
                return ["No relevant tools identified for this query. This may be outside the scope of engagement data (wishes, surveys, learning, badges, preferences)."]
            
            results = []
            
            # Use injected client or create one for this request
            if self.mcp_client:
                client = self.mcp_client
            else:
                client = create_mcp_client(
                    base_url=self.config.mcp.base_url,
                    mock_mode=self.config.enable_mock_mode,
                    timeout=self.config.mcp.timeout
                )
            
            # Execute each tool
            for tool_name in tools:
                try:
                    # Create request object
                    request = MCPToolRequest(
                        tool_name=tool_name,
                        jwt_token=jwt,
                        from_date=from_date,
                        to_date=to_date
                    )
                    
                    # Execute tool
                    result = client.execute_tool(request)
                    
                    # Format result for LLM
                    formatted = self._format_tool_result(result)
                    results.append(formatted)
                    
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    results.append(
                        f"Tool: {tool_name}\nStatus: Error\n"
                        f"Message: {str(e)}\n---"
                    )
            
            return results
        
        add_tool_results = RunnablePassthrough.assign(
            tool_results=RunnableLambda(execute_tools)
        )
        
        # Step 4: Final synthesis LLM
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("user", "{query}")
        ])
        
        def prepare_synthesis_input(state: Dict[str, Any]) -> Dict[str, Any]:
            """Prepare input for synthesis LLM."""
            return {
                "query": state["query"],
                "tool_context": "\n".join(state["tool_results"]),
                "current_time": state["current_time"],
                "from_date": state["parsed"]["from_date"] or "N/A",
                "to_date": state["parsed"]["to_date"] or "N/A"
            }
        
        synthesis_chain = (
            RunnableLambda(prepare_synthesis_input)
            | synthesis_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Complete chain
        return (
            add_time
            | add_parsed
            | add_tool_results
            | synthesis_chain
        )
    
    def _format_tool_result(self, result: Dict[str, Any]) -> str:
        """
        Format tool execution result for LLM context.
        
        Args:
            result: Dictionary with keys: tool_name, status, data/error
            
        Returns:
            Formatted string for LLM
        """
        tool_name = result.get("tool_name", "unknown")
        status = result.get("status", "unknown")
        
        if status == "success":
            data = result.get("data", {})
            return (
                f"Tool: {tool_name}\n"
                f"Status: Success\n"
                f"Data:\n{json.dumps(data, indent=2)}\n"
                f"---"
            )
        else:
            error = result.get("error", "Unknown error")
            return (
                f"Tool: {tool_name}\n"
                f"Status: Error\n"
                f"Error: {error}\n"
                f"---"
            )
    
    def invoke(self, query: str, jwt_token: str = "", **kwargs) -> str:
        """
        Invoke the Engage Bot chain.
        
        Args:
            query: User query
            jwt_token: JWT authentication token
            **kwargs: Additional parameters (emp_info, emp_attr, etc.)
        
        Returns:
            Final answer as string
        """
        try:
            input_data = {
                "query": query,
                "jwt": jwt_token,
                **kwargs
            }
            
            result = self.chain.invoke(input_data)
            return result
            
        except Exception as e:
            logger.error(f"Chain execution error: {e}", exc_info=True)
            return f"An error occurred while processing your request. Please try again."


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    from config import EngageBotConfig
    
    # Initialize configuration
    config = EngageBotConfig()
    
    # Create bot instance
    bot = EngageBotChain(config)
    
    # Example queries
    test_queries = [
        {
            "query": "Who all are celebrating their bday today and what are my pending surveys?",
            "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        },
        {
            "query": "What learning courses are available for me?",
            "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        },
        {
            "query": "Show me badges received by the team last month",
            "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        }
    ]
    
    print("=" * 70)
    print("ENGAGE BOT - LANGCHAIN LCEL IMPLEMENTATION")
    print("=" * 70)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {test['query']}")
        print(f"{'─' * 70}")
        
        try:
            answer = bot.invoke(
                query=test["query"],
                jwt_token=test["jwt"]
            )
            print(f"\n{answer}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("=" * 70)


if __name__ == "__main__":
    main()