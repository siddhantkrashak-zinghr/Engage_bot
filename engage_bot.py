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
    """
    default_output = {
        "mcp_tools": [],
        "from_date": None,
        "to_date": None
    }
    
    if not json_str:
        return default_output
    
    try:
        if isinstance(json_str, dict):
            data = json_str
        elif isinstance(json_str, list):
            logger.warning(f"LLM returned list instead of dict: {json_str}")
            return {
                "mcp_tools": [str(item) for item in json_str if item],
                "from_date": None,
                "to_date": None
            }
        elif isinstance(json_str, str):
            # Clean and parse JSON string
            cleaned = json_str.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(
                    line for line in lines 
                    if not line.strip().startswith("```")
                )
            
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # Fallback: simple regex extraction if JSON parsing fails
                logger.warning(f"Failed to parse JSON, attempting regex extraction: {cleaned[:200]}")
                tool_pattern = r'"tool"\s*:\s*\[([^\]]+)\]|"tool"\s*:\s*"([^"]+)"'
                matches = re.findall(tool_pattern, cleaned)
                
                if matches:
                    tools = []
                    for match in matches:
                        if match[0]:
                            tools.extend([t.strip(' "\'') for t in match[0].split(',')])
                        elif match[1]:
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
# PROMPT TEMPLATES (Dynamic Router Prompt)
# ============================================================================

def generate_router_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """Dynamically generates the router system prompt based on available tools."""
    
    prompt = """You are an Engage Bot router. Analyze user queries and identify the necessary MCP tools to fetch data.

**Current Time Source**: {current_time}

**Available MCP Tools**:
"""
    # Add Tools dynamically
    for i, tool in enumerate(tools, 1):
        # We assume the MCP definition contains name, description, and optionally keywords
        description = tool.get('description', 'No description available')
        
        # Simple keyword extraction (can be enhanced if tool schema contains explicit keywords)
        keywords = ", ".join(
            re.findall(r'[a-zA-Z]+', description.lower())
        )
        
        prompt += f"{i}. **{tool['name']}**: {description}\n"
        prompt += f"   - Keywords (Inferred): {keywords}\n"
        
    # Add General Rules
    prompt += """
**IMPORTANT**: If the query is about a topic NOT covered by the tools above, return an empty tool array.

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
3. Return empty array [] if query is unrelated to any available tool
4. Single tool returns string, multiple tools return array

**Examples**:
User: "Show me birthdays today"
Output: {{"tool": "get_wishes_data", "from_date": null, "to_date": null}}

User: "What surveys are pending?"
Output: {{"tool": "get_survey_data", "from_date": null, "to_date": null}}

User: "Show birthdays and surveys"
Output: {{"tool": ["get_wishes_data", "get_survey_data"], "from_date": null, "to_date": null}}
"""
    return prompt


SYNTHESIS_SYSTEM_PROMPT = """You are a professional assistant helping users with employee engagement data.

**Tasks**:
1. Understand user query: {query}
2. Use provided tool data: {tool_context}
3. Answer concisely and professionally

**Rules**:
- **Language**: Detect and respond in the SAME language as the query
- **Formatting**: Use clean markdown
- **Detail Control**: Provide detailed, well-structured responses
- **Time Handling**: Current time (internal use): {current_time}
- **Error Handling**:
  - Authentication errors: "Please check if you are logged in."
  - No tools executed: Provide a helpful message indicating the query is outside the scope of available tools.
  - Other errors: Professional, user-friendly message
- **Tone**: Direct, professional, avoid redundancy
- **Out of Scope**: If the query is not about the data fetched by the available tools, politely inform the user that this is outside your current scope.

**Output**: Markdown-formatted answer (summary optional)"""


# ============================================================================
# LANGCHAIN LCEL CHAIN CONSTRUCTION
# ============================================================================

class EngageBotChain:
    """Main Engage Bot chain using LangChain LCEL."""
    
    def __init__(self, config, mcp_client=None, available_tools: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize Engage Bot Chain.
        
        Args:
            config: EngageBotConfig instance
            mcp_client: MCPClient instance (required for tool execution)
            available_tools: List of tool definitions for dynamic prompt generation.
        """
        self.config = config
        self.mcp_client = mcp_client
        self.available_tools = available_tools if available_tools is not None else []
        
        if not self.mcp_client:
            raise ValueError("MCP Client is required for EngageBotChain.")
        
        # Initialize LLM
        llm_kwargs = {
            "model": config.llm.model,
            "api_key": config.llm.api_key,
            "temperature": config.llm.temperature
        }
        if config.llm.base_url:
            llm_kwargs["base_url"] = config.llm.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Build chain
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the complete LCEL chain."""
        
        # 1. Generate the dynamic router prompt
        router_prompt_text = generate_router_system_prompt(self.available_tools)
        
        # 2. Add current time
        add_time = RunnablePassthrough.assign(
            current_time=lambda x: get_current_time(self.config)
        )
        
        # 3. Router LLM - extract tools and dates
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", router_prompt_text),
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
        
        # 4. Execute MCP tools (iteration)
        def execute_tools(state: Dict[str, Any]) -> List[str]:
            """Execute all MCP tools and collect results."""
            tools = state["parsed"]["mcp_tools"]
            jwt = state.get("jwt", "")
            from_date = state["parsed"]["from_date"]
            to_date = state["parsed"]["to_date"]
            
            if not tools:
                return ["No relevant tools identified for this query. This may be outside the scope of available tools on the MCP server."]
            
            results = []
            client = self.mcp_client # Use the injected client
            
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
        
        # 5. Final synthesis LLM
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