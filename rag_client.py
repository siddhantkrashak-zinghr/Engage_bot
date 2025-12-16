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
import requests
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
    """
    default_output = {
        "mcp_tools": [],
        "from_date": None,
        "to_date": None
    }
    
    if not json_str:
        return default_output
    
    try:
        # Handle cases where LLM returns a dictionary directly (not string)
        if isinstance(json_str, dict):
            data = json_str
        elif isinstance(json_str, list):
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
                # Regex Fallback
                tool_pattern = r'"tool"\s*:\s*\[([^\]]+)\]|"tool"\s*:\s*"([^"]+)"'
                matches = re.findall(tool_pattern, cleaned)
                if matches:
                    tools = []
                    for match in matches:
                        if match[0]:
                            tools.extend([t.strip(' "\'') for t in match[0].split(',')])
                        elif match[1]:
                            tools.append(match[1])
                    return {"mcp_tools": tools, "from_date": None, "to_date": None}
                return default_output
        else:
            return default_output
        
        # Standardize Data
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
        return default_output
        
    except Exception as e:
        logger.error(f"Parameter parsing error: {e}")
        return default_output


# ============================================================================
# RAG CLIENT
# ============================================================================

class RAGClient:
    """Client for ZingHR RAG API."""
    def __init__(self, base_url: str = "https://mservices-uat.zinghr.com/astra/ragapi", model_name: str = "zinghr-gpt-oss-20b"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name

    def query(self, query_prompt: str, jwt_token: str) -> str:
        """Call RAG endpoint and return content directly."""
        url = f"{self.base_url}/v1/chat/completions"
        
        messages = [
            {"role": "system", "content": jwt_token},
            {"role": "user", "content": query_prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            logger.info(f"Querying RAG endpoint: {self.base_url}")
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                if not content:
                    return "The Knowledge Base returned an empty response."
                return content
            
            return "No content returned from Knowledge Base."
            
        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return f"Error retrieving info from Knowledge Base: {str(e)}"


# ============================================================================
# PROMPT TEMPLATES (Dify Module Logic)
# ============================================================================

RAG_TOOL_DEF = {
    "name": "consult_knowledge_base",
    "description": "MODULE: OTHERS. Use for 'how-to', policies, templates, general HR queries, or anything not covered by specific TNA/PMS/SOCIAL tools.",
    "inputSchema": {}
}

def generate_router_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Generates the router prompt enforcing the Module Classification Logic (TNA, PMS, SOCIAL, OTHERS).
    """
    prompt = """You are an Engage Bot router. Your first task is to classify the user query into one of these modules:

**MODULES**:
1. **TNA** (Time & Attendance): Leaves, balances, attendance, punch status, holidays, regularization, outdoor duty.
2. **PMS** (Performance): Goals, KRAs, objectives, performance reviews.
3. **SOCIAL** (Engagement): Birthdays, anniversaries, surveys, learning courses (LMS), badges, payslips.
4. **OTHERS** (Knowledge Base): Policies, "how-to" guides, templates, processes, general help, or queries not matching the above.

**Current Time**: {current_time}

**Available Tools**:
"""
    # List available tools for the LLM to map to modules
    for i, tool in enumerate(tools, 1):
        prompt += f"- **{tool['name']}**: {tool.get('description', '')}\n"
        
    prompt += """
**DECISION LOGIC**:
1. **Identify the Module** based on the user query.
2. **Select Tool**:
   - If **TNA**, **PMS**, or **SOCIAL**: Pick the specific tool from the list above that matches the intent (e.g., 'get_leave_balance' for leave queries).
   - If **OTHERS** (e.g., "how to apply", "policy", "templates", "resign"): Return **"consult_knowledge_base"**.
3. **Fallback**: If no specific tool matches, default to "consult_knowledge_base".

**Date Rules**:
- Relative dates (e.g., "next Friday") -> Calculate using Current Time.
- No date mentioned -> Use null.

**Output Format (JSON Only)**:
{{
  "tool": ["tool_name"],
  "from_date": "yyyy-mm-dd" or null,
  "to_date": "yyyy-mm-dd" or null
}}
"""
    return prompt


SYNTHESIS_SYSTEM_PROMPT = """You are a professional assistant.
Summarize the following tool data concisely for the user.
Current Time: {current_time}

Tool Data:
{tool_context}
"""


# ============================================================================
# MAIN CHAIN CLASS
# ============================================================================

class EngageBotChain:
    """Main Engage Bot chain using LangChain LCEL."""
    
    def __init__(self, config, mcp_client=None, available_tools: Optional[List[Dict[str, Any]]] = None):
        self.config = config
        self.mcp_client = mcp_client
        
        # Initialize RAG Client
        rag_base = getattr(config, 'rag', None) and getattr(config.rag, 'base_url', None) or "https://mservices-uat.zinghr.com/astra/ragapi"
        rag_model = getattr(config, 'rag', None) and getattr(config.rag, 'model', None) or "zinghr-gpt-oss-20b"
        self.rag_client = RAGClient(base_url=rag_base, model_name=rag_model)

        # Ensure RAG tool is available
        self.available_tools = (available_tools if available_tools is not None else []) + [RAG_TOOL_DEF]
        
        if not self.mcp_client:
            raise ValueError("MCP Client is required.")
        
        # Initialize LLM
        llm_kwargs = {
            "model": config.llm.model,
            "api_key": config.llm.api_key,
            "temperature": config.llm.temperature
        }
        if config.llm.base_url:
            llm_kwargs["base_url"] = config.llm.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        self.chain = self._build_chain()
    
    def _build_chain(self):
        # 1. Router Step
        router_prompt_text = generate_router_system_prompt(self.available_tools)
        
        add_time = RunnablePassthrough.assign(
            current_time=lambda x: get_current_time(self.config)
        )
        
        router_chain = (
            ChatPromptTemplate.from_messages([
                ("system", router_prompt_text),
                ("user", "{query}")
            ])
            | self.llm
            | StrOutputParser()
            | RunnableLambda(clean_llm_response)
            | RunnableLambda(parse_tool_parameters)
        )
        
        # 2. Conditional Execution Step (Module-based Routing)
        def route_and_execute(state: Dict[str, Any]) -> str:
            """
            Executes logic based on Module Classification.
            OTHERS -> RAG (Raw Passthrough)
            TNA/PMS/SOCIAL -> MCP Tools (Synthesized)
            """
            parsed = state["parsed"]
            tools = parsed["mcp_tools"]
            jwt = state.get("jwt", "")
            query = state["query"]
            
            # DEFAULT TO RAG (OTHERS) if no tools selected
            if not tools:
                tools = ["consult_knowledge_base"]

            # --- PATH A: OTHERS (RAG Direct Passthrough) ---
            if "consult_knowledge_base" in tools:
                # Return raw output immediately to preserve formatting
                return self.rag_client.query(query, jwt)
            
            # --- PATH B: TNA/PMS/SOCIAL (MCP Execution + Synthesis) ---
            tool_results = []
            for tool_name in tools:
                try:
                    req = MCPToolRequest(
                        tool_name=tool_name, 
                        jwt_token=jwt,
                        from_date=parsed["from_date"],
                        to_date=parsed["to_date"]
                    )
                    res = self.mcp_client.execute_tool(req)
                    
                    if res.get("status") == "success":
                        tool_results.append(f"Tool: {tool_name}\nData: {json.dumps(res.get('data', {}))}")
                    else:
                        tool_results.append(f"Tool: {tool_name}\nError: {res.get('error', 'Unknown')}")
                        
                except Exception as e:
                    tool_results.append(f"Tool: {tool_name}\nException: {str(e)}")
            
            # Synthesize MCP Results
            context = "\n---\n".join(tool_results)
            synthesis_input = {
                "query": query,
                "tool_context": context,
                "current_time": state["current_time"]
            }
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYNTHESIS_SYSTEM_PROMPT),
                ("user", "{query}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke(synthesis_input)

        # 3. Assemble Final Chain
        return (
            add_time
            | RunnablePassthrough.assign(parsed=router_chain)
            | RunnableLambda(route_and_execute)
        )
    
    def invoke(self, query: str, jwt_token: str = "", **kwargs) -> str:
        try:
            return self.chain.invoke({
                "query": query,
                "jwt": jwt_token,
                **kwargs
            })
        except Exception as e:
            logger.error(f"Chain error: {e}", exc_info=True)
            return "An error occurred. Please try again."