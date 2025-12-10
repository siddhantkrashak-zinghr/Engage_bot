"""
MCP Client for making API calls to ZingHR MCP Server.

File: mcp_client.py

This module handles all interactions with the MCP (Model Context Protocol) server,
including tool execution and response handling.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import asyncio

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# MCP SSE Client imports
from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


@dataclass
class MCPToolRequest:
    """Request structure for MCP tool execution"""
    tool_name: str
    jwt_token: str
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    additional_params: Dict[str, Any] = None


class MCPClient:
    """
    Client for interacting with ZingHR MCP Server using SSE.
    
    This client handles:
    - Tool execution via SSE endpoint
    - Request retries and timeout handling
    - Response parsing and error handling
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize MCP Client.
        
        Args:
            base_url: Base URL of MCP server (should end with /sse)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        # Ensure URL ends with /sse for SSE connection
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/sse'):
            self.base_url = f"{self.base_url}/sse"
        
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Cache for tools list
        self._tools_cache: Optional[List] = None
    
    def execute_tool(self, request: MCPToolRequest) -> Dict[str, Any]:
        """
        Execute a single MCP tool using SSE connection.
        
        Args:
            request: MCPToolRequest containing tool details
            
        Returns:
            Dictionary containing tool execution result
        """
        try:
            # Run async execution in sync context
            return asyncio.run(self._execute_tool_async(request))
            
        except Exception as e:
            logger.error(f"Tool execution failed for {request.tool_name}: {e}")
            return self._error_response(request.tool_name, str(e))
    
    async def _execute_tool_async(self, request: MCPToolRequest) -> Dict[str, Any]:
        """
        Async execution of MCP tool via SSE.
        
        This matches the working code pattern from your reference.
        """
        try:
            # Connect to MCP server via SSE
            async with sse_client(url=self.base_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize session
                    await session.initialize()
                    
                    # Prepare arguments
                    arguments = {
                        "jwt_token": request.jwt_token
                    }
                    
                    # Add optional parameters
                    if request.from_date:
                        arguments["from_date"] = request.from_date
                    if request.to_date:
                        arguments["to_date"] = request.to_date
                    if request.additional_params:
                        arguments.update(request.additional_params)
                    
                    # Call the tool
                    logger.info(f"Calling MCP tool: {request.tool_name}")
                    response = await session.call_tool(
                        request.tool_name,
                        arguments=arguments
                    )
                    
                    # Parse response
                    return self._parse_mcp_response(response, request.tool_name)
                    
        except Exception as e:
            logger.error(f"MCP SSE execution failed: {e}", exc_info=True)
            return self._error_response(request.tool_name, str(e))
    
    def _parse_mcp_response(self, response: Any, tool_name: str) -> Dict[str, Any]:
        """
        Parse MCP server response from SSE.
        
        Args:
            response: Response object from session.call_tool()
            tool_name: Name of the tool that was called
            
        Returns:
            Standardized response dictionary
        """
        try:
            # Extract content from response
            if hasattr(response, 'content') and response.content:
                # Combine all text content
                result_text = ""
                for content in response.content:
                    if hasattr(content, 'type') and content.type == "text":
                        result_text += content.text
                
                # Try to parse as JSON
                try:
                    data = json.loads(result_text)
                    return {
                        "tool_name": tool_name,
                        "status": "success",
                        "data": data
                    }
                except json.JSONDecodeError:
                    # Return as plain text if not JSON
                    return {
                        "tool_name": tool_name,
                        "status": "success",
                        "data": {"text": result_text}
                    }
            else:
                return {
                    "tool_name": tool_name,
                    "status": "success",
                    "data": {"message": "No content returned"}
                }
                
        except Exception as e:
            logger.error(f"Error parsing MCP response: {e}")
            return self._error_response(tool_name, f"Parse error: {str(e)}")
    
    async def list_tools_async(self) -> List[Dict[str, Any]]:
        """
        List all available tools from MCP server.
        
        Returns:
            List of tool definitions
        """
        try:
            async with sse_client(url=self.base_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    result = await session.list_tools()
                    tools = []
                    
                    for tool in result.tools:
                        tools.append({
                            "name": tool.name,
                            "description": tool.description if hasattr(tool, 'description') else "",
                            "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        })
                    
                    return tools
                    
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for listing tools.
        
        Returns:
            List of tool definitions
        """
        if self._tools_cache is None:
            self._tools_cache = asyncio.run(self.list_tools_async())
        return self._tools_cache
    
    def _error_response(self, tool_name: str, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "tool_name": tool_name,
            "status": "error",
            "error": error_msg
        }
    
    def close(self):
        """Close the client (cleanup if needed)"""
        logger.info("MCP Client closed")


class MCPClientError(Exception):
    """Custom exception for MCP client errors"""
    pass


# ============================================================================
# MOCK CLIENT FOR TESTING
# ============================================================================

class MockMCPClient(MCPClient):
    """
    Mock MCP client for testing without actual API calls.
    
    Returns predefined responses based on tool name.
    """
    
    def __init__(self, *args, **kwargs):
        # Don't call super().__init__() to avoid SSE connection requirements
        self.base_url = kwargs.get('base_url', '')
        self.timeout = kwargs.get('timeout', 30)
        self.max_retries = kwargs.get('max_retries', 3)
        
        self.mock_responses = {
            "get_wishes_data": {
                "birthdays": [
                    {
                        "name": "Amit Kumar",
                        "date": "2025-12-10",
                        "department": "Engineering",
                        "email": "amit.kumar@zinghr.com"
                    },
                    {
                        "name": "Priya Sharma",
                        "date": "2025-12-10",
                        "department": "HR",
                        "email": "priya.sharma@zinghr.com"
                    }
                ],
                "anniversaries": [
                    {
                        "name": "Rahul Verma",
                        "years": 5,
                        "date": "2025-12-10",
                        "joined_date": "2020-12-10"
                    }
                ],
                "new_joinees": []
            },
            "get_survey_data": {
                "pending_surveys": 2,
                "completed_surveys": 5,
                "total_surveys": 7,
                "surveys": [
                    {
                        "id": 1,
                        "name": "Q4 Feedback Survey",
                        "status": "pending",
                        "deadline": "2025-12-15"
                    },
                    {
                        "id": 2,
                        "name": "Team Culture Assessment",
                        "status": "pending",
                        "deadline": "2025-12-20"
                    }
                ]
            },
            "get_learning_courses": {
                "available_courses": 15,
                "enrolled_courses": 3,
                "courses": [
                    {
                        "id": 1,
                        "name": "Python Advanced Programming",
                        "duration": "4 weeks",
                        "instructor": "Dr. Smith",
                        "enrollment_status": "available"
                    },
                    {
                        "id": 2,
                        "name": "Leadership and Management Skills",
                        "duration": "2 weeks",
                        "instructor": "Jane Doe",
                        "enrollment_status": "enrolled"
                    }
                ]
            },
            "get_badges_with_reaction": {
                "total_badges": 25,
                "badges": [
                    {
                        "recipient": "Engineering Team",
                        "badge": "Excellence Award Q4",
                        "date": "2025-12-01",
                        "given_by": "CEO"
                    },
                    {
                        "recipient": "John Doe",
                        "badge": "Innovation Star",
                        "date": "2025-11-28",
                        "given_by": "CTO"
                    }
                ]
            },
            "get_my_badges_with_reaction": {
                "my_badges": [
                    {
                        "badge": "Top Performer Q3",
                        "date": "2025-11-15",
                        "given_by": "Manager",
                        "reason": "Exceeded targets"
                    },
                    {
                        "badge": "Team Player",
                        "date": "2025-10-20",
                        "given_by": "Team Lead",
                        "reason": "Excellent collaboration"
                    }
                ]
            },
            "get_preference_data": {
                "supported_languages": [
                    "English",
                    "Hindi",
                    "Tamil",
                    "Telugu",
                    "Bengali",
                    "Marathi",
                    "Gujarati",
                    "Kannada"
                ],
                "current_language": "English"
            }
        }
    
    def execute_tool(self, request: MCPToolRequest) -> Dict[str, Any]:
        """Return mock response based on tool name"""
        logger.info(f"[MOCK] Executing tool: {request.tool_name}")
        
        # Simulate authentication check
        if not request.jwt_token:
            return {
                "tool_name": request.tool_name,
                "status": "error",
                "error": "Authentication required"
            }
        
        # Return mock data
        mock_data = self.mock_responses.get(
            request.tool_name,
            {"message": f"Mock data for {request.tool_name}"}
        )
        
        return {
            "tool_name": request.tool_name,
            "status": "success",
            "data": mock_data
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Return mock tools list"""
        return [
            {
                "name": tool_name,
                "description": f"Mock tool: {tool_name}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "jwt_token": {"type": "string"},
                        "from_date": {"type": "string"},
                        "to_date": {"type": "string"}
                    }
                }
            }
            for tool_name in self.mock_responses.keys()
        ]


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_mcp_client(
    base_url: str,
    mock_mode: bool = False,
    **kwargs
) -> MCPClient:
    """
    Factory function to create MCP client.
    
    Args:
        base_url: MCP server base URL
        mock_mode: If True, return MockMCPClient
        **kwargs: Additional arguments for client initialization
        
    Returns:
        MCPClient or MockMCPClient instance
    """
    if mock_mode:
        logger.info("Creating MockMCPClient (no real API calls)")
        return MockMCPClient(base_url=base_url, **kwargs)
    else:
        logger.info("Creating real MCPClient with SSE connection")
        return MCPClient(base_url, **kwargs)