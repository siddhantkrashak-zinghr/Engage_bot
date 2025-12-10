"""
Configuration management for Engage Bot.

File: config.py
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """LLM model configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    max_tokens: int = 8000
    
    # For custom model endpoints
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
    )


@dataclass
class MCPConfig:
    """MCP server configuration"""
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "MCP_BASE_URL",
            "https://mservices-dev.zinghr.com/zingmcpserver"
        )
    )
    tools_list_endpoint: str = "/tools/list"  # Endpoint to fetch available tools
    execute_endpoint: str = "/mcp_execute"
    timeout: int = 30
    max_retries: int = 3
    cache_tools: bool = True  # Cache tools after first fetch


@dataclass
class TimeConfig:
    """Time and date configuration"""
    timezone: str = "Asia/Kolkata"
    time_format: str = "%Y-%m-%d %H:%M:%S"
    date_format: str = "%Y-%m-%d"


@dataclass
class EngageBotConfig:
    """Main configuration for Engage Bot"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Feature flags
    enable_mock_mode: bool = field(
        default_factory=lambda: os.getenv("MOCK_MODE", "false").lower() == "true"
    )
    
    # Dynamic tool discovery
    enable_dynamic_tools: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DYNAMIC_TOOLS", "true").lower() == "true"
    )
    
    @classmethod
    def from_env(cls) -> "EngageBotConfig":
        """Create configuration from environment variables"""
        return cls()
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.llm.api_key:
            raise ValueError("LLM API key is required. Set OPENAI_API_KEY in .env file")
        
        if not self.mcp.base_url:
            raise ValueError("MCP base URL is required. Set MCP_BASE_URL in .env file")
        
        return True


# Default configuration instance
config = EngageBotConfig.from_env()