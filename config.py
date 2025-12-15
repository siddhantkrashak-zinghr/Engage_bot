"""
Configuration management for Engage Bot.

File: engage_bot/config.py
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
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "zinghr-gpt-oss-20b"))
    temperature: float = 0.0
    max_tokens: int = 8000
    
    # For custom model endpoints (e.g., zinghr-gpt-oss-20b-new)
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
    sse_endpoint: str = "/sse"
    execute_endpoint: str = "/mcp_execute"
    timeout: int = 30
    max_retries: int = 3


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
    
    @classmethod
    def from_env(cls) -> "EngageBotConfig":
        """Create configuration from environment variables"""
        return cls()
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.llm.api_key:
            raise ValueError("LLM API key is required")
        
        if not self.mcp.base_url:
            raise ValueError("MCP base URL is required")
        
        return True


# Default configuration instance
config = EngageBotConfig.from_env()