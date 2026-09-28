"""
AFRA Configuration Module.

Centralized configuration management using Pydantic BaseSettings.
Loads settings from environment variables and .env files.
"""

from config.settings import Settings

__all__ = ["Settings"]
