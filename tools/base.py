"""
Base Tool — Abstract base class for all AFRA agent tools.

Every tool in the system inherits from BaseTool and implements the `_execute`
method. The base class provides common functionality: input validation against
JSON Schema, structured result wrapping, and logging.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Result Containers ─────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Standard result wrapper returned by every tool execution.

    Attributes:
        success: Whether the tool executed without errors.
        data: The tool's output data (structure varies per tool).
        error: Error message if the tool failed, empty string otherwise.
        source: Identifier for the data source (e.g. "sec_edgar", "fmp_api").
        metadata: Additional metadata about the result (timestamps, confidence, etc.).
        execution_time_ms: Wall-clock execution time in milliseconds.
    """

    success: bool
    data: Any = None
    error: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


@dataclass
class ToolSchema:
    """OpenAI function-calling compatible schema definition.

    Attributes:
        name: Tool name (no spaces, snake_case).
        description: 1-3 sentence description of when/why to use this tool.
        parameters: JSON Schema object defining input types, required fields, defaults.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool-use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


# ── Base Tool ─────────────────────────────────────────────────────────────────


class BaseTool(ABC):
    """Abstract base class for all AFRA agent tools.

    Subclasses must define:
        - name: Unique tool identifier (snake_case).
        - description: When and why the LLM should use this tool.
        - parameters_schema: JSON Schema dict for input validation.
        - _execute(**kwargs): The actual tool logic.

    Optional overrides:
        - fallback_tools: List of tool names to try if this tool fails.
    """

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {}
    fallback_tools: list[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that subclasses define required attributes."""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", ""):
            # Allow abstract intermediate classes
            if not getattr(cls, "__abstractmethods__", set()):
                raise TypeError(f"{cls.__name__} must define a 'name' attribute")

    # ── Public API ────────────────────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with input validation and timing.

        This is the public entry point. It validates inputs, runs the tool,
        and wraps the result in a ToolResult with execution timing.

        Args:
            **kwargs: Tool-specific keyword arguments matching parameters_schema.

        Returns:
            ToolResult with success=True and data, or success=False and error.
        """
        start = time.perf_counter()
        try:
            self.validate_inputs(**kwargs)
            result = await self._execute(**kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            if isinstance(result, ToolResult):
                result.execution_time_ms = elapsed
                return result

            # If _execute returns raw data, wrap it
            return ToolResult(
                success=True,
                data=result,
                source=self.name,
                execution_time_ms=elapsed,
            )

        except ValidationError as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Tool %s validation error: %s", self.name, e)
            return ToolResult(
                success=False,
                error=f"Input validation error: {e}",
                source=self.name,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Tool %s execution error: %s", self.name, e, exc_info=True)
            return ToolResult(
                success=False,
                error=f"Execution error: {type(e).__name__}: {e}",
                source=self.name,
                execution_time_ms=elapsed,
            )

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ToolResult | Any:
        """Implement the actual tool logic.

        Subclasses must override this method. It can return either a ToolResult
        directly or raw data (which will be auto-wrapped).

        Args:
            **kwargs: Validated tool-specific keyword arguments.

        Returns:
            ToolResult or raw data to be wrapped.
        """
        ...

    # ── Schema ────────────────────────────────────────────────────────────

    def get_schema(self) -> ToolSchema:
        """Return the tool's schema for LLM registration."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )

    # ── Validation ────────────────────────────────────────────────────────

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate input arguments against the tool's parameters_schema.

        Checks:
        1. All required parameters are present.
        2. Parameter types match the schema (basic type checking).
        3. Enum constraints are satisfied.

        Raises:
            ValidationError: If any validation check fails.
        """
        schema = self.parameters_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required parameters
        for param_name in required:
            if param_name not in kwargs:
                raise ValidationError(
                    f"Missing required parameter: '{param_name}'"
                )

        # Validate types and constraints for provided parameters
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for param_name, value in kwargs.items():
            if param_name not in properties:
                continue  # Allow extra params (flexible)

            prop = properties[param_name]
            expected_type = prop.get("type")

            if expected_type and expected_type in type_map:
                python_type = type_map[expected_type]
                if not isinstance(value, python_type):
                    raise ValidationError(
                        f"Parameter '{param_name}' expects {expected_type}, "
                        f"got {type(value).__name__}"
                    )

            # Check enum constraints
            if "enum" in prop and value not in prop["enum"]:
                raise ValidationError(
                    f"Parameter '{param_name}' must be one of {prop['enum']}, "
                    f"got '{value}'"
                )

    # ── Representation ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def load_schema_from_file(cls, schema_path: str | Path) -> dict[str, Any]:
        """Load a JSON schema from a file path.

        Args:
            schema_path: Path to the JSON schema file.

        Returns:
            Parsed JSON schema dictionary.
        """
        path = Path(schema_path)
        with path.open() as f:
            return json.load(f)


class ValidationError(Exception):
    """Raised when tool input validation fails."""
