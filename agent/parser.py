"""
Response Parser — Parses LLM outputs into structured tool calls and reasoning traces.

Handles both OpenAI and Anthropic response formats, extracting tool invocations,
thought traces, and final answers from the LLM's output.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedToolCall:
    """A tool call extracted from an LLM response.

    Attributes:
        tool_name: Name of the tool to invoke.
        arguments: Dict of arguments to pass to the tool.
        call_id: Unique identifier for the tool call (from the LLM).
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ParsedThought:
    """A reasoning trace extracted from an LLM response.

    Attributes:
        content: The reasoning text.
        step_number: Which step in the ReAct loop this thought belongs to.
    """

    content: str
    step_number: int = 0


@dataclass
class ParsedResponse:
    """Complete parsed LLM response.

    Attributes:
        thoughts: List of reasoning traces.
        tool_calls: List of tool calls to execute.
        final_answer: The final response text (if no more tool calls needed).
        raw_content: The raw LLM response text.
        is_final: Whether this is the agent's final answer (no more tool calls).
    """

    thoughts: list[ParsedThought] = field(default_factory=list)
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    final_answer: str = ""
    raw_content: str = ""
    is_final: bool = False


class ResponseParser:
    """Parses LLM responses from both OpenAI and Anthropic APIs.

    Supports:
    - OpenAI function calling responses (tool_calls in message)
    - Anthropic tool_use content blocks
    - Plain text responses with embedded tool call notation
    """

    def parse_openai_response(self, response: Any) -> ParsedResponse:
        """Parse an OpenAI ChatCompletion response.

        Args:
            response: The OpenAI API response object.

        Returns:
            ParsedResponse with extracted tool calls and/or final answer.
        """
        parsed = ParsedResponse()

        # Handle both raw API response and LangChain-wrapped response
        message = None
        if hasattr(response, "choices"):
            message = response.choices[0].message
        elif hasattr(response, "content"):
            # LangChain AIMessage
            message = response
        else:
            parsed.final_answer = str(response)
            parsed.is_final = True
            return parsed

        # Extract text content
        content = ""
        if hasattr(message, "content") and message.content:
            content = message.content
            parsed.raw_content = content

        # Extract tool calls
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    # OpenAI native format
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}

                    parsed.tool_calls.append(
                        ParsedToolCall(
                            tool_name=tc.function.name,
                            arguments=args,
                            call_id=getattr(tc, "id", ""),
                        )
                    )
                elif isinstance(tc, dict):
                    # Dict format (e.g., from LangChain)
                    name = tc.get("name", tc.get("function", {}).get("name", ""))
                    args = tc.get("args", tc.get("function", {}).get("arguments", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}

                    parsed.tool_calls.append(
                        ParsedToolCall(
                            tool_name=name,
                            arguments=args,
                            call_id=tc.get("id", ""),
                        )
                    )

        # Extract thoughts from content
        if content:
            thoughts = self._extract_thoughts(content)
            parsed.thoughts = thoughts

        # Determine if this is a final answer
        parsed.is_final = len(parsed.tool_calls) == 0
        if parsed.is_final:
            parsed.final_answer = content

        return parsed

    def parse_anthropic_response(self, response: Any) -> ParsedResponse:
        """Parse an Anthropic API response.

        Args:
            response: The Anthropic API response object.

        Returns:
            ParsedResponse with extracted tool calls and/or final answer.
        """
        parsed = ParsedResponse()

        content_blocks = []
        if hasattr(response, "content"):
            content_blocks = response.content
        elif isinstance(response, dict):
            content_blocks = response.get("content", [])

        text_parts = []
        for block in content_blocks:
            if hasattr(block, "type"):
                block_type = block.type
            elif isinstance(block, dict):
                block_type = block.get("type", "")
            else:
                continue

            if block_type == "text":
                text = block.text if hasattr(block, "text") else block.get("text", "")
                text_parts.append(text)
            elif block_type == "tool_use":
                name = block.name if hasattr(block, "name") else block.get("name", "")
                args = block.input if hasattr(block, "input") else block.get("input", {})
                call_id = block.id if hasattr(block, "id") else block.get("id", "")
                parsed.tool_calls.append(
                    ParsedToolCall(tool_name=name, arguments=args, call_id=call_id)
                )

        content = "\n".join(text_parts)
        parsed.raw_content = content

        if content:
            parsed.thoughts = self._extract_thoughts(content)

        parsed.is_final = len(parsed.tool_calls) == 0
        if parsed.is_final:
            parsed.final_answer = content

        return parsed

    def parse_langchain_response(self, response: Any) -> ParsedResponse:
        """Parse a LangChain AIMessage response.

        Works with both OpenAI and Anthropic backends via LangChain.
        """
        parsed = ParsedResponse()

        # LangChain AIMessage
        content = ""
        if hasattr(response, "content"):
            if isinstance(response.content, str):
                content = response.content
            elif isinstance(response.content, list):
                # Anthropic-style content blocks
                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")
                    elif isinstance(block, str):
                        content += block

        parsed.raw_content = content

        # Tool calls (LangChain unified format)
        tool_calls = getattr(response, "tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    parsed.tool_calls.append(
                        ParsedToolCall(
                            tool_name=tc.get("name", ""),
                            arguments=tc.get("args", {}),
                            call_id=tc.get("id", ""),
                        )
                    )

        # Additional tool calls from additional_kwargs
        additional = getattr(response, "additional_kwargs", {})
        if "tool_calls" in additional:
            for tc in additional["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                parsed.tool_calls.append(
                    ParsedToolCall(
                        tool_name=func.get("name", ""),
                        arguments=args,
                        call_id=tc.get("id", ""),
                    )
                )

        if content:
            parsed.thoughts = self._extract_thoughts(content)

        parsed.is_final = len(parsed.tool_calls) == 0
        if parsed.is_final:
            parsed.final_answer = content

        return parsed

    def _extract_thoughts(self, content: str) -> list[ParsedThought]:
        """Extract reasoning traces from response content.

        Looks for patterns like:
        - "Thought: ..." or "Thinking: ..."
        - "I need to..." or "I should..."
        - "Let me..." or "First, I'll..."
        """
        thoughts = []
        step = 1

        # Look for explicit thought markers
        thought_patterns = [
            r"(?:Thought|Thinking|Reasoning)(?:\s*\d*)?:\s*(.+?)(?=\n(?:Action|Tool|Thought|\Z))",
            r"(?:Step\s*\d+):\s*(.+?)(?=\n(?:Step|\Z))",
        ]

        for pattern in thought_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                thoughts.append(ParsedThought(content=match.strip(), step_number=step))
                step += 1

        # If no explicit markers, treat the entire content as a thought
        if not thoughts and content.strip():
            thoughts.append(ParsedThought(content=content.strip(), step_number=1))

        return thoughts
