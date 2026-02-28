"""
title: Anthropic API Integration
author: Podden (https://github.com/Podden/)
original_author: Balaxxe (Updated by nbellochi)
version: 0.2
license: MIT
requirements: pydantic>=2.0.0, aiohttp>=3.8.0
environment_variables:
    - ANTHROPIC_API_KEY (required)

Supports:
- Uses Anthropic Python SDK
- Fetch Claude Models from API Endpoint
- Tool Call Loop (call multiple Tools in the same response)
- web_search Tool
- citations for web_search
- Streaming responses
- Prompt caching (server-side)
- Promt Caching of System Promts, Messages- and Tools Array
- Comprehensive error handling

Todo:
- Image and PDF processing
- Web_Search Toggle Action
- Fine Grained Tool Streaming
- Extended Thinking Toggle Action
- Bash Tool
- Text Editor Tool
- Files API
- PDF support
- Vision
- PDF support
- Non-Streaming
- Usage for OpenWebUI
- API Key from UserValves as Alternative
- MCP Connector (mcpo is enought for me atm)
"""

from collections.abc import Awaitable
import json
import logging
import asyncio
import random
from typing import Any, Callable, List, Union, Dict, Optional
from pydantic import BaseModel, Field
import aiohttp
from anthropic import APIStatusError, AsyncAnthropic, RateLimitError, APIConnectionError, AuthenticationError, BadRequestError, InternalServerError, PermissionDeniedError, NotFoundError, UnprocessableEntityError
import json
import inspect


class Pipe:
    API_VERSION = "2023-06-01"  # Current API version as of May 2025
    MODEL_URL = "https://api.anthropic.com/v1/messages"
    # Model max tokens - comprehensive list of all Claude models
    MODEL_MAX_TOKENS = {
        # Claude 3 family
        "claude-3-opus-20240229": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-haiku-20240307": 4096,
        # Claude 3.5 family
        "claude-3-5-sonnet-20240620": 8192,
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-5-haiku-20241022": 8192,
        # Claude 3.7 family
        "claude-3-7-sonnet-20250219": 16384,  # 16K by default, 128K with beta
        # Claude 4 family - NEW MODELS
        "claude-sonnet-4-20250514": 32000,  # 32K by default, 128K with beta
        "claude-opus-4-20250514": 32000,  # 32K by default, 128K with beta
        # Latest aliases
        "claude-3-opus-latest": 4096,
        "claude-3-sonnet-latest": 4096,
        "claude-3-haiku-latest": 4096,
        "claude-3-5-sonnet-latest": 8192,
        "claude-3-5-haiku-latest": 8192,
        "claude-3-7-sonnet-latest": 16384,  # 16K by default, 128K with beta
        "claude-sonnet-4-latest": 32000,  # 32K by default, 128K with beta
        "claude-opus-4-latest": 32000,  # 32K by default, 128K with beta
    }
    # Model context lengths - maximum input tokens
    MODEL_CONTEXT_LENGTH = {
        # Claude 3 family
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        # Claude 3.5 family
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        # Claude 3.7 family
        "claude-3-7-sonnet-20250219": 200000,
        # Claude 4 family - NEW MODELS
        "claude-sonnet-4-20250514": 1000000,
        "claude-opus-4-20250514": 200000,
        # Latest aliases
        "claude-3-opus-latest": 200000,
        "claude-3-sonnet-latest": 200000,
        "claude-3-haiku-latest": 200000,
        "claude-3-5-sonnet-latest": 200000,
        "claude-3-5-haiku-latest": 200000,
        "claude-3-7-sonnet-latest": 200000,
        "claude-sonnet-4-latest": 1000000,
        "claude-opus-4-latest": 200000,
    }
    # Models that support extended thinking
    THINKING_SUPPORTED_MODELS = [
        "claude-3-7-sonnet-latest",
        "claude-3-7-sonnet-20250219",
        # Claude 4 models with enhanced thinking capabilities
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-sonnet-4-latest",
        "claude-opus-4-latest",
    ]
    # Models that support 1M token context window (Claude Sonnet 4 only)
    MODELS_SUPPORTING_1M_CONTEXT = [
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-latest",
    ]
    REQUEST_TIMEOUT = (
        300  # Increased timeout for longer responses with extended thinking
    )
    THINKING_BUDGET_TOKENS = 16000  # Default thinking budget tokens (max 16K)
    CLAUDE_4_THINKING_BUDGET = 32000  # Enhanced thinking budget for Claude 4 models

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = "Your API Key Here"
        ENABLE_THINKING: bool = (
            True  # Changed to True to enable streaming of thinking tokens
        )
        MAX_OUTPUT_TOKENS: bool = True  # Valve to use maximum possible output tokens
        THINKING_BUDGET_TOKENS: int = Field(
            default=16000, ge=0, le=16000
        )  # Configurable thinking budget tokens 16,000 max
        ENABLE_1M_CONTEXT: bool = Field(
            default=False,
            description="Enable 1M token context window for Claude Sonnet 4 (requires Tier 4 API access)",
        )
        WEB_SEARCH: bool = Field(
            default=True,
            description="Enable web search tool for Claude models",
        )
        WEB_SEARCH_MAX_USES: int = Field(
            default=5, ge=1, le=20,
            description="Maximum number of web searches allowed per conversation",
        )
        WEB_SEARCH_USER_CITY: str = Field(
            default="Leipzig",
            description="User's city for web search location context",
        )
        WEB_SEARCH_USER_REGION: str = Field(
            default="Saxony",
            description="User's region/state for web search location context",
        )
        WEB_SEARCH_USER_COUNTRY: str = Field(
            default="DE",
            description="User's country code for web search location context",
        )
        WEB_SEARCH_USER_TIMEZONE: str = Field(
            default="Europe/Berlin",
            description="User's timezone for web search location context",
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging to see requests and responses",
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.request_id = None

    async def get_anthropic_models(self) -> List[dict]:
        """
        Fetches the current list of Anthropic models using the official Anthropic Python SDK.
        Fallback to static list on error. Returns OpenWebUI model dicts.
        """
        from anthropic import AsyncAnthropic

        models = []
        try:
            api_key = self.valves.ANTHROPIC_API_KEY
            client = AsyncAnthropic(api_key=api_key)
            async for m in client.models.list():
                name = m.id
                display_name = getattr(m, "display_name", name)
                context_length = self.MODEL_CONTEXT_LENGTH.get(name, 200000)
                max_output_tokens = self.MODEL_MAX_TOKENS.get(name, 4096)
                supports_thinking = name in self.THINKING_SUPPORTED_MODELS
                is_hybrid = name in self.THINKING_SUPPORTED_MODELS
                supports_vision = True
                models.append(
                    {
                        "id": f"anthropic/{name}",
                        "name": display_name,
                        "context_length": context_length,
                        "supports_vision": supports_vision,
                        "supports_thinking": supports_thinking,
                        "is_hybrid_model": is_hybrid,
                        "max_output_tokens": max_output_tokens,
                    }
                )
            return models
        except Exception as e:
            logging.warning(
                f"Could not fetch models from SDK/API, using static list. Reason: {e}"
            )
        # Fallback: use static list (old logic)
        # ...existing code for static models...
        standard_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
        hybrid_models = [
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-latest",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
        ]
        for name in standard_models:
            models.append(
                {
                    "id": f"anthropic/{name}",
                    "name": name,
                    "context_length": self.MODEL_CONTEXT_LENGTH.get(name, 200000),
                    "supports_vision": name != "claude-3-5-haiku-20241022",
                    "supports_thinking": False,
                    "is_hybrid_model": False,
                    "max_output_tokens": self.MODEL_MAX_TOKENS.get(name, 4096),
                }
            )
        for name in hybrid_models:
            models.append(
                {
                    "id": f"anthropic/{name}",
                    "name": f"{name} (Standard)",
                    "context_length": self.MODEL_CONTEXT_LENGTH.get(name, 200000),
                    "supports_vision": True,
                    "supports_thinking": False,
                    "is_hybrid_model": True,
                    "thinking_mode": "standard",
                    "max_output_tokens": self.MODEL_MAX_TOKENS.get(name, 16384),
                }
            )
            models.append(
                {
                    "id": f"anthropic/{name}-thinking",
                    "name": f"{name} (Extended Thinking)",
                    "context_length": self.MODEL_CONTEXT_LENGTH.get(name, 200000),
                    "supports_vision": True,
                    "supports_thinking": True,
                    "is_hybrid_model": True,
                    "thinking_mode": "extended",
                    "max_output_tokens": (
                        131072
                        if self.valves.MAX_OUTPUT_TOKENS
                        else self.MODEL_MAX_TOKENS.get(name, 16384)
                    ),
                }
            )
        return models

    async def pipes(self) -> List[dict]:
        return await self.get_anthropic_models()

    def _create_payload(
        self,
        body: Dict,
        __user__: Optional[dict],
        __tools__: Optional[Dict[str, Dict[str, Any]]],
    ) -> tuple[dict, dict]:
        """
        Create the payload and headers for Claude API request.

        Args:
            body: The request body from OpenWebUI
            messages: Processed messages
            system_message: Optional system message

        Returns:
            tuple: (payload, headers)
        """
        # --- 1. Model selection & output token calculation ---
        model_name = body["model"].split("/")[-1]
        is_thinking_variant = model_name.endswith("-thinking")
        actual_model_name = (
            model_name.replace("-thinking", "") if is_thinking_variant else model_name
        )

        if actual_model_name not in self.MODEL_MAX_TOKENS and self.valves.DEBUG:
            logging.warning(
                f"Unknown model: {actual_model_name}, using default token limit"
            )

        max_tokens_limit = self.MODEL_MAX_TOKENS.get(actual_model_name, 4096)
        max_tokens = (
            max_tokens_limit
            if self.valves.MAX_OUTPUT_TOKENS
            else min(body.get("max_tokens", max_tokens_limit), max_tokens_limit)
        )
        print(body)
        payload: dict[str, Any] = {
            "model": actual_model_name,
            "max_tokens": max_tokens,
            "stream": body.get("stream", True),
            "metadata": body.get("metadata", {}),
        }
        if body.get("temperature") is not None:
            payload["temperature"] = float(body.get("temperature"))
        if body.get("top_k") is not None:
            payload["top_k"] = float(body.get("top_k"))
        if body.get("top_p") is not None:
            payload["top_p"] = float(body.get("top_p"))

        # TODO: Implement Thinking with a Toggle Actiom over __metadata__
        # if actual_model_name in self.THINKING_SUPPORTED_MODELS:
        #     # Hybrid models: enable thinking only when -thinking variant chosen
        #     should_enable_thinking = is_thinking_variant
        # else:
        #     should_enable_thinking = self.valves.ENABLE_THINKING

        # if should_enable_thinking and actual_model_name in self.THINKING_SUPPORTED_MODELS:
        #     thinking_budget = self.valves.THINKING_BUDGET_TOKENS
        #     if any(x in actual_model_name for x in ["claude-sonnet-4", "claude-opus-4"]):
        #         thinking_budget = min(self.CLAUDE_4_THINKING_BUDGET, self.valves.THINKING_BUDGET_TOKENS * 2)
        #     payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        if "response_format" in body:
            payload["response_format"] = {"type": body["response_format"].get("type")}

        raw_messages = body.get("messages", []) or []
        system_messages = []
        processed_messages: list[dict] = []
        for msg in raw_messages:
            role = msg.get("role")
            processed_content = self._process_content(msg.get("content"))
            if not processed_content:
                continue
            if role == "system":
                for sc in processed_content:
                    system_messages.append(sc)
            else:
                processed_messages.append({"role": role, "content": processed_content})

        # Correct Order for Caching: Tools, System, Messages
        tools_list = self._convert_tools_to_claude_format(__tools__)
        if tools_list and len(tools_list) > 0:
            payload["tools"] = tools_list
            payload["tool_choice"] = {"type": "auto"}

        if system_messages and len(system_messages) > 0:
            system_messages[-1]["cache_control"] = {"type": "ephemeral"}
            payload["system"] = system_messages

        if processed_messages and len(processed_messages) > 0:
            last_msg = processed_messages[-1]
            content_blocks = last_msg.get("content", [])
            if content_blocks:
                last_content_block = content_blocks[-1]
                last_content_block.setdefault("cache_control", {"type": "ephemeral"})
            payload["messages"] = processed_messages

        # Remove any None values
        # payload = {k: v for k, v in payload.items() if v is not None}

        # --- 10. Headers & beta flags ---
        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        beta_headers: list[str] = []

        # Add 1M context header if enabled and model supports it
        if (
            self.valves.ENABLE_1M_CONTEXT
            and actual_model_name in self.MODELS_SUPPORTING_1M_CONTEXT
        ):
            beta_headers.append("context-1m-2025-08-07")

        if beta_headers and len(beta_headers) > 0:
            headers["anthropic-beta"] = ",".join(beta_headers)

        if self.valves.DEBUG:
            print(f"[DEBUG] Payload: {json.dumps(payload, indent=2)}")
            print(f"[DEBUG] Headers: {headers}")
        return payload, headers

    def _convert_tools_to_claude_format(self, __tools__):
        """
        Convert OpenWebUI tools format to Claude API format.
        Args:
            __tools__: Dict of tools from OpenWebUI
        Returns:
            list: Tools in Claude API format
        """
        claude_tools = []

        if self.valves.DEBUG:
            print(f"[DEBUG] Converting tools: {json.dumps(__tools__, indent=2)}")

        # Add web search tool if enabled
        if self.valves.WEB_SEARCH:
            claude_tools.append(
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": self.valves.WEB_SEARCH_MAX_USES,
                    "user_location": {
                        "type": "approximate",
                        "city": self.valves.WEB_SEARCH_USER_CITY,
                        "region": self.valves.WEB_SEARCH_USER_REGION,
                        "country": self.valves.WEB_SEARCH_USER_COUNTRY,
                        "timezone": self.valves.WEB_SEARCH_USER_TIMEZONE,
                    },
                }
            )

        if not __tools__ or len(__tools__) == 0:
            if self.valves.DEBUG:
                print(f"[DEBUG] No tools provided, using default Claude tools")
            if claude_tools:
                claude_tools[-1]["cache_control"] = {"type": "ephemeral"}
            return claude_tools

        for tool_name, tool_data in __tools__.items():
            if not isinstance(tool_data, dict) or "spec" not in tool_data:
                if self.valves.DEBUG:
                    print(f"[DEBUG] Skipping invalid tool: {tool_name} - missing spec")
                continue

            spec = tool_data["spec"]

            # Extract basic tool info
            name = spec.get("name", tool_name)
            description = spec.get("description", f"Tool: {name}")
            parameters = spec.get("parameters", {})

            # Convert OpenWebUI parameters to Claude input_schema format
            # OpenWebUI parameters are typically already in JSON Schema format
            input_schema = {
                "type": "object",
                "properties": parameters.get("properties", {}),
            }

            # Add required fields if they exist
            if "required" in parameters:
                input_schema["required"] = parameters["required"]

            # Create Claude tool format
            claude_tool = {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            }
            claude_tools.append(claude_tool)

            if self.valves.DEBUG:
                print(f"[DEBUG] Converted tool '{name}' to Claude format")

        if self.valves.DEBUG:
            print(f"[DEBUG] Total tools converted: {len(claude_tools)}")

        if claude_tools:
            claude_tools[-1]["cache_control"] = {"type": "ephemeral"}
        return claude_tools

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __metadata__: dict[str, Any] = {},
        __tools__: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        OpenWebUI Claude streaming pipe with integrated streaming logic.
        """
        if not self.valves.ANTHROPIC_API_KEY:
            error_msg = "Error: ANTHROPIC_API_KEY is required"
            if self.valves.DEBUG:
                print(f"[DEBUG] {error_msg}")
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "No API Key Set!",
                            "done": True,
                            "hidden": True,
                        },
                    }
                )
            return

        try:
            if inspect.isawaitable(__tools__):
                __tools__ = await __tools__

            try:
                payload, headers = self._create_payload(body, __user__, __tools__)
            except Exception as e:
                # Handle payload creation errors
                await self.handle_errors(e, __event_emitter__)
                return

            if payload.get("stream"):
                try:
                    api_key = self.valves.ANTHROPIC_API_KEY
                    client = AsyncAnthropic(api_key=api_key, default_headers=headers)
                    payload_for_stream = {k: v for k, v in payload.items() if k != "stream"}
                except Exception as e:
                    # Handle client creation errors
                    await self.handle_errors(e, __event_emitter__)
                    return

                # Stream loop variables
                token_buffer_size = getattr(self.valves, "TOKEN_BUFFER_SIZE", 1)
                is_model_thinking = False
                conversation_ended = False
                max_function_calls = 5
                current_function_calls = 0
                has_pending_tool_calls = False
                tools_buffer = ""
                tool_calls = []
                chunk = ""
                chunk_count = 0

                try:
                    while (
                        current_function_calls < max_function_calls
                        and not conversation_ended
                    ):
                        async with client.messages.stream(
                            **payload_for_stream
                        ) as stream:
                            async for event in stream:
                                event_type = getattr(event, "type", None)
                                if self.valves.DEBUG:
                                    # Only log event_type and minimal event info, skip snapshot fields
                                    if hasattr(event, "__dict__"):
                                        event_dict = {
                                            k: v
                                            for k, v in event.__dict__.items()
                                            if k != "snapshot"
                                        }
                                        print(
                                            f"[Anthropic] Received event: {event_type} with {str(event_dict)[:100]}{'...' if len(str(event_dict)) > 100 else ''}"
                                        )
                                    else:
                                        print(
                                            f"[Anthropic] Received event: {event_type} with {str(event)[:100]}{'...' if len(str(event)) > 100 else ''}"
                                        )
                                if event_type == "message_start":
                                    pass

                                elif event_type == "content_block_start":
                                    if current_function_calls > 0:
                                        await __event_emitter__(
                                            {"type": "status", "data": {"hidden": True}}
                                        )
                                    content_block = getattr(
                                        event, "content_block", None
                                    )
                                    content_type = getattr(content_block, "type", None)
                                    if not content_block:
                                        continue
                                    if content_type == "text":
                                        chunk += content_block.text or ""
                                    if content_type == "thinking":
                                        is_model_thinking = True
                                        chunk += "<thinking>"
                                    if content_type == "tool_use":
                                        tools_buffer = (
                                            "{"
                                            f'"type": "{content_block.type}", '
                                            f'"id": "{content_block.id}", '
                                            f'"name": "{content_block.name}", '
                                            f'"input": '
                                        )

                                    if content_type == "server_tool_use":
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": "Searching the Web...",
                                                    "done": False,
                                                },
                                            }
                                        )
                                    if content_type == "web_search_tool_result":
                                        if self.valves.DEBUG:
                                            print(
                                                f"[DEBUG] Processing web search result event: {event}"
                                            )
                                        content_items = getattr(content_block, "content", [])
                                        if content_items and len(content_items) > 0:
                                            error_code = getattr(content_block, "error_code", None)
                                            if error_code:
                                                await self.handle_errors(Exception(f"Web search error: {error_code}"),__event_emitter__)
                                            else:
                                                await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": "Web Search Complete",
                                                        "done": True,
                                                    },
                                                }
                                            )
                                        
                                elif event_type == "content_block_delta":
                                    delta = getattr(event, "delta", None)
                                    if delta:
                                        delta_type = getattr(delta, "type", None)
                                        if delta_type == "thinking_delta":
                                            chunk += getattr(delta, "thinking", "")
                                        elif delta_type == "text_delta":
                                            text_delta = getattr(delta, "text", "")
                                            chunk += text_delta
                                            chunk_count += 1
                                        elif delta_type == "input_json_delta":
                                            tools_buffer += getattr(
                                                delta, "partial_json", ""
                                            )
                                        elif delta_type == "citations_delta":
                                            # Handle citations within content_block_delta
                                            await self.handle_citation(
                                                event, __event_emitter__
                                            )

                                elif event_type == "content_block_stop":
                                    event_name = getattr(event, "name", "")
                                    if event_name == "web_search":
                                        if tools_buffer.endswith("}"):
                                            message = ""
                                            server_tool = json.loads(tools_buffer)
                                            tool_name = server_tool.get("name", "")
                                            if tool_name == "web_search":
                                                message = f"Searching the web for: {server_tool.get('input', {}).get('query', '')}"
                                            await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": message,
                                                        "done": False,
                                                    },
                                                }
                                            )
                                        else:
                                            await self.handle_errors(
                                                Exception(
                                                    f"Malformed tool_use JSON, cannot execute tool. tools_buffer:\n {tools_buffer}" 
                                                ),
                                                __event_emitter__,
                                            )
                                            break

                                    if is_model_thinking:
                                        chunk += "</thinking>"
                                        is_model_thinking = False

                                elif event_type == "message_delta":
                                    delta = getattr(event, "delta", None)
                                    if delta:
                                        stop_reason = getattr(
                                            delta, "stop_reason", None
                                        )
                                        if stop_reason == "tool_use":
                                            if tools_buffer.endswith('"input": '):
                                                tools_buffer += "{}"
                                            tools_buffer += "}"
                                            tool_calls.append(tools_buffer)
                                            tools_buffer = ""
                                            has_pending_tool_calls = True
                                        elif stop_reason == "max_tokens":
                                            chunk += "Claude has Reached the maximum token limit!"
                                        elif stop_reason == "end_turn":
                                            conversation_ended = True
                                        elif stop_reason == "pause_turn":
                                            conversation_ended = True
                                            chunk += "Claude was unable to process this request"

                                elif event_type == "message_stop":
                                    pass

                                elif event_type == "message_error":
                                    error = getattr(event, "error", None)
                                    if error:
                                        # Handle stream errors through handle_errors method
                                        error_details = f"Stream Error: {getattr(error, 'message', str(error))}"
                                        if hasattr(error, 'type'):
                                            error_details = f"Stream Error ({error.type}): {getattr(error, 'message', str(error))}"
                                        
                                        # Create a mock exception for consistent error handling
                                        stream_error = Exception(error_details)
                                        await self.handle_errors(stream_error, __event_emitter__)
                                        return

                                if chunk_count > token_buffer_size:
                                    await __event_emitter__(
                                        {
                                            "type": "chat:message:delta",
                                            "data": {"content": chunk},
                                        }
                                    )
                                    chunk = ""
                                    chunk_count = 0

                        # Handle tool use at the end of the stream
                        if has_pending_tool_calls and tool_calls:
                            # Execute all tools with error handling
                            try:
                                tool_results = await self.execute_tools(
                                    tool_calls, __tools__, __event_emitter__
                                )
                            except Exception as e:
                                # Handle tool execution errors
                                await self.handle_errors(e, __event_emitter__)
                                return

                            # Build assistant message with tool_use blocks
                            assistant_content = []
                            if chunk.strip():
                                assistant_content.append(
                                    {"type": "text", "text": chunk}
                                )

                            # Add tool_use blocks to assistant message
                            for tool_call_json in tool_calls:
                                try:
                                    tool_call_data = json.loads(tool_call_json)
                                    tool_use_block = {
                                        "type": "tool_use",
                                        "id": tool_call_data.get("id", ""),
                                        "name": tool_call_data.get("name", ""),
                                        "input": tool_call_data.get("input", {}),
                                    }
                                    assistant_content.append(tool_use_block)
                                except Exception as e:
                                    if self.valves.DEBUG:
                                        print(
                                            f"🔧 [DEBUG] Failed to parse tool call for assistant message: {e}"
                                        )

                            # Add assistant message to conversation
                            if assistant_content:
                                payload_for_stream["messages"].append(
                                    {"role": "assistant", "content": assistant_content}
                                )

                            # Add user message with tool results
                            user_content = tool_results.copy()
                            if user_content:
                                payload_for_stream["messages"].append(
                                    {"role": "user", "content": user_content}
                                )

                            # Ensure we added at least one message, otherwise break the loop
                            if not assistant_content and not user_content:
                                if self.valves.DEBUG:
                                    print(
                                        f"🔧 [DEBUG] No valid content to add, ending conversation"
                                    )
                                break

                            # Reset state for next iteration

                            current_function_calls += len(tool_calls)
                            has_pending_tool_calls = False
                            tool_calls = []
                            chunk = ""
                            chunk_count = 0
                            continue
                except RateLimitError as e:
                    # Rate limit error (429)
                    await self.handle_errors(e, __event_emitter__)
                    return
                except AuthenticationError as e:
                    # API key issues (401)
                    await self.handle_errors(e, __event_emitter__)
                    return
                except PermissionDeniedError as e:
                    # Permission issues (403)
                    await self.handle_errors(e, __event_emitter__)
                    return
                except NotFoundError as e:
                    # Resource not found (404)
                    await self.handle_errors(e, __event_emitter__)
                    return
                except BadRequestError as e:
                    # Invalid request format (400)
                    await self.handle_errors(e, __event_emitter__)
                    return
                except UnprocessableEntityError as e:
                    # Unprocessable entity (422)
                    await self.handle_errors(e, __event_emitter__)
                    return
                except InternalServerError as e:
                    # Server errors (500, 529)
                    await self.handle_errors(e, __event_emitter__)
                    return
                except APIConnectionError as e:
                    # Network/connection issues
                    await self.handle_errors(e, __event_emitter__)
                    return
                except APIStatusError as e:
                    # Catch any other Anthropic API errors
                    await self.handle_errors(e, __event_emitter__)
                    return
                except Exception as e:
                    # Catch all other exceptions
                    await self.handle_errors(e, __event_emitter__)
                    return
                finally:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "", "done": True, "hidden": True},
                        }
                    )

        except Exception as e:
            await self.handle_errors(e, __event_emitter__)
            return

    async def handle_errors(self, exception, __event_emitter__):
        # Determine specific error message based on exception type
        if isinstance(exception, RateLimitError):
            error_msg = "Rate limit exceeded. Please wait before making more requests."
            user_msg = "⚠️ Rate limit reached. Please try again in a moment."
        elif isinstance(exception, AuthenticationError):
            error_msg = "Authentication failed. Please check your API key."
            user_msg = "🔑 Invalid API key. Please verify your Anthropic API key is correct."
        elif isinstance(exception, PermissionDeniedError):
            error_msg = "Permission denied. Your API key may not have access to this resource."
            user_msg = "🚫 Access denied. Your API key doesn't have permission for this request."
        elif isinstance(exception, NotFoundError):
            error_msg = "Resource not found. The requested model or endpoint may not exist."
            user_msg = "❓ Resource not found. Please check if the model is available."
        elif isinstance(exception, BadRequestError):
            error_msg = f"Bad request: {str(exception)}"
            user_msg = "📝 Invalid request format. Please check your input and try again."
        elif isinstance(exception, UnprocessableEntityError):
            error_msg = f"Unprocessable entity: {str(exception)}"
            user_msg = "📄 Request format issue. Please check your message structure and try again."
        elif isinstance(exception, InternalServerError):
            error_msg = "Anthropic server error. Please try again later."
            user_msg = "🔧 Server temporarily unavailable. Please try again in a few moments."
        elif isinstance(exception, APIConnectionError):
            error_msg = "Network connection error. Please check your internet connection."
            user_msg = "🌐 Connection error. Please check your network and try again."
        elif isinstance(exception, APIStatusError):
            status_code = getattr(exception, 'status_code', 'Unknown')
            error_msg = f"API Error ({status_code}): {str(exception)}"
            user_msg = f"⚡ API Error ({status_code}). Please try again or contact support."
        else:
            error_msg = f"Unexpected error: {str(exception)}"
            user_msg = "💥 An unexpected error occurred. Please try again."

        if self.valves.DEBUG:
            print(f"[DEBUG] Exception: {error_msg}")
            # Add request ID if available for debugging
            if isinstance(exception, APIStatusError) and hasattr(exception, 'response'):
                try:
                    request_id = exception.response.headers.get('request-id')
                    if request_id:
                        print(f"[DEBUG] Request ID: {request_id}")
                except Exception:
                    pass  # Ignore if we can't get request ID

        await __event_emitter__(
            {
                "type": "notification",
                "data": {
                    "type": "error",  # "success", "warning", "error"
                    "content": user_msg,
                },
            }
        )
        import traceback
        tb = traceback.format_exc()
        await __event_emitter__(
            {
                "type": "source",
                "data": {
                    "type": "error",
                    "source": "anthropic api",
                    "content": tb,
                },
            }
        )

    async def execute_tools(
        self,
        tool_calls: list,
        __tools__,
        __event_emitter__,
    ):
        """
        Receives tool_calls in Anthropic format, tries to map them to OpenWebUI __tools__, execute them asynchronously and returns the results
        """

        tool_results = []
        # Step A: For each pending function call, add a "function_call" item,
        # and prepare the async tasks for the actual tool calls.
        tasks = []
        tool_call_data_list = []  # Store parsed tool call data for later use
        tool_call_messages = ""
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Calling Tool",
                    "done": False,
                },
            }
        )

        for i, fc_item in enumerate(tool_calls):
            # Parse the JSON string to get tool call data
            try:
                tool_call_data = json.loads(fc_item)
                tool_type = tool_call_data.get("type", "")
                if tool_type == "server_tool_use":
                    if self.valves.DEBUG:
                        print(f"🔧 [DEBUG] Parsed tool call is Server sided - Skipping")
                    continue
                tool_name = tool_call_data.get("name", "")
                tool_input = tool_call_data.get("input", {})
                tool_id = tool_call_data.get("id", "")
                tool_call_data_list.append(tool_call_data)

                if self.valves.DEBUG:
                    print(
                        f"🔧 [DEBUG] Parsed tool call - name: {tool_name}, input: {tool_input}, id: {tool_id}"
                    )
                tool_call_messages += f"Tool: {tool_name}, Input: {tool_input}\n"

            except Exception as e:
                if self.valves.DEBUG:
                    print(f"🔧 [DEBUG] Failed to parse tool call JSON: {e}")
                continue

            # Skip if we don't have any local tool calls
            if len(tool_call_data_list) == 0:
                if self.valves.DEBUG:
                    print(f"🔧 [DEBUG] No valid tool calls to process")
                return tool_results

            # Look up and queue the tool callable
            tool = __tools__.get(tool_name)
            if tool is None:
                continue

            try:
                # tool_input is already a dict from the parsed JSON
                args = tool_input if isinstance(tool_input, dict) else {}
            except Exception as e:
                args = {}

            tasks.append(asyncio.create_task(tool["callable"](**args)))
            if self.valves.DEBUG:
                print(f"🔧 [DEBUG] Created async task for '{tool_name}'")

        # Step B: Collect the results of all tool calls
        try:
            results = await asyncio.gather(*tasks)
            if self.valves.DEBUG:
                print(
                    f"🔧 [DEBUG] Tool execution completed, got {len(results)} results"
                )
                for i, result in enumerate(results):
                    print(
                        f"🔧 [DEBUG] Result {i}: {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}"
                    )
        except Exception as ex:
            # Tool execution failed - create error results for all tools
            if self.valves.DEBUG:
                print(f"🔧 [DEBUG] Tool execution failed: {ex}")
            
            # Create error messages for each tool that failed
            results = []
            for tool_data in tool_call_data_list:
                tool_name = tool_data.get("name", "unknown")
                error_result = f"Error executing tool '{tool_name}': {str(ex)}"
                results.append(error_result)
        for i, (tool_call_data, tool_result) in enumerate(
            zip(tool_call_data_list, results)
        ):

            # Get the tool_use_id from the parsed data
            tool_use_id = tool_call_data.get("id", "")

            # Determine if the result is an error
            is_error = isinstance(tool_result, str) and tool_result.startswith("Error:")

            # Build the content block
            result_block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": str(tool_result),
            }
            if is_error:
                result_block["is_error"] = True
            tool_results.append(result_block)
            tool_name = tool_call_data.get("name", "")
            # Format tool_result as pretty JSON if possible
            try:
                parsed_json = json.loads(tool_result)
                formatted_result = f"```json\n{json.dumps(parsed_json, indent=2, ensure_ascii=False)}\n```"
            except Exception:
                formatted_result = str(tool_result)

            await __event_emitter__(
                {
                    "type": "chat:message:delta",
                    "data": {
                        "content": (
                            f"\n<details>\n"
                            f"<summary>Results for {tool_name}</summary>\n\n"
                            f"{formatted_result}\n"
                            f"</details>\n"
                        )
                    },
                }
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Tool Call completed", "done": True},
                }
            )
        return tool_results

    async def handle_server_tools_waiting(self, tools_buffer, __event_emitter__):
        """
        Handle server-side tool calls by notifying the user.
        """
        if self.valves.DEBUG:
            print(f"🔧 [DEBUG] Received server-side tool call: {tools_buffer}")

        # Parse the tools_buffer to extract tool information
        try:
            tool_data = json.loads(tools_buffer)
            tool_name = tool_data.get("name", "unknown")
            tool_input = tool_data.get("input", {})

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Executing server-side tool: {tool_name}",
                        "done": False,
                    },
                }
            )
        except json.JSONDecodeError:
            if self.valves.DEBUG:
                print(f"🔧 [DEBUG] Failed to parse tools_buffer: {tools_buffer}")

        # In a full implementation, we would parse tools_buffer and handle accordingly.
        # For now, just log it.

    def _process_content(self, content: Union[str, List[dict]]) -> List[dict]:
        """
        Process content from OpenWebUI format to Claude API format.
        Handles text, images, PDFs, tool_calls, and tool_results.
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        processed_content = []
        for item in content:
            if item["type"] == "text":
                processed_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                # Simple image processing - for now just pass through
                # In a full implementation, we'd convert data URLs to Claude format
                processed_content.append(
                    {
                        "type": "text",
                        "text": "[Image content not supported in simplified version]",
                    }
                )
            elif item["type"] == "tool_calls":
                # Convert OpenWebUI tool_calls to Claude tool_use format
                converted_calls = self._process_tool_calls(item)
                processed_content.extend(converted_calls)
            elif item["type"] == "tool_results":
                # Convert OpenWebUI tool_results to Claude tool_result format
                converted_results = self._process_tool_results(item)
                processed_content.extend(converted_results)
        return processed_content

    def _process_tool_calls(self, tool_calls_item):
        """
        Convert OpenWebUI tool_calls format to Claude tool_use format.
        """
        claude_tool_uses = []

        if "tool_calls" in tool_calls_item:
            for tool_call in tool_calls_item["tool_calls"]:
                if tool_call.get("type") == "function" and "function" in tool_call:
                    function_def = tool_call["function"]

                    claude_tool_use = {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": function_def.get("name", ""),
                        "input": function_def.get("arguments", {}),
                    }
                    claude_tool_uses.append(claude_tool_use)

        return claude_tool_uses

    def _process_tool_results(self, tool_results_item):
        """
        Convert OpenWebUI tool_results format to Claude tool_result format.
        """
        claude_tool_results = []

        if "results" in tool_results_item:
            for result_item in tool_results_item["results"]:
                if "call" in result_item and "result" in result_item:
                    tool_call = result_item["call"]
                    result_content = result_item["result"]

                    # Extract tool_use_id from the call
                    tool_use_id = tool_call.get("id", "")

                    if tool_use_id:
                        claude_result = {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(result_content),
                        }
                        claude_tool_results.append(claude_result)

        return claude_tool_results

    async def handle_citation(self, event, __event_emitter__):
        """
        Handle web search citation events from Anthropic API and emit appropriate source events to OpenWebUI.

        Args:
            event: The citation event from Anthropic (content_block_delta with citations_delta)
            __event_emitter__: OpenWebUI event emitter function
        """
        try:
            if self.valves.DEBUG:
                print(
                    f"[DEBUG] Processing citation event type: {getattr(event, 'type', 'unknown')}"
                )

            # Extract citation from delta within content_block_delta event
            delta = getattr(event, "delta", None)
            citation = None

            if delta and hasattr(delta, "citation"):
                citation = delta.citation
            elif hasattr(event, "citation"):
                # Fallback: direct citation in event
                citation = event.citation

            if not citation:
                if self.valves.DEBUG:
                    print(f"[DEBUG] No citation data found in event")
                return

            # Only handle web search result citations
            citation_type = getattr(citation, "type", "")
            if citation_type != "web_search_result_location":
                if self.valves.DEBUG:
                    print(
                        f"[DEBUG] Skipping non-web-search citation type: {citation_type}"
                    )
                return

            # Extract web search citation information
            url = getattr(citation, "url", "")
            title = getattr(citation, "title", "Unknown Source")
            encrypted_index = getattr(citation, "encrypted_index", "")
            cited_text = getattr(citation, "cited_text", "")

            if self.valves.DEBUG:
                print(f"[DEBUG] Web search citation - URL: {url}, Title: {title}")
                print(f"[DEBUG] Cited text: {cited_text[:100]}...")

            # Build source event data for OpenWebUI
            from datetime import datetime

            source_data = {
                "document": [cited_text] if cited_text else [title],
                "metadata": [
                    {
                        "source": title,
                        "url": url,
                        "citation_type": "web_search_result_location",
                        "date_accessed": datetime.now().isoformat(),
                        "cited_text": cited_text,
                        "encrypted_index": encrypted_index,
                    }
                ],
                "source": {"name": title, "url": url},
            }

            # Emit the source event
            await __event_emitter__({"type": "source", "data": source_data})

            if self.valves.DEBUG:
                print(f"[DEBUG] Emitted web search citation for '{title}' - {url}")

        except Exception as e:
            if self.valves.DEBUG:
                print(f"[DEBUG] Error handling citation: {str(e)}")
            await self.handle_errors(e, __event_emitter__)
                

    def _process_messages(self, messages: List[dict]) -> List[dict]:
        """
        Process messages for the Anthropic API format with full content processing.
        """
        processed_messages = []
        for message in messages:
            # Skip system messages - they are handled separately
            if message.get("role") == "system":
                continue

            # Process content using the full content processor
            processed_content = self._process_content(message["content"])

            if processed_content:  # Only add messages with content
                processed_messages.append(
                    {"role": message["role"], "content": processed_content}
                )
        return processed_messages

    async def _send_request(
        self, url: str, headers: dict, payload: dict
    ) -> tuple[dict, Optional[dict]]:
        """
        Send a request to the Anthropic API with enhanced retry logic.

        Args:
            url: The API endpoint URL
            headers: Request headers
            payload: Request payload

        Returns:
            Tuple of (response_data, cache_metrics)
        """
        retry_count = 0
        base_delay = 1  # Start with 1 second delay
        max_retries = 5  # Increased from 3 to 5 for better reliability
        retry_status_codes = [429, 500, 502, 503, 504]  # Status codes to retry on

        while retry_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                    async with session.post(
                        url, headers=headers, json=payload, timeout=timeout
                    ) as response:
                        self.request_id = response.headers.get("x-request-id")
                        org_id = response.headers.get("anthropic-organization-id")

                        response_text = await response.text()

                        # Handle rate limiting and server errors with exponential backoff
                        if response.status in retry_status_codes:
                            # Use retry-after header if available, otherwise use exponential backoff
                            retry_after = int(
                                response.headers.get(
                                    "retry-after", base_delay * (2**retry_count)
                                )
                            )
                            # Add jitter to avoid thundering herd problem
                            jitter = random.uniform(0, 0.1 * retry_after)
                            retry_time = retry_after + jitter

                            logging.warning(
                                f"Request failed with status {response.status}. "
                                f"Retrying in {retry_time:.2f} seconds. "
                                f"Retry count: {retry_count + 1}/{max_retries}"
                            )
                            await asyncio.sleep(retry_time)
                            retry_count += 1
                            continue

                        if response.status != 200:
                            error_msg = f"Error: HTTP {response.status}"
                            try:
                                error_data = json.loads(response_text).get("error", {})
                                error_msg += (
                                    f": {error_data.get('message', response_text)}"
                                )
                                # Include error type and code if available
                                if error_data.get("type"):
                                    error_msg += f" (Type: {error_data.get('type')})"
                                if error_data.get("code"):
                                    error_msg += f" (Code: {error_data.get('code')})"
                            except:
                                error_msg += f": {response_text}"

                            if self.request_id:
                                error_msg += f" (Request ID: {self.request_id})"

                            logging.error(error_msg)
                            return {"content": error_msg, "format": "text"}, None

                        result = json.loads(response_text)
                        usage = result.get("usage", {})
                        cache_metrics = {
                            "cache_creation_input_tokens": usage.get(
                                "cache_creation_input_tokens", 0
                            ),
                            "cache_read_input_tokens": usage.get(
                                "cache_read_input_tokens", 0
                            ),
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                        }

                        # Log usage metrics for monitoring
                        logging.info(
                            f"Request successful. Input tokens: {usage.get('input_tokens', 0)}, "
                            f"Output tokens: {usage.get('output_tokens', 0)}"
                        )

                        return result, cache_metrics

            except aiohttp.ClientError as e:
                logging.error(f"Request failed: {str(e)}")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    retry_time = base_delay * (2**retry_count)
                    logging.info(
                        f"Retrying in {retry_time} seconds. Retry count: {retry_count}/{max_retries}"
                    )
                    await asyncio.sleep(retry_time)
                    continue
                raise
            except asyncio.TimeoutError:
                logging.error(f"Request timed out after {self.REQUEST_TIMEOUT} seconds")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    retry_time = base_delay * (2**retry_count)
                    logging.info(
                        f"Retrying in {retry_time} seconds after timeout. Retry count: {retry_count}/{max_retries}"
                    )
                    await asyncio.sleep(retry_time)
                    continue
                raise

        logging.error(f"Max retries ({max_retries}) exceeded.")
        return {
            "content": f"Max retries ({max_retries}) exceeded",
            "format": "text",
        }, None
