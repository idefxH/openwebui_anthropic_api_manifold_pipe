"""
title: Anthropic API Integration
author: Podden (https://github.com/Podden/)
original_author: Balaxxe (Updated by nbellochi)
version: 0.3.5
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
- Image processing
- Web_Search Toggle Action
- Fine Grained Tool Streaming
- Extended Thinking Toggle Action
- Code Execution Tool
- Vision

Todo:
- Files API
- PDF support
- API Key from UserValves as Alternative
- MCP Connector (mcpo is enought for me atm)

Changelog:
v0.3.4
- Added Claude 4.5 Sonnet
- Small Bugfix with final_message
- Added OpenWebUI Token Usage Compatibility
- Added a Check for Duplicate Tool Names and private tool name (starting with "_") to avoid API errors

v0.3.3
- Fixed Tool Call error

v0.3.2
- Fixed type and added changelog

v0.3.1
- Fixed a bug where message would disappear after Error occurs

v0.3
- Added Vision support (__files__ handling & image processing improvements)
- Added Extended Thinking filter & metadata override with clamped budget logic (default 10K, safe min/max enforcement)
- Added Web Search Enforcement toggle (one‑shot metadata flag forces web_search tool_choice)
- Added Anthropic Code Execution Tool with toggle filter & beta header
- Enabled fine‑grained tool streaming beta by default
- Added metadata & valve controlled injection of code execution tool spec
- Improved cache control: auto‑disables cache when dynamic Memory / RAG blocks detected; ephemeral caching for stable blocks
- Refined tool_choice precedence (enforced web search before auto)
- Added 1M context optional beta header for supported Sonnet 4 models
- Improved malformed tool_use JSON salvage (_finalize_tool_buffer) & robust final chunk flush
- Misc debug output refinements & system prompt cleanup

v0.2
- Fixed caching by moving Memories to Messages instead of system prompt
- You can show Cache Usage Statistics with a Valve as Source Event
- Fixed error where last chunk is not shown in frontend
- Fixed defective event_emitters and removed unneeded method
- Fixed unnecessary requirements
- Implemented Web Search Valves and error handling
- Robust error handling
- Added Cache_Control for System_Prompt, Tools, and Message Array
- Refactored for readability and support for new models
"""

from collections.abc import Awaitable
import json
import logging
import asyncio
import random
from typing import Any, Callable, List, Union, Dict, Optional
from pydantic import BaseModel, Field
import aiohttp
from anthropic import (
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)
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
        "claude-opus-4-1-20250805": 32000,  # 32K by default
        "claude-sonnet-4-5-20250929": 32000,  # 32K by default
        # Latest aliases
        "claude-3-opus-latest": 4096,
        "claude-3-sonnet-latest": 4096,
        "claude-3-haiku-latest": 4096,
        "claude-3-5-sonnet-latest": 8192,
        "claude-3-5-haiku-latest": 8192,
        "claude-3-7-sonnet-latest": 16384,  # 16K by default, 128K with beta
        "claude-sonnet-4-latest": 32000,  # 32K by default, 128K with beta
        "claude-opus-4-latest": 32000,  # 32K by default, 128K with beta
        "claude-opus-4-1-latest": 32000,  # 32K by default
        "claude-sonnet-4-5-latest": 32000,  # 32K by default
   
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
        "claude-opus-4-1-20250805": 200000,
        "claude-sonnet-4-5-20250929": 1000000,
        # Latest aliases
        "claude-3-opus-latest": 200000,
        "claude-3-sonnet-latest": 200000,
        "claude-3-haiku-latest": 200000,
        "claude-3-5-sonnet-latest": 200000,
        "claude-3-5-haiku-latest": 200000,
        "claude-3-7-sonnet-latest": 200000,
        "claude-sonnet-4-latest": 1000000,
        "claude-opus-4-latest": 200000,
        "claude-opus-4-1-latest": 200000,
        "claude-sonnet-4-5-latest": 1000000,
    
    }
    THINKING_SUPPORTED_MODELS = [
        "claude-3-7-sonnet-latest",
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-5-20250929"
        "claude-sonnet-4-latest",
        "claude-opus-4-latest",
        "claude-opus-4-1-latest",
        "claude-sonnet-4-5-latest"
    ]
    MODELS_SUPPORTING_1M_CONTEXT = [
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-5-20250929"
        "claude-sonnet-4-latest",
        "claude-sonnet-4-5-latest"
        
    ]
    MODELS_SUPPORTING_MEMORY_TOOL = [
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-20250514",
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-5-latest",
        "claude-sonnet-4-latest",
        "claude-opus-4-1-latest",
        "claude-opus-4-latest",
    ]

    REQUEST_TIMEOUT = (
        300  # Increased timeout for longer responses with extended thinking
    )
    THINKING_BUDGET_TOKENS = 4096  # Default thinking budget tokens (max 16K)

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = "Your API Key Here"
        ENABLE_THINKING: bool = Field(
            default=False,
            description="Force Enable Extended Thinking. Use Anthropic Thinking Toggle Function for fine grained control",
        )
        MAX_OUTPUT_TOKENS: bool = True  # Valve to use maximum possible output tokens
        THINKING_BUDGET_TOKENS: int = Field(
            default=4096, ge=0, le=32000
        )
        # ENABLE_CLAUDE_MEMORY: bool = Field(
        #     default=False,
        #     description="Enable Claude memory tool",
        # )
        ENABLE_1M_CONTEXT: bool = Field(
            default=False,
            description="Enable 1M token context window for Claude Sonnet 4 (requires Tier 4 API access)",
        )
        WEB_SEARCH: bool = Field(
            default=True,
            description="Enable web search tool for Claude models. Use Anthropic Web Search Toggle Function for fine grained control",
        )
        WEB_SEARCH_MAX_USES: int = Field(
            default=5,
            ge=1,
            le=20,
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
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-5-20250929"
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

    async def _create_payload(
        self,
        body: Dict,
        __metadata__: dict[str, Any],
        __user__: Optional[dict],
        __tools__: Optional[Dict[str, Dict[str, Any]]],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __files__: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, dict]:
        model_name = body["model"].split("/")[-1]
        is_thinking_variant = model_name.endswith("-thinking")
        actual_model_name = (
            model_name.replace("-thinking", "") if is_thinking_variant else model_name
        )

        max_tokens_limit = self.MODEL_MAX_TOKENS.get(actual_model_name, 4096)
        requested_max_tokens = body.get("max_tokens", max_tokens_limit)
        max_tokens = (
            max_tokens_limit
            if self.valves.MAX_OUTPUT_TOKENS
            else min(requested_max_tokens, max_tokens_limit)
        )
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

        if self.valves.DEBUG:
            try:
                print(
                    f"[DEBUG] Thinking Filter: {__metadata__.get('anthropic_thinking')}"
                )
                print(f"[DEBUG] Tools: {json.dumps(__tools__, indent=2)}")
            except Exception as e:
                print(f"[DEBUG] JSON dump failed: {e}")
                print(f"[DEBUG] raw __metadata__: {__metadata__}")

        if "anthropic_thinking" in __metadata__:
            should_enable_thinking = __metadata__.get("anthropic_thinking", False)
        elif actual_model_name in self.THINKING_SUPPORTED_MODELS:
            # Hybrid models: enable thinking only when -thinking variant chosen
            should_enable_thinking = is_thinking_variant
        else:
            should_enable_thinking = self.valves.ENABLE_THINKING

        if self.valves.DEBUG:
            print(f"[DEBUG] Thinking Enabled?: {should_enable_thinking}")

        if (
            should_enable_thinking
            and actual_model_name in self.THINKING_SUPPORTED_MODELS
        ):
            # Ensure thinking.budget_tokens < max_tokens and at least 1024
            requested_thinking_budget = self.valves.THINKING_BUDGET_TOKENS
            # Clamp thinking budget to valid range
            max_valid_thinking_budget = max(max_tokens - 1, 1023)
            thinking_budget = max(
                1024, min(requested_thinking_budget, max_valid_thinking_budget)
            )
            # If max_tokens is too low, set thinking_budget to max_tokens - 1
            if max_tokens <= 1024:
                thinking_budget = max_tokens - 1 if max_tokens > 1 else 1
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        if "response_format" in body:
            payload["response_format"] = {"type": body["response_format"].get("type")}

        raw_messages = body.get("messages", []) or []
        system_messages = []
        processed_messages: list[dict] = []
        # Extract dynamic context from system messages for injection into user messages
        dynamic_context_blocks = []
        disable_cache = False
        # if self.valves.DEBUG:
        #     print(f"[DEBUG] Raw Messages: {json.dumps(raw_messages, indent=2)}")
        for msg in raw_messages:
            role = msg.get("role")
            processed_content = self._process_content(msg.get("content"))
            if not processed_content:
                continue
            if role == "system":
                for block in processed_content:
                    text = block["text"]
                    # Extract dynamic context blocks (Memory, RAG, Documents)
                    extracted_context = self._extract_dynamic_context(text)
                    if extracted_context:
                        dynamic_context_blocks.append(extracted_context)
                        # Remove from system prompt completely
                        text = self._remove_dynamic_context(text)
                        block["text"] = text
                    
                    # Only add non-empty blocks to system
                    if (len(text.strip()) > 0) and (not text.isspace()):
                        block["cache_control"] = {"type": "ephemeral"}
                        system_messages.append(block)
            else:
                # Keep all historical messages as-is (they contain context from their time)
                processed_messages.append({"role": role, "content": processed_content})

        if not processed_messages:
            raise ValueError("No valid messages to process")
        # if self.valves.DEBUG:
        #     print(
        #         f"[DEBUG] Processed Messages: {json.dumps(processed_messages, indent=2)}"
        #     )
        #     print(f"[DEBUG] System Messages: {json.dumps(system_messages, indent=2)}")
        #     print(f"[DEBUG] Disable Cache: {disable_cache}")
        # Correct Order for Caching: Tools, System, Messages
        tools_list = self._convert_tools_to_claude_format(__tools__, actual_model_name)
        # Decide on code execution inclusion early so we can set beta headers later
        activate_code_execution = __metadata__.get(
            "activate_code_execution_tool", False
        )
        # Append code execution tool (no parameters) if enabled
        if activate_code_execution:
            code_exec_tool = {
                "type": "code_execution_20250522",
                "name": "code_execution",
            }
            # Avoid duplicates if already added
            if not any(t.get("name") == "code_execution" for t in tools_list):
                tools_list.append(code_exec_tool)

        if tools_list and len(tools_list) > 0:
            if not disable_cache and tools_list:
                tools_list[-1]["cache_control"] = {"type": "ephemeral"}

        if tools_list:
            # Check for enforced web search or code execution in metadata (precedence: specific enforcement first)
            if __metadata__.get("web_search_enforced") and "thinking" not in payload:
                payload["tool_choice"] = {"type": "tool", "name": "web_search"}
                __metadata__["web_search_enforced"] = False  # one-shot
            else:
                payload["tool_choice"] = {"type": "auto"}
                # Skip forced web search when thinking is enabled to avoid API error
                if __metadata__.get("web_search_enforced") and "thinking" in payload:
                    __metadata__["web_search_enforced"] = False  # one-shot
                    if self.valves.DEBUG:
                        print("[DEBUG] Skipped forced web_search due to active thinking")
                    # Notify user about the conflict
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {
                                "type": "info",
                                "content": "🧠 Thinking mode is active - Web search enforcement was disabled to allow extended thinking. Claude can still use web search if needed.",
                            },
                        }
                    )
            payload["tools"] = tools_list

        if system_messages and len(system_messages) > 0:
            payload["system"] = system_messages

        if processed_messages and len(processed_messages) > 0:
            last_msg = processed_messages[-1]
            content_blocks = last_msg.get("content", [])
            
            # Inject dynamic context into last user message
            if dynamic_context_blocks and last_msg.get("role") == "user":
                # Add context blocks as text blocks with clear markers
                for ctx in dynamic_context_blocks:
                    context_block = {
                        "type": "text",
                        "text": f"\n\n<!-- INJECTED_CONTEXT_START -->\n{ctx}\n<!-- INJECTED_CONTEXT_END -->\n"
                    }
                    content_blocks.append(context_block)
                if self.valves.DEBUG:
                    print(f"[DEBUG] Injected {len(dynamic_context_blocks)} context block(s) into last user message")
            
            # Apply cache control to last content block
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

        # Enable fine-grained tool streaming beta unconditionally for now (AIT-86)
        # This only affects chunking of tool input JSON; execution logic unchanged.
        beta_headers.append("fine-grained-tool-streaming-2025-05-14")

        # Add code execution beta if valve or metadata enforcement active
        if activate_code_execution:
            beta_headers.append("code-execution-2025-05-22")

        # Add 1M context header if enabled and model supports it
        if self.valves.ENABLE_1M_CONTEXT and actual_model_name in self.MODELS_SUPPORTING_1M_CONTEXT:
            beta_headers.append("context-1m-2025-08-07")

        # Add Memory Tool Beta Header if enabled and model supports it
        # if self.valves.ENABLE_CLAUDE_MEMORY and actual_model_name in self.MODELS_SUPPORTING_MEMORY_TOOL:
        #     beta_headers.append("context-management-2025-06-27")

        if beta_headers and len(beta_headers) > 0:
            headers["anthropic-beta"] = ",".join(beta_headers)

        # if self.valves.DEBUG:
        #     print(f"[DEBUG] Payload: {json.dumps(payload, indent=2)}")
        #     print(f"[DEBUG] Headers: {headers}")
        return payload, headers

    def _convert_tools_to_claude_format(self, __tools__, actual_model_name: str) -> List[dict]:
        """
        Convert OpenWebUI tools format to Claude API format.
        Args:
            __tools__: Dict of tools from OpenWebUI
        Returns:
            list: Tools in Claude API format
        """
        claude_tools = []
        tool_names_seen = set()  # Track unique tool names

        if self.valves.DEBUG:
            # Only log tool names and specs, not the callable functions
            if __tools__:
                try:
                    print(
                        f"[DEBUG] Converting tools: {json.dumps(__tools__, indent=2)}"
                    )
                except Exception as e:
                    print(
                        f"[DEBUG] JSON dump failed, printing tools directly: {__tools__}"
                    )
                    print(f"[DEBUG] Error was: {e}")
            else:
                print(f"[DEBUG] No tools to convert")

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
            tool_names_seen.add("web_search")

        # Add Claude Memory tool if enabled and supported by model
        # if self.valves.ENABLE_CLAUDE_MEMORY and actual_model_name in self.MODELS_SUPPORTING_MEMORY_TOOL:
        #     claude_tools.append(
        #         {
        #             "type": "memory_20250818",
        #             "name": "memory"
        #         }
        #     )
        #     tool_names_seen.add("memory")

        if not __tools__ or len(__tools__) == 0:
            if self.valves.DEBUG:
                print(f"[DEBUG] No tools provided, using default Claude tools")
            return claude_tools

        for tool_name, tool_data in __tools__.items():
            if not isinstance(tool_data, dict) or "spec" not in tool_data:
                if self.valves.DEBUG:
                    print(f"[DEBUG] Skipping invalid tool: {tool_name} - missing spec")
                continue

            spec = tool_data["spec"]

            # Extract basic tool info
            name = spec.get("name", tool_name)
            
            # Skip if tool name already exists
            if name in tool_names_seen:
                print(f"[ANTHROPIC] Skipping duplicate tool: {name}")
                continue

            # Skip if toolname starts with _ or __
            if name.startswith("_"):
                if self.valves.DEBUG:
                    print(f"[DEBUG] Skipping private tool: {name}")
                continue
            
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
            tool_names_seen.add(name)

            if self.valves.DEBUG:
                print(f"[DEBUG] Converted tool '{name}' to Claude format")

        if self.valves.DEBUG:
            print(f"[DEBUG] Total tools converted: {len(claude_tools)}")

        # Cache control for tools handled during payload creation where disable_cache is in scope.
        return claude_tools

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __metadata__: dict[str, Any] = {},
        __tools__: Optional[Dict[str, Dict[str, Any]]] = None,
        __files__: Optional[Dict[str, Any]] = None,
    ):
        final_message = ""
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
            return error_msg

        try:
            if inspect.isawaitable(__tools__):
                __tools__ = await __tools__

            try:
                payload, headers = await self._create_payload(
                    body, __metadata__, __user__, __tools__, __event_emitter__, __files__
                )
            except Exception as e:
                # Handle payload creation errors
                await self.handle_errors(e, __event_emitter__)
                return f"\n\nException of type {type(e).__name__} occurred. Reason: {e}"

            if payload.get("stream"):
                try:
                    api_key = self.valves.ANTHROPIC_API_KEY
                    client = AsyncAnthropic(api_key=api_key, default_headers=headers)
                    payload_for_stream = {
                        k: v for k, v in payload.items() if k != "stream"
                    }
                except Exception as e:
                    # Handle client creation errors
                    await self.handle_errors(e, __event_emitter__)
                    return f"\n\nException of type {type(e).__name__} occurred. Reason: {e}"

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
                final_message = ""
                thinking_message = ""
                thinking_blocks = []  # Preserve thinking blocks for multi-turn
                current_thinking_block = {}  # Track current thinking block
                current_search_query = ""  # Track the current web search query
                citation_counter = 0  # Track citation numbers for inline citations
                citations_list = []  # Store citations for reference list
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
                                            f"[Anthropic] Received event: {event_type} with {str(event_dict)[:200]}{'...' if len(str(event_dict)) > 200 else ''}"
                                        )
                                    else:
                                        print(
                                            f"[Anthropic] Received event: {event_type} with {str(event)[:200]}{'...' if len(str(event)) > 200 else ''}"
                                        )
                                if event_type == "message_start":
                                    message = getattr(event, "message", None)
                                    if message:
                                        self.request_id = getattr(
                                            message, "id", None
                                        )
                                        if self.valves.DEBUG:
                                            print(
                                                f"[DEBUG] Message started with ID: {self.request_id}"
                                            )
                                        usage = getattr(message, "usage", {})
                                        if usage:
                                            input_tokens = getattr(
                                                usage, "input_tokens", 0
                                            )
                                            output_tokens = getattr(
                                                usage, "output_tokens", 0
                                            )
                                            cache_creation_input_tokens = getattr(
                                                usage,
                                                "cache_creation_input_tokens",
                                                0,
                                            )
                                            cache_read_input_tokens = getattr(
                                                usage, "cache_read_input_tokens", 0
                                            )
                                            if self.valves.DEBUG:
                                                print(
                                                    f"[DEBUG] Usage stats: input={input_tokens}, output={output_tokens}, cache_creation={cache_creation_input_tokens}, cache_read={cache_read_input_tokens}"
                                                )
                                            
                                            # Emit usage as completion event for OpenWebUI Generation Info
                                            usage_data = {
                                                "input tokens": input_tokens,
                                                "output tokens": output_tokens,
                                                "total_tokens": input_tokens + output_tokens,
                                            }
                                            # Add Anthropic-specific cache fields if present
                                            if cache_creation_input_tokens > 0:
                                                usage_data["cache_creation_input_tokens"] = cache_creation_input_tokens
                                            if cache_read_input_tokens > 0:
                                                usage_data["cache_read_input_tokens"] = cache_read_input_tokens
                                            
                                            await __event_emitter__(
                                                {
                                                    "type": "chat:completion",
                                                    "data": {
                                                        "usage": usage_data,
                                                        "done": False,
                                                    },
                                                }
                                            )

                                elif event_type == "content_block_start":
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
                                        thinking_message = "\n<details>\n<summary>Claude is Thinking...</summary>\n"
                                        await __event_emitter__(
                                                {
                                                    "type": "status",
                              ""                      "data": {
                                                        "description": "Thinking...",
                                                        "done": False,
                                                    },
                                                }
                                            )
                                    if content_type == "tool_use":
                                        tools_buffer = (
                                            "{"
                                            f'"type": "{content_block.type}", '
                                            f'"id": "{content_block.id}", '
                                            f'"name": "{content_block.name}", '
                                            f'"input": '
                                        )

                                    if content_type == "server_tool_use":
                                        name = getattr(content_block, "name", "")
                                        if name == "code_execution":
                                            await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": "Executing Code...",
                                                        "done": False,
                                                    },
                                                }
                                            )

                                    if content_type == "code_execution_tool_result":
                                        result_block = getattr(
                                            content_block, "content", {}
                                        )
                                        stdout = result_block.get("stdout", "")
                                        stderr = result_block.get("stderr", "")
                                        if stdout or stderr:
                                            await __event_emitter__(
                                                {
                                                    "type": "chat:message:delta",
                                                    "data": {
                                                        "content": (
                                                            f"\n<details>\n"
                                                            f"<summary>Code Execution Result</summary>\n\n"
                                                            + (
                                                                f"```\n{stdout}\n```\n"
                                                                if stdout
                                                                else ""
                                                            )
                                                            + (
                                                                f"```\n{stderr}\n```"
                                                                if stderr
                                                                else ""
                                                            )
                                                            + f"</details>\n"
                                                        )
                                                    },
                                                }
                                            )
                                    if content_type == "web_search_tool_result":
                                        if self.valves.DEBUG:
                                            print(
                                                f"[DEBUG] Processing web search result event: {event}"
                                            )
                                        content_items = getattr(
                                            content_block, "content", []
                                        )
                                        if content_items and len(content_items) > 0:
                                            error_code = getattr(
                                                content_block, "error_code", None
                                            )
                                            if error_code:
                                                await self.handle_errors(
                                                    Exception(
                                                        f"Web search error: {error_code}"
                                                    ),
                                                    __event_emitter__,
                                                )
                                            else:
                                                # Extract first result title for status
                                                first_result = content_items[0] if content_items else None
                                                result_title = getattr(first_result, "title", "") if first_result else ""
                                                result_count = len(content_items)
                                                
                                                if result_title and result_count > 0:
                                                    status_desc = f"Found {result_count} results - {result_title}"
                                                    if result_count > 1:
                                                        status_desc += f" +{result_count-1} more"
                                                else:
                                                    status_desc = "Web Search Complete"
                                                
                                                await __event_emitter__(
                                                    {
                                                        "type": "status",
                                                        "data": {
                                                            "description": status_desc,
                                                            "done": True,
                                                        },
                                                    }
                                                )

                                elif event_type == "content_block_delta":
                                    delta = getattr(event, "delta", None)
                                    if delta:
                                        delta_type = getattr(delta, "type", None)
                                        if delta_type == "thinking_delta":
                                            thinking_text = getattr(delta, "thinking", "")
                                            thinking_message += thinking_text
                                            # Preserve thinking for API
                                            if current_thinking_block:
                                                current_thinking_block["thinking"] += thinking_text
                                        elif delta_type == "signature_delta":
                                            # Capture signature for thinking block preservation
                                            signature = getattr(delta, "signature", "")
                                            if current_thinking_block and signature:
                                                current_thinking_block["signature"] = signature
                                        elif delta_type == "text_delta":
                                            text_delta = getattr(delta, "text", "")
                                            chunk += text_delta
                                            chunk_count += 1
                                        elif delta_type == "input_json_delta":
                                            partial = getattr(delta, "partial_json", "")
                                            tools_buffer += partial
                                            
                                            # Try to extract search query from partial JSON for web_search
                                            try:
                                                if self.valves.DEBUG:
                                                    print(f"[DEBUG] Tools buffer update: {tools_buffer}")
                                                
                                                # Check if we have a complete query JSON
                                                
                                                # Try to parse as complete JSON to check if it's valid
                                                try:
                                                    parsed = json.loads(tools_buffer)
                                                    if 'query' in parsed:
                                                        new_query = parsed['query']
                                                        if self.valves.DEBUG:
                                                            print(f"[DEBUG] Complete JSON found with query: '{new_query}'")
                                                            
                                                        # Emit status only once when we have the complete query
                                                        if new_query and new_query != current_search_query:
                                                            current_search_query = new_query
                                                            if self.valves.DEBUG:
                                                                print(f"[DEBUG] Emitting status for complete query: {current_search_query}")
                                                            
                                                            await __event_emitter__(
                                                                {
                                                                    "type": "status",
                                                                    "data": {
                                                                        "description": f"🔍 Searching for: {current_search_query}",
                                                                        "done": False,
                                                                    },
                                                                }
                                                            )
                                                except json.JSONDecodeError:
                                                    # JSON is not complete yet, continue streaming
                                                    if self.valves.DEBUG:
                                                        print(f"[DEBUG] JSON not complete yet, waiting for more data")
                                                    pass
                                                
                                            except Exception as e:
                                                if self.valves.DEBUG:
                                                    print(f"[DEBUG] Query extraction error: {e}")
                                                pass
                                        elif delta_type == "citations_delta":
                                            # Handle citations within content_block_delta AND add inline citation number
                                            citation_counter += 1
                                            # Add inline citation number to chunk
                                            chunk += f"[{citation_counter}]"
                                            chunk_count += 1
                                            
                                            # Process and store citation
                                            await self.handle_citation(
                                                event, __event_emitter__, citation_counter
                                            )

                                elif event_type == "content_block_stop":
                                    content_block = getattr(
                                        event, "content_block", None
                                    )
                                    content_type = (
                                        getattr(content_block, "type", None)
                                        if content_block
                                        else None
                                    )
                                    event_name = getattr(event, "name", "")

                                    # Close tools_buffer for normal tool_use content blocks
                                    if content_type == "tool_use" and tools_buffer:
                                        # Check if it's valid JSON already, if not close it
                                        try:
                                            json.loads(tools_buffer)
                                            # Already valid JSON, no need to close
                                            if self.valves.DEBUG:
                                                print(
                                                    f"[DEBUG] tools_buffer already valid JSON: {tools_buffer}"
                                                )
                                        except json.JSONDecodeError:
                                            # Check if input is empty (ends with "input": )
                                            if tools_buffer.rstrip().endswith('"input":') or tools_buffer.rstrip().endswith('"input": '):
                                                # Add empty object for input
                                                tools_buffer += ' {}'
                                                if self.valves.DEBUG:
                                                    print(
                                                        f"[DEBUG] Added empty input object: {tools_buffer}"
                                                    )
                                            # Invalid JSON, need to close the main object
                                            tools_buffer += "}"
                                            if self.valves.DEBUG:
                                                print(
                                                    f"[DEBUG] Closed tools_buffer in content_block_stop: {tools_buffer}"
                                                )

                                    if event_name == "web_search":
                                        # Finalize tools_buffer into valid JSON (may wrap invalid JSON)
                                        try:
                                            finalized = self._finalize_tool_buffer(
                                                tools_buffer
                                            )
                                            server_tool = json.loads(finalized)
                                            tool_name = server_tool.get("name", "")
                                            if tool_name == "web_search":
                                                query = server_tool.get('input', {}).get('query', current_search_query)
                                                if query:
                                                    await __event_emitter__(
                                                        {
                                                            "type": "status",
                                                            "data": {
                                                                "description": f"Searching: {query}",
                                                                "done": False,
                                                            },
                                                        }
                                                    )
                                        except Exception as e:
                                            # If even finalization failed, emit an error but continue gracefully
                                            await self.handle_errors(
                                                Exception(
                                                    f"Malformed tool_use JSON, cannot execute tool. tools_buffer:\n {tools_buffer} | error: {e}"
                                                ),
                                                __event_emitter__,
                                            )
                                            break

                                    if is_model_thinking:
                                        thinking_message += "\n</details>"
                                        # Preserve thinking block for multi-turn (API auto-filters)
                                        if current_thinking_block and current_thinking_block.get("thinking"):
                                            thinking_blocks.append(current_thinking_block)
                                            if self.valves.DEBUG:
                                                print(f"[DEBUG] Preserved thinking block with {len(current_thinking_block.get('thinking', ''))} chars")
                                        await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": "Thinking Completed",
                                                        "done": True,
                                                    },
                                                }
                                            )
                                        # Don't add thinking to final_message (only show in UI)
                                        await __event_emitter__(
                                        {
                                            "type": "chat:message:delta",
                                            "data": {"content": thinking_message},
                                        }
                                        )
                                        is_model_thinking = False
                                        current_thinking_block = {}

                                elif event_type == "message_delta":
                                    delta = getattr(event, "delta", None)
                                    if delta:
                                        stop_reason = getattr(
                                            delta, "stop_reason", None
                                        )
                                        if stop_reason == "tool_use":
                                            # Ensure tools_buffer is finalized into valid JSON
                                            try:
                                                finalized = self._finalize_tool_buffer(
                                                    tools_buffer
                                                )
                                                tool_calls.append(finalized)
                                                if self.valves.DEBUG:
                                                    print(
                                                        f"[DEBUG] Tool use detected, finalizing tools_buffer: {tools_buffer}\nTool_Call JSON: {tool_calls}"
                                                    )
                                            except Exception:
                                                # If finalization fails, still append a wrapped payload
                                                tool_calls.append(
                                                    self._finalize_tool_buffer(
                                                        tools_buffer
                                                    )
                                                )

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
                                        if hasattr(error, "type"):
                                            error_details = f"Stream Error ({error.type}): {getattr(error, 'message', str(error))}"

                                        # Create a mock exception for consistent error handling
                                        stream_error = Exception(error_details)
                                        await self.handle_errors(
                                            stream_error, __event_emitter__
                                        )
                                        return (
                                            final_message
                                            + f"\n\nAn error occurred: {error_details}"
                                        )

                                if chunk_count > token_buffer_size:
                                    await __event_emitter__(
                                        {
                                            "type": "chat:message:delta",
                                            "data": {"content": chunk},
                                        }
                                    )
                                    final_message += chunk
                                    chunk = ""
                                    chunk_count = 0

                        # Sende letzten Chunk, falls noch etwas übrig ist
                        if chunk.strip():
                            await __event_emitter__(
                                {
                                    "type": "chat:message:delta",
                                    "data": {"content": chunk},
                                }
                            )
                            final_message += chunk
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
                                return (
                                    final_message
                                    + f"\n\nException of type {type(e).__name__} occurred during tool execution. Reason: {e}"
                                )

                            # Build assistant message with tool_use blocks
                            assistant_content = []
                            
                            # Add preserved thinking blocks first (for multi-turn reasoning)
                            # API will auto-filter & cache only relevant blocks
                            if thinking_blocks:
                                assistant_content.extend(thinking_blocks)
                                if self.valves.DEBUG:
                                    print(f"[DEBUG] Adding {len(thinking_blocks)} thinking block(s) to assistant message for API")
                            
                            if chunk.strip():
                                assistant_content.append(
                                    {"type": "text", "text": chunk}
                                )

                            # Add tool_use blocks to assistant message
                            for tool_call_json in tool_calls:
                                try:
                                    tool_call_data = json.loads(
                                        self._finalize_tool_buffer(tool_call_json)
                                    )
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
                            thinking_blocks = []  # Reset after adding to messages
                            chunk = ""
                            chunk_count = 0
                            current_search_query = ""  # Reset search query for next iteration
                            citation_counter = 0  # Reset citation counter for next iteration
                            continue
                except RateLimitError as e:
                    # Rate limit error (429)
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: Rate limit exceeded. Please wait before making more requests. {e.message}"
                    )
                except AuthenticationError as e:
                    # API key issues (401)
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: API key issues. Reason: {e.message}"
                    )
                except PermissionDeniedError as e:
                    # Permission issues (403)
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: Permission denied. Reason: {e.message}"
                    )
                except NotFoundError as e:
                    # Resource not found (404)
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: Resource not found. Reason: {e.message}"
                    )
                except BadRequestError as e:
                    # Invalid request format (400)
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: Invalid request format. Reason: {e.message}"
                    )

                except UnprocessableEntityError as e:
                    # Unprocessable entity (422)
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: Unprocessable entity. Reason: {e.message}"
                    )
                except InternalServerError as e:
                    # Server errors (500, 529)
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: Internal server error. Reason: {e.message}"
                    )
                except APIConnectionError as e:
                    # Network/connection issues
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: Network/connection issues. Reason: {e.message}"
                    )
                except APIStatusError as e:
                    # Catch any other Anthropic API errors
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: Anthropic API error. Reason: {e.message}"
                    )
                except Exception as e:
                    # Catch all other exceptions
                    await self.handle_errors(e, __event_emitter__)
                    return (
                        final_message
                        + f"\n\nError: {type(e).__name__} occurred. Reason: {e}"
                    )
                finally:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "", "done": True, "hidden": True},
                        }
                    )
        except Exception as e:
            await self.handle_errors(e, __event_emitter__)
        return final_message

    async def handle_errors(self, exception, __event_emitter__):
        # Determine specific error message based on exception type
        if isinstance(exception, RateLimitError):
            error_msg = "Rate limit exceeded. Please wait before making more requests."
            user_msg = "⚠️ Rate limit reached. Please try again in a moment."
        elif isinstance(exception, AuthenticationError):
            error_msg = "Authentication failed. Please check your API key."
            user_msg = (
                "🔑 Invalid API key. Please verify your Anthropic API key is correct."
            )
        elif isinstance(exception, PermissionDeniedError):
            error_msg = (
                "Permission denied. Your API key may not have access to this resource."
            )
            user_msg = "🚫 Access denied. Your API key doesn't have permission for this request."
        elif isinstance(exception, NotFoundError):
            error_msg = (
                "Resource not found. The requested model or endpoint may not exist."
            )
            user_msg = "❓ Resource not found. Please check if the model is available."
        elif isinstance(exception, BadRequestError):
            error_msg = f"Bad request: {str(exception)}"
            user_msg = (
                "📝 Invalid request format. Please check your input and try again."
            )
        elif isinstance(exception, UnprocessableEntityError):
            error_msg = f"Unprocessable entity: {str(exception)}"
            user_msg = "📄 Request format issue. Please check your message structure and try again."
        elif isinstance(exception, InternalServerError):
            error_msg = "Anthropic server error. Please try again later."
            user_msg = (
                "🔧 Server temporarily unavailable. Please try again in a few moments."
            )
        elif isinstance(exception, APIConnectionError):
            error_msg = (
                "Network connection error. Please check your internet connection."
            )
            user_msg = "🌐 Connection error. Please check your network and try again."
        elif isinstance(exception, APIStatusError):
            status_code = getattr(exception, "status_code", "Unknown")
            error_msg = f"API Error ({status_code}): {str(exception)}"
            user_msg = (
                f"⚡ API Error ({status_code}). Please try again or contact support."
            )
        else:
            error_msg = f"Unexpected error: {str(exception)}"
            user_msg = "💥 An unexpected error occurred. Please try again."

        print(f"Exception: {error_msg}")
        # Add request ID if available for debugging
        if isinstance(exception, APIStatusError) and hasattr(exception, "response"):
            try:
                request_id = exception.response.headers.get("request-id")
                if request_id:
                    print(f"Request ID: {request_id}")
            except Exception:
                pass  # Ignore if we can't get request ID

        await __event_emitter__(
            {
                "type": "notification",
                "data": {
                    "type": "error",
                    "content": user_msg,
                },
            }
        )
        import traceback

        tb = traceback.format_exc()
        from datetime import datetime

        await __event_emitter__(
            {
                "type": "source",
                "data": {
                    "source": {"name": "Anthropic Error", "url": None},
                    "document": [tb],
                    "metadata": [
                        {
                            "source": "anthropic api",
                            "type": "error",
                            "date_accessed": datetime.utcnow().isoformat(),
                        }
                    ],
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
                # if tool_name == "memory":
                #     # Handle Memory Tool Calls
                #     await self.handle_memory_call(tool_call_data, __event_emitter__)
                #     continue
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


    # async def handle_memory_call(self, tool_call_data, __event_emitter__):
    #     """
    #     Handle memory tool calls by notifying the user.
    #     """
    #     if self.valves.DEBUG:
    #         print(f"🧠 [DEBUG] Received memory tool call: {tool_call_data}")

        # tool_name = tool_call_data.get("name", "memory")
        # tool_input = tool_call_data.get("input", {})

        # tool_command = tool_input.get("command", "")
        # tool_path = tool_input.get("path", "")

        # if tool_command == "view":
           
        # elif tool_command == "create":
        # elif tool_command == "str_replace":
        # elif tool_command == "insert":
        # elif tool_command == "delete":
        # elif tool_command == "rename":
        

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
        Handles text, images, PDFs, tool_calls, and tool_results according to
        Anthropic API documentation.
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        processed_content = []
        for item in content:
            if item.get("type") == "text":
                processed_content.append({"type": "text", "text": item.get("text", "")})

            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")

                if image_url.startswith("data:image"):
                    # Handle base64 encoded image data
                    try:
                        header, encoded = image_url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]

                        # Validate supported image formats according to Anthropic docs
                        supported_formats = [
                            "image/jpeg",
                            "image/png",
                            "image/gif",
                            "image/webp",
                        ]

                        if mime_type not in supported_formats:
                            if self.valves.DEBUG:
                                print(
                                    f"[DEBUG] Unsupported image mime type: {mime_type}"
                                )
                            processed_content.append(
                                {
                                    "type": "text",
                                    "text": f"[Image type {mime_type} not supported. Supported formats: JPEG, PNG, GIF, WebP]",
                                }
                            )
                            continue

                        # Check image size - API has 32MB request limit, but be conservative
                        # Also check for API limits: 8000x8000 px for single image
                        MAX_IMAGE_SIZE = 25 * 1024 * 1024  # 25 MB (conservative)
                        try:
                            import base64

                            decoded_bytes = base64.b64decode(encoded)
                            if len(decoded_bytes) > MAX_IMAGE_SIZE:
                                if self.valves.DEBUG:
                                    print(
                                        f"[DEBUG] Image too large: {len(decoded_bytes)} bytes"
                                    )
                                processed_content.append(
                                    {
                                        "type": "text",
                                        "text": f"[Image too large for Anthropic API. Max size: 25MB, received: {len(decoded_bytes)//1024//1024}MB]",
                                    }
                                )
                                continue
                        except Exception as decode_ex:
                            if self.valves.DEBUG:
                                print(
                                    f"[DEBUG] Image base64 decode failed: {decode_ex}"
                                )
                            processed_content.append(
                                {
                                    "type": "text",
                                    "text": "[Image data could not be decoded - invalid base64 format]",
                                }
                            )
                            continue

                        processed_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": encoded,
                                },
                            }
                        )

                    except ValueError as e:
                        if self.valves.DEBUG:
                            print(f"[DEBUG] Error parsing image data URL: {e}")
                        processed_content.append(
                            {
                                "type": "text",
                                "text": "[Error processing image - invalid data URL format]",
                            }
                        )
                    except Exception as e:
                        if self.valves.DEBUG:
                            print(f"[DEBUG] Unexpected error processing image: {e}")
                        processed_content.append(
                            {
                                "type": "text",
                                "text": "[Unexpected error processing image]",
                            }
                        )
                else:
                    # For image URLs (not base64), Claude API supports URL references
                    # but we need to validate the URL format
                    if image_url.startswith(("http://", "https://")):
                        processed_content.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": image_url},
                            }
                        )
                    else:
                        processed_content.append(
                            {
                                "type": "text",
                                "text": f"[Invalid image URL format: {image_url}. Only HTTP/HTTPS URLs are supported]",
                            }
                        )

            elif item.get("type") == "tool_calls":
                # Convert OpenWebUI tool_calls to Claude tool_use format
                converted_calls = self._process_tool_calls(item)
                processed_content.extend(converted_calls)

            elif item.get("type") == "tool_results":
                # Convert OpenWebUI tool_results to Claude tool_result format
                converted_results = self._process_tool_results(item)
                processed_content.extend(converted_results)

            # Handle any other content types by converting to text
            else:
                if self.valves.DEBUG:
                    print(
                        f"[DEBUG] Unknown content type: {item.get('type')}, converting to text"
                    )
                # Convert unknown types to text representation
                processed_content.append(
                    {
                        "type": "text",
                        "text": f"[Unsupported content type: {item.get('type')}]",
                    }
                )

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

    def _extract_dynamic_context(self, text: str) -> Optional[str]:
        """
        Extract dynamic context blocks (Memory, RAG, Documents) from system prompt.
        Returns the extracted context or None if no dynamic content found.
        """
        extracted = []
        
        # Extract User Context (Memory System)
        if "User Context:" in text:
            parts = text.split("User Context:", 1)
            if len(parts) > 1:
                # Extract everything after "User Context:" until next section or end
                context_part = parts[1].split("\n\n\n")[0]  # Stop at triple newline (section separator)
                if context_part.strip():
                    extracted.append(f"# User Context\n{context_part.strip()}")
        
        # Extract RAG/Document Context (### Task: ... </user_query>)
        if "### Task:" in text and "</user_query>" in text:
            start_idx = text.find("### Task:")
            end_idx = text.find("</user_query>") + len("</user_query>")
            if start_idx >= 0 and end_idx > start_idx:
                rag_content = text[start_idx:end_idx]
                extracted.append(f"# Knowledge Base Context\n{rag_content}")
        
        return "\n\n".join(extracted) if extracted else None
    
    def _remove_dynamic_context(self, text: str) -> str:
        """
        Remove dynamic context blocks from system prompt text.
        """
        # Remove User Context section
        if "User Context:" in text:
            parts = text.split("User Context:", 1)
            if len(parts) > 1:
                # Keep prefix, remove context part
                remaining = parts[1].split("\n\n\n", 1)
                text = parts[0].rstrip() + ("\n\n\n" + remaining[1] if len(remaining) > 1 else "")
        
        # Remove RAG/Document Context
        if "### Task:" in text and "</user_query>" in text:
            start_idx = text.find("### Task:")
            end_idx = text.find("</user_query>") + len("</user_query>")
            if start_idx >= 0 and end_idx > start_idx:
                text = text[:start_idx] + text[end_idx:]
        
        return text.strip()

    def _finalize_tool_buffer(self, tools_buffer: str) -> str:
        """
        Ensure the tools_buffer is valid JSON. If the buffer is incomplete or invalid
        (common with fine-grained tool streaming), attempt to salvage name/id fields
        and wrap the input portion in an object like {"INVALID_JSON": "<raw>"}
        as recommended by Anthropic docs. This preserves the original malformed
        content for debugging while keeping the outer structure valid JSON.
        """
        tb = tools_buffer or ""
        tb = tb.strip()
        # Quick path: already looks like complete JSON
        if tb.endswith("}"):
            try:
                json.loads(tb)
                return tb
            except Exception:
                # fall through to wrapping logic
                pass

        # Try to preserve prefix fields (type, id, name) before the "\"input\":"
        try:
            if '"input":' in tb:
                prefix, remainder = tb.split('"input":', 1)
                prefix = prefix.rstrip()
                # Build a safe input value wrapping the remainder string
                safe_input = {"INVALID_JSON": remainder}
                safe_input_json = json.dumps(safe_input, ensure_ascii=False)

                # If prefix ends with '{', just append the field
                if prefix.endswith("{"):
                    candidate = prefix + '"input": ' + safe_input_json + "}"
                else:
                    # Ensure there's a separating comma if needed
                    if not prefix.endswith(","):
                        candidate = prefix + ', "input": ' + safe_input_json + "}"
                    else:
                        candidate = prefix + ' "input": ' + safe_input_json + "}"

                # Validate candidate JSON
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    # Fallback to full wrapper below
                    pass

        except Exception:
            # Fall through to full wrapper
            pass

        # As a last resort, return a JSON object containing the invalid raw string
        try:
            return json.dumps({"INVALID_JSON": tb}, ensure_ascii=False)
        except Exception:
            # If even that fails, return a minimal valid JSON
            return json.dumps({"INVALID_JSON": "<unrecoverable>"})

    async def handle_citation(self, event, __event_emitter__, citation_counter=None):
        """
        Handle web search citation events from Anthropic API and emit appropriate source events to OpenWebUI.

        Args:
            event: The citation event from Anthropic (content_block_delta with citations_delta)
            __event_emitter__: OpenWebUI event emitter function
            citation_counter: Optional citation number for inline citations
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
            
            print(
                    f"[DEBUG] Citation data found: {citation}"
                )

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
            cited_text = getattr(citation, "cited_text", "")
            from datetime import datetime

            # CRITICAL: metadata.source is used by OpenWebUI as the grouping ID
            # Must be unique for each citation to prevent Citation merging
            metadata = {
                "source": f"{url}#{citation_counter}",
                "date_accessed": datetime.now().isoformat(),
                "name": f"[{citation_counter}]",
            }
            
            source_data = {
                "source": {
                    "name": title,
                    "url": url,
                    "id": f"{citation_counter}",  # Unique source ID
                    },
                "document": [cited_text],
                "metadata": [metadata],
                
            }

            # Emit the source event
            await __event_emitter__({"type": "source", "data": source_data})

        except Exception as e:
            if self.valves.DEBUG:
                print(f"[DEBUG] Error handling citation: {str(e)}")
            await self.handle_errors(e, __event_emitter__)
