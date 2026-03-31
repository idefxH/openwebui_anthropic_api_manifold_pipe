# 🚀 Anthropic API Manifold Pipe for Open WebUI

> **Claude Feature parity for Open WebUI - streaming, prompt caching, tool use, extended thinking, code execution with file support, programmatic tool calling, agent skills, and more.**

---

## 📖 Overview

A comprehensive Anthropic API integration for Open WebUI built on the **Anthropic Python SDK**. I tried to include as much Claude API features as possible while stay compatible with all Openwebui Features. It Enables Claude to orchestrate complex multi-tool workflows
- *"Grab my Jira tasks and send a summary on Slack"* — work token-effician and in a single request with programmatic tool calling, parallel execution.
- *"Check out this finance report - Extract the important information and distill it down into a nice Powerpoint Presentation"* — code_execution with Files API Up- and Download Support and pptx Skill will do the trick.
- *"What's the meaning of life?"* — Extended Thinking can use up to 64k Tokens and web_search can do ask the internet before giving a good answer. Interleaved Thinking allows to think about the path it took even during the response.

### 🎯 Key Highlights

| Feature | Description |
|---------|-------------|
| 🔧 **Programmatic Tool Calling** | Claude orchestrates tools through code execution — multi-tool workflows in one go |
| ⚡ **Parallel Execution** | Independent tools execute simultaneously |
| 💾 **Prompt Caching** | 4-level cache for system prompts, tools, and messages; compatible with RAG & Memory |
| 🧠 **Extended Thinking** | Classic budget, adaptive, and interleaved thinking with live streaming |
| 💻 **Code Execution** | Sandboxed Python with persistent container state, file upload/download |
| 🌐 **Web Search & Fetch** | Dynamic filtering, inline citations, URL content analysis |
| 🔍 **Tool Search** | BM25/Regex deferred loading for large toolsets |
| 🧹 **Context Editing** | Auto-clear old tool results and thinking blocks |
| 📊 **1M Token Context** | Extended context for Opus 4.6, Sonnet 4/4.5 |
| 🎨 **Agent Skills** | pptx, xlsx, docx, pdf & custom skills via Files API |

---

## ✨ Features

### Core

| Feature | Description |
|---------|-------------|
| **Anthropic Python SDK** | Official SDK for streaming and message accumulation |
| **Model Auto-Discovery** | Fetches available models from your API key |
| **Streaming** | Fine-grained tool streaming with eager input streaming (GA) |
| **Tool Call Loop** | Multiple tools per response cycle with configurable limit |
| **Parallel Tool Execution** | Local tools run concurrently |
| **Error Handling** | Retry logic for rate limits and transient errors |
| **Task Support** | Title, tag, and follow-up generation |
| **Notes & Channels** | Full OpenWebUI integration |
| **Token Count** | Toggleable context window progress bar per response |

### Claude API Features

| Feature | Description |
|---------|-------------|
| **Vision** | Image analysis (JPEG, PNG, GIF, WebP) |
| **Native PDF Upload** | Visual PDF analysis bypassing RAG extraction |
| **Citations** | Correctly positioned streaming citations from web search |
| **Extended Thinking** | Classic `budget_tokens` and adaptive thinking (Opus 4.6) |
| **Interleaved Thinking** | Claude thinks between tool calls |
| **Live Thinking Streaming** | Real-time streaming, collapsing into `<details>` blocks |
| **Files API (standalone)** | Native file handling for code execution |
| **Web Search** | `web_search_20260209` with dynamic filtering and location-aware results |
| **Web Fetch** | URL content retrieval and analysis |
| **Code Execution** | Persistent container state, unified display (code + tool calls + output) |
| **Programmatic Tool Calling** | Tools callable from within code execution; multi-tool coordination |
| **Tool Search** | BM25/Regex deferred loading for hundreds/thousands of tools |
| **Context Editing** | Auto-clear tool results & thinking blocks with token-count triggers |
| **Agent Skills** | pptx, xlsx, docx, pdf & custom skills, validated via List Skills API |
| **Data Residency** | `inference_geo` parameter (global or US) |
| **Fast Mode** | Up to 2.5× faster output for Opus 4.6 |
| **1M Token Context** | Opus 4.6, Sonnet 4/4.5 (Tier 4 required) |
| **Effort Parameter** | low / medium / high / max (GA) |
| **Prompt Caching** | 4-level cache: tools, system prompt, messages; RAG/Memory aware |

---

## 🗺️ Roadmap

| Status | Feature | Notes |
|--------|---------|-------|
| 📌 | **[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)** | Server-side context summarization for infinite conversations |

---

## 📦 Installation

### Option 1: Install from OpenWebUI Community (Recommended)

| Component | Link |
|-----------|------|
| **Main Pipe** | [anthropic_pipe](https://openwebui.com/f/podden/anthropic_pipe) |
| **Thinking Toggle** | [anthropic_pipe_thinking_toggle](https://openwebui.com/f/podden/anthropic_pipe_thinking_toggle) |
| **Companion Filter** | [anthropic_manifold_companion](https://openwebui.com/f/podden/anthropic_manifold_companion) |

### Option 2: Manual Installation

1. **Admin Settings** → **Functions** → **"+ New Function"**
2. Paste the source code, set name/ID/description
3. Repeat for each toggle filter

### Configuration

1. **Set API Key** in the pipe's Valves
2. **Configure Models** (Admin Settings → Models):
   - Activate Thinking and Companion Filter for each Claude model or globally
   - Activate web_search and code_interpreter capabilities
   - Optional: Add usage to see token consumption
   - Set **Function Calling** to `Native` in Advanced Parameters
2a.**Set Valves and UserValves** Experiment with the Settings to find values to your liking
3. **Start chatting!**
---

## 🔧 Configuration
### Valves (Global / Admin Settings)
| Valve | Default | Description |
|-------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key (required) |
| `ENABLE_FAST_MODE` | `false` | Fast Mode for Opus 4.6 (up to 2.5× faster, higher cost) |
| `ENABLE_1M_CONTEXT` | `false` | 1M token context window (Tier 4 required) |
| `ENABLE_INTERLEAVED_THINKING` | `true` | Claude thinks between tool calls |
| `WEB_SEARCH` | `true` | Enable web search tool |
| `WEB_FETCH` | `true` | Enable web fetch tool (URL content retrieval) |
| `MAX_TOOL_CALLS` | `15` | Max tool execution loops per request (1–50) |
| `MAX_RETRIES` | `3` | Max retries for failed requests (0–50) |
| `CACHE_CONTROL` | `cache tools array, system prompt and messages` | Prompt caching scope (see below) |
| `ENABLE_PROGRAMMATIC_TOOL_CALLING` | `false` | Tools callable from within code execution |
| `ENABLE_TOOL_SEARCH` | `false` | BM25/Regex tool search for large toolsets |
| `TOOL_SEARCH_TYPE` | `bm25` | `regex` or `bm25` |
| `TOOL_SEARCH_MAX_DESCRIPTION_LENGTH` | `100` | Tools with longer JSON defs are deferred |
| `TOOL_SEARCH_EXCLUDE_TOOLS` | `[web_search, web_fetch, ...]` | Always-loaded tools |
| `CONTEXT_EDITING_STRATEGY` | `none` | `none` / `clear_tool_results` / `clear_thinking` / `clear_both` |
| `CONTEXT_EDITING_THINKING_KEEP` | `5` | Thinking blocks to keep |
| `CONTEXT_EDITING_TOOL_TRIGGER` | `50000` | Token threshold for clearing tool results |
| `CONTEXT_EDITING_TOOL_KEEP` | `5` | Recent tool results to preserve |
| `CONTEXT_EDITING_TOOL_CLEAR_AT_LEAST` | `10000` | Minimum tokens to clear |
| `CONTEXT_EDITING_TOOL_CLEAR_TOOL_INPUT` | `false` | Also clear tool input parameters |
| `DATA_RESIDENCY` | `global` | `global` or `us` (1.1× cost for US) |
| `WEB_SEARCH_USER_*` | — | Default location for web searches (city, region, country, timezone) |

#### Cache Control Options
| Option | Description |
|--------|-------------|
| `cache disabled` | No caching |
| `cache tools array only` | Cache tool definitions |
| `cache tools array and system prompt` | Cache tools + system prompt |
| `cache tools array, system prompt and messages` | Full caching (recommended) |

> 💡 **RAG & Memory**: The pipe is aware of your settings and your intention, for example if you're attaching a PDF document with full context mode with NATIVE_PDF_UPLOAD active, it removed the RAG Promt and Sources entirely. If there's additional knowledge added, it strips the PDF RAG sources from RAG and moves the caching point to the previous messages as the last message is now always changing. It also extracts Memories from the System Promt and add them to the last user message when the Memory System is active to avoid cache misses. If you're encountering problems, feel free to open an issue!

### UserValves (Per-User Settings)
| Valve | Default | Range | Description |
|-------|---------|-------|-------------|
| `ENABLE_THINKING` | `false` | — | Enable Extended Thinking |
| `THINKING_BUDGET_TOKENS` | `8192` | 1024–64000 | Token budget for thinking |
| `EFFORT` | `high` | low/medium/high/max | Effort level (also controllable via OpenWebUI's `reasoning_effort`) |
| `USE_PDF_NATIVE_UPLOAD` | `true` | — | Visual PDF analysis instead of RAG extraction |
| `SHOW_TOKEN_COUNT` | `false` | — | Show context window progress bar |
| `WEB_SEARCH_MAX_USES` | `5` | 1–20 | Max web searches per turn |
| `WEB_FETCH_MAX_USES` | `5` | 1–20 | Max web fetch requests per turn |
| `WEB_SEARCH_USER_*` | — | — | Override global location settings |
| `SKILLS` | `[]` | — | Skills to activate (e.g., `pptx`, `xlsx`, `docx`, `pdf`, or custom IDs) |
| `DEBUG_MODE` | `false` | — | Logs some internal and external parameters as citation to send me for debugging ;) |

### Toggle Filters & Companion
| Filter | Purpose |
|--------|---------|
| **Thinking Toggle** | 🧠 Enable thinking for the next message |
| **Companion Filter** | 🔀 Intercepts OpenWebUI's built-in `web_search` / `code_interpreter` buttons and routes them to native Anthropic tools |

---

## 📝 Changelog

### v0.8.1
- Added experimental Files API Support for uploading files to the Container. Feedback welcome!
- Added a Valve to control wheter Opus/Sonnet 4.6 should use the new dynamic web_fetching and web_searching (At least I have issues with that)

### v0.8.0
- Major streaming refactor: rebuilt on Anthropic SDK message accumulation
- Programmatic Tool Calling — Claude orchestrates tools from within code execution
- Web Fetch tool — Claude can fetch and analyze URL content
- Fine-grained tool streaming with eager input streaming (GA)
- Unified code execution display (code + tool calls + output in one block)
- Updated web_search to `web_search_20260209` with dynamic filtering
- Citations now correctly appear after cited text
- Tool search status shows actual search query
- Model capabilities updated for Sonnet 4.5/4.6 and Opus 4.6
- Stop reason debug logging for tool loop diagnostics

### v0.7.1
- Removed deprecated Sonnet 3.7 and Haiku 3 models

### v0.7.0
- Sonnet 4.6 model support
- Fast Mode for Opus 4.6 (`speed: "fast"`)
- Web fetch tool (URL content retrieval)
- Memory tool integration with OpenWebUI memory system
- Fixed task model bug (`_run_task_model_request()` extra argument)

### v0.6.3
- Opus 4.6 model support
- Effort: `max` support
- Data Residency (`inference_geo`) support
- Messages for stop_reason (refusal, stop_sequence, context exceeded)
- `ENABLE_INTERLEAVED_THINKING` valve
- Homogenized thinking and tool call/result streaming to match built-in UX

### v0.6.2
- Reordered payload for better caching

### v0.6.1
- Full Skills support (pptx, xlsx, docx, pdf, custom) with API validation and caching

### v0.6.0
- Live thinking streaming with collapsible blocks
- Companion Filter for routing OpenWebUI web_search/code_interpreter to Anthropic tools
- Files API upload for code execution file access
- Built-in OpenWebUI tools support (0.7.x)
- Native PDF markers for multi-turn file persistence
- Container ID persistence for code execution state
- Fixed RAG + Native PDF Upload interaction

<details>
<summary><b>v0.5.x</b> (click to expand)</summary>

### v0.5.12
- Thinking is now streamed in the UI and folded when the thought process has ended

### v0.5.11
- Compatibility with built-in tools from OpenWebUI 0.7.x

### v0.5.10
- Pre-compiled regex patterns at module level
- Debug logging guards for expensive JSON serialization
- Comprehensive docstring and section comments

### v0.5.9
- Native PDF upload via `USE_PDF_NATIVE_UPLOAD` UserValve

### v0.5.8
- Fixed UnboundLocalError for `total_usage` variable
- Code execution added to `TOOL_SEARCH_EXCLUDE_TOOLS`

### v0.5.7
- Tool search exclusion valve
- Web Search Toggle overrides `WEB_SEARCH` valve
- Fixed Tool Search return

### v0.5.6
- Context Editing (clear_tool_results, clear_thinking)
- Tool Search (BM25/Regex) with deferred loading
- Status events for context clearing

### v0.5.4
- Fixed Message Caching Problems when using RAG or Memories

### v0.5.3
- Effort Levels (`low`, `medium`, `high`)
- Opus 4.5 support
- UserValves for per-user settings

### v0.5.2
- Fixed usage statistics accumulation for multi-step tool calls

### v0.5.1
- Fixed caching in tool execution loops

### v0.5.0
- **CRITICAL FIX**: Eliminated cross-talk between concurrent users/requests

</details>

<details>
<summary><b>v0.4.x</b> (click to expand)</summary>

#### v0.4.9
- Performance optimization: Moved local imports to top level
- Fixed fallback logic for model fetching

#### v0.4.8
- Added configurable `MAX_TOOL_CALLS` valve (default: 15)
- Immediate tool execution status events
- Proactive warnings when approaching tool call limit

#### v0.4.7
- Fixed potential data leakage between concurrent users

#### v0.4.6
- Tool results now display input parameters at the top

#### v0.4.5
- Added status events for local tool execution
- Better UX for long-running tools

#### v0.4.4
- Tool calls now execute in parallel
- Fixed server tool identification

#### v0.4.3
- Fixed compatibility with OpenWebUI "Chat with Notes"

#### v0.4.2
- Fixed NoneType error in OpenWebUI Channels

#### v0.4.1
- Added token count display valve
- Auto-enable native function calling

#### v0.4.0
- Added Task Support (titles, tags, follow-ups)
- Fixed server + local tool use conflict

</details>

<details>
<summary><b>v0.3.x</b> (click to expand)</summary>

#### v0.3.9
- Added fine-grained cache control valve (4 levels)

#### v0.3.8
- Removed MAX_OUTPUT_TOKENS valve
- Reworked caching with OpenWebUI Memory System
- Added retry logic for transient errors

#### v0.3.7
- Fixed Extended Thinking compatibility with Tool Use

#### v0.3.6
- Added Claude 4.5 Haiku Model

#### v0.3.5
- Fixed last chunk not sent bug
- Added correct citation handling for Web Search

#### v0.3.4
- Added Claude 4.5 Sonnet support
- Added OpenWebUI token usage compatibility
- Added duplicate tool name validation

#### v0.3.3 - v0.3.1
- Various bug fixes

#### v0.3.0 (September 2025)
- Added Vision support
- Added Extended Thinking filter
- Added Web Search enforcement toggle
- Added Anthropic Code Execution Tool
- Improved cache control with dynamic Memory/RAG detection
- Added 1M context beta header for Sonnet 4

</details>

<details>
<summary><b>v0.2.0</b> (August 2025)</summary>

- Fixed caching by moving Memories to Messages
- Cache usage statistics display
- Fixed last chunk not showing in frontend
- Implemented Web Search valves and error handling
- Added Cache_Control for System_Prompt, Tools, and Messages

</details>

---

## 🤝 Contributing

Bug reports and feature requests are welcome! Feel free to [open an issue](https://github.com/Podden/openwebui_anthropic_api_manifold_pipe/issues) if you encounter any problems.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built for [Open WebUI](https://github.com/open-webui/open-webui)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- Thanks Balaxxe and nbellochi for their original Pipe
---
