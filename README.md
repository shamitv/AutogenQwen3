# QwenOpenAIChatCompletionClient

A Python client library for interacting with Qwen3 and DeepSeek models via OpenAI-compatible API, built on top of AutoGen. This client provides structured output support, function calling, and comprehensive model configuration for building agentic AI applications.

## Installation

Install from PyPI:

```bash
pip install qwen3-autogen-client
```

## Attribution

This project is based on the excellent work from:
- **Author**: [Data Leads Future](https://www.dataleadsfuture.com/build-autogen-agents-with-qwen3-structured-output-thinking-mode/)
- **GitHub Repository**: [Agentic AI Playground - AutoGen-Qwen3 Integration](https://github.com/qtalen/agentic-ai-playground/tree/main/03_Master_AutoGen-Qwen3_Integration)

## Features

- **Multi-Model Support**: Qwen3, Qwen2.5, and DeepSeek models
- **Structured Output**: Pydantic model-based JSON schema enforcement for reliable AI responses
- **Function Calling**: Full support for tool usage and function calling capabilities
- **Async Support**: Both streaming and non-streaming async operations
- **Token Management**: Intelligent token counting and remaining token calculation
- **Comprehensive Logging**: Built-in logging for debugging and monitoring
- **AutoGen Integration**: Seamless integration with AutoGen's agent framework

## Supported Models

### Qwen3 Models
- `Qwen3-32B` (32K context)
- `Qwen3-14B` (32K context)
- `Qwen3-8B` (32K context)
- `Qwen3-4B` (32K context)
- `Qwen3-1.7B` (32K context)
- `qwen-max` (32K context)
- `qwen-max-latest` (128K context)
- `qwen-plus` (128K context)
- `qwen-plus-latest` (128K context)
- `qwen-turbo` (1M context)
- `qwen-turbo-latest` (1M context)
- `qwen3-235b-a22b` (128K context)
- `qwen3-30b-a3b` (128K context)

### Qwen2.5 Models
- `Qwen2.5-Omni-7B` (32K context, vision)
- `Qwen2.5-Omni-3B` (32K context, vision)
- `Qwen2.5-VL-32B-Instruct` (32K context, vision)
- `Qwen2.5-VL-7B-Instruct` (32K context, vision)

### DeepSeek Models
- `deepseek-chat` (64K context, function calling supported)
- `deepseek-reasoner` (64K context, reasoning mode)

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from qwen3_autogen_client import QwenOpenAIChatCompletionClient
from autogen_core.models import UserMessage

# Initialize the client
client = QwenOpenAIChatCompletionClient(
    model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your_api_key_here"
)

# Create a simple completion
messages = [UserMessage(content="Hello, how are you?")]
result = await client.create(messages=messages)
print(result.content)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel
from typing import List

class TaskList(BaseModel):
    tasks: List[str]
    priority: str

# Get structured output
messages = [UserMessage(content="Create a task list for planning a vacation")]
result = await client.create(
    messages=messages,
    json_output=TaskList
)
# Result will be automatically formatted according to TaskList schema
```

### Streaming Response

```python
async for chunk in client.create_stream(messages=messages):
    if isinstance(chunk, str):
        print(chunk, end="")
    else:
        # Final result
        print(f"\nFinal result: {chunk}")
```

### Function Calling

```python
from autogen_core.tools import Tool

# Define a tool
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

tool = Tool(get_weather, name="get_weather", description="Get weather for a location")

# Use with function calling
result = await client.create(
    messages=[UserMessage(content="What's the weather in Tokyo?")],
    tools=[tool]
)
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### Direct Configuration

```python
client = QwenOpenAIChatCompletionClient(
    model="qwen-max-latest",
    base_url="https://your-api-endpoint.com",
    api_key="your_api_key",
    # Additional OpenAI client parameters
    timeout=30.0,
    max_retries=3
)
```

## API Reference

### QwenOpenAIChatCompletionClient

#### `__init__(model: str, base_url: str, **kwargs)`

Initialize the client.

**Parameters:**
- `model` (str): The model name to use (required)
- `base_url` (str): Base URL for the API endpoint (required)
- `**kwargs`: Additional parameters passed to the underlying OpenAI client

**Example:**
```python
client = QwenOpenAIChatCompletionClient(
    model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your_key"
)
```

#### `async create(messages, *, tools=[], json_output=None, extra_create_args={}, cancellation_token=None) -> CreateResult`

Create a completion.

**Parameters:**
- `messages`: Sequence of LLMMessage objects
- `tools`: Optional sequence of Tool or ToolSchema objects
- `json_output`: Optional bool or Pydantic BaseModel class for structured output
- `extra_create_args`: Additional arguments for the API call
- `cancellation_token`: Optional cancellation token

#### `async create_stream(...) -> AsyncGenerator[Union[str, CreateResult], None]`

Create a streaming completion with the same parameters as `create()`.

#### `remaining_tokens(messages, *, tools=[]) -> int`

Calculate remaining tokens available for the conversation.

## Model Capabilities

| Model | Function Calling | JSON Output | Vision | Context Window |
|-------|-----------------|-------------|---------|----------------|
| qwen-max | ✅ | ✅ | ❌ | 32K |
| qwen-max-latest | ✅ | ✅ | ❌ | 128K |
| qwen-plus | ✅ | ✅ | ❌ | 128K |
| qwen-turbo | ✅ | ✅ | ❌ | 1M |
| deepseek-chat | ✅ | ✅ | ❌ | 64K |
| deepseek-reasoner | ❌ | ❌ | ❌ | 64K |

## License

This project is licensed under the MIT License - see the LICENSE file for details.
