import time
from typing import Any

import openai
from openai import OpenAI

from custom_types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        enable_web_search: bool = False,
        enable_code_interpreter: bool = False,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.enable_web_search = enable_web_search
        self.enable_code_interpreter = enable_code_interpreter

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}
        
    # def _get_tools(self) -> list[dict[str, Any]]:
    #     """Get the list of enabled tools."""
    #     tools = []
    #     if self.enable_web_search:
    #         tools.append({"type": "web_search_preview"})
    #     if self.enable_code_interpreter:
    #         tools.append({"type": "code_interpreter"})
    #     return tools

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
            
        # # Prepare tools if any are enabled
        # tools = self._get_tools()
        
        trial = 0
        while True:
            try:
                kwargs = {
                    "model": self.model,
                    "messages": message_list,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                
                # # Only add tools if any are enabled
                # if tools:
                #     kwargs["tools"] = tools
                
                response = self.client.chat.completions.create(**kwargs)
                
                # Handle tool calls if any
                message = response.choices[0].message
                content = message.content or ""
                
                # Process tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Format tool calls into the response
                    tool_call_texts = []
                    for i, tool_call in enumerate(message.tool_calls, 1):
                        if tool_call.type == 'function':
                            func = tool_call.function
                            tool_call_texts.append(
                                f"Tool Call {i}:\n"
                                f"  ID: {tool_call.id}\n"
                                f"  Type: {tool_call.type}\n"
                                f"  Function: {func.name}\n"
                                f"  Arguments: {func.arguments}"
                            )
                    
                    if tool_call_texts:
                        tool_calls_section = "\n\n=== TOOL CALLS ===\n" + "\n\n".join(tool_call_texts)
                        content = f"{content or ''}{tool_calls_section}"
                
                if not content and not hasattr(message, 'tool_calls'):
                    raise ValueError("OpenAI API returned empty response; retrying")
                    
                return SamplerResponse(
                    response_text=content,
                    response_metadata={
                        "usage": response.usage,
                        "tool_calls": getattr(message, 'tool_calls', None)
                    },
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
