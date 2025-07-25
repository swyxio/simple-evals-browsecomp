import time
from typing import Any

import openai
from openai import OpenAI

from custom_types import MessageList, SamplerBase, SamplerResponse


class OChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API for o series models
    """

    def __init__(
        self,
        *,
        reasoning_effort: str | None = None,
        model: str = "o1-mini",
        enable_web_search: bool = False,
        enable_code_interpreter: bool = False,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.image_format = "url"
        self.reasoning_effort = reasoning_effort
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

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def _get_tools(self) -> list[dict[str, Any]]:
        """Get the list of enabled tools."""
        tools = []
        if self.enable_web_search:
            tools.append({"type": "web_search"})
        if self.enable_code_interpreter:
            tools.append({"type": "code_interpreter"})
        return tools

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Prepare tools if any are enabled
        tools = self._get_tools()
        
        trial = 0
        while True:
            try:
                # Fall back to standard chat completions if no tools are enabled
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    reasoning_effort=self.reasoning_effort,
                    tools=tools
                )
                content = response.choices[0].message.content
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="",
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
