import os
import time
from typing import Any

import openai
from openai import OpenAI

from custom_types import MessageList, SamplerBase, SamplerResponse


class ResponsesSampler(SamplerBase):
    """
    Sample from OpenAI's responses API
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        system_message: str | None = None,
        temperature: float = 0.5,
        # max_tokens: int = 1024,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        enable_web_search: bool = False,
        enable_code_interpreter: bool = False,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        assert os.environ.get("OPENAI_API_KEY"), "Please set OPENAI_API_KEY"
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        # self.max_tokens = max_tokens
        self.image_format = "url"
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.enable_web_search = enable_web_search
        self.enable_code_interpreter = enable_code_interpreter

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        new_image = {
            "type": "input_image",
            "image_url": f"data:image/{format};{encoding},{image}",
        }
        return new_image

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _get_tools(self) -> list[dict[str, Any]]:
        """Get the list of enabled tools."""
        tools = []
        if self.enable_web_search:
            tools.append({"type": "web_search_preview"})
        if self.enable_code_interpreter:
            tools.append({"type": "code_interpreter"})
        return tools

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list
        
        # Prepare tools if any are enabled
        tools = self._get_tools()
        
        trial = 0
        while True:
            try:
                # Prepare the API call parameters
                api_params = {
                    "model": self.model,
                    "input": message_list,
                    "temperature": self.temperature,
                    "tools": tools
                }
                
                # Add reasoning parameters if using reasoning model
                if self.reasoning_model:
                    if self.reasoning_effort:
                        api_params["reasoning"] = {"effort": self.reasoning_effort}
                
                # Make the API call
                response = self.client.responses.create(**api_params)
                
                # Create the sampler response
                sampler_resp = SamplerResponse(
                    response_text=response.output_text,
                    response_metadata={"usage": getattr(response, "usage", {})},
                    actual_queried_message_list=message_list,
                )
                
                # Debug: Print the full response structure
                import json
                print("\n=== FULL RESPONSE STRUCTURE ===")
                print(json.dumps(response.to_dict(), indent=2, default=str))
                print("===============================\n")
                
                # Extract search results and tool calls
                tool_calls = []
                output = {}
                
                # Extract search data from the response
                if hasattr(response, 'output') and isinstance(response.output, list):
                    search_data = []
                    
                    for item in response.output:
                        # Handle search queries
                        if hasattr(item, 'action') and hasattr(item.action, 'query'):
                            search_data.append({
                                'type': 'search_query',
                                'query': item.action.query,
                                'status': getattr(item, 'status', 'completed'),
                                'timestamp': getattr(item, 'created_at', None)
                            })
                        
                        # Handle search results (citations and content in the response)
                        if hasattr(item, 'content') and isinstance(item.content, list):
                            for content_item in item.content:
                                # Handle URL citations (search results)
                                if hasattr(content_item, 'annotations') and content_item.annotations:
                                    for annotation in content_item.annotations:
                                        if hasattr(annotation, 'type') and annotation.type == 'url_citation':
                                            # Extract the citation text if available
                                            snippet = ''
                                            if hasattr(annotation, 'start_index') and hasattr(annotation, 'end_index'):
                                                try:
                                                    snippet = content_item.text[annotation.start_index:annotation.end_index]
                                                except (TypeError, IndexError):
                                                    snippet = content_item.text[:200] + '...' if content_item.text else ''
                                            
                                            search_data.append({
                                                'type': 'search_result',
                                                'title': getattr(annotation, 'title', 'Search Result'),
                                                'url': getattr(annotation, 'url', '#'),
                                                'snippet': snippet,
                                                'timestamp': getattr(item, 'created_at', None)
                                            })
                                
                                # Also include the full text as a search result if it contains URLs
                                elif hasattr(content_item, 'text') and content_item.text and 'http' in content_item.text:
                                    search_data.append({
                                        'type': 'search_result',
                                        'title': 'Search Result',
                                        'url': '#',
                                        'snippet': content_item.text[:500] + '...' if len(content_item.text) > 500 else content_item.text,
                                        'timestamp': getattr(item, 'created_at', None),
                                        'is_raw_text': True
                                    })
                    
                    # Store the raw search data in the output
                    if search_data:
                        output['search_data'] = search_data
                        query_count = len([d for d in search_data if d['type'] == 'search_query'])
                        result_count = len([d for d in search_data if d['type'] == 'search_result'])
                        print(f"Collected {query_count} search queries and {result_count} search results")
                
                # Extract tool calls from the response
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    if isinstance(response.tool_calls, list):
                        tool_calls = response.tool_calls
                        print(f"Found {len(tool_calls)} tool calls in response")
                        for i, tc in enumerate(tool_calls):
                            print(f"Tool call {i}: {tc}")
                
                # Attach the collected data to the response
                setattr(sampler_resp, "tool_calls", tool_calls or None)
                setattr(sampler_resp, "output", output or None)
                return sampler_resp
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                bad_resp = SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
                setattr(bad_resp, "tool_calls", None)
                setattr(bad_resp, "output", None)
                return bad_resp
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
