import json
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

    def _print_response_attrs(self, response):
        """Safely print attributes of a response object"""
        try:
            print(f"Response type: {type(response)}")
            attrs = [a for a in dir(response) if not a.startswith('_')]
            print(f"Available attributes: {', '.join(attrs[:20])}{'...' if len(attrs) > 20 else ''}")
            
            # Try to get the most important attributes directly
            for attr in ['output_text', 'input', 'tool_calls', 'usage', 'model', 'id']:
                if hasattr(response, attr):
                    try:
                        value = getattr(response, attr)
                        if value is not None:
                            print(f"{attr}: {str(value)[:500]}{'...' if len(str(value)) > 500 else ''}")
                    except Exception as e:
                        print(f"Error getting {attr}: {str(e)}")
                        
        except Exception as e:
            print(f"Error in _print_response_attrs: {str(e)}")

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
        max_retries = 3
        while True:
            try:
                # Prepare the API call parameters
                api_params = {
                    "model": self.model,
                    "input": message_list,
                    "tools": tools
                }
                
                # Only include temperature for models that support it
                # o4-mini doesn't support temperature parameter
                if not self.reasoning_model:
                    api_params["temperature"] = self.temperature
                
                # Add reasoning parameters if using reasoning model
                if self.reasoning_model:
                    if self.reasoning_effort:
                        api_params["reasoning"] = {"effort": self.reasoning_effort}
                
                # Debug: Print the API parameters
                print("\n=== API PARAMETERS ===")
                print(json.dumps({
                    k: v for k, v in api_params.items() 
                    if k not in ['input']  # Skip large fields
                }, indent=2))
                print("======================\n")
                
                # Make the API call
                try:
                    # Clean up api_params to remove any None values which might cause issues
                    clean_api_params = {k: v for k, v in api_params.items() if v is not None}
                    
                    # For debugging, print what we're sending
                    print("\n=== SENDING API REQUEST ===")
                    print(f"Model: {clean_api_params.get('model')}")
                    print(f"Input length: {len(clean_api_params.get('input', []))} messages")
                    print(f"Using tools: {bool(clean_api_params.get('tools'))}")
                    if 'temperature' in clean_api_params:
                        print(f"Temperature: {clean_api_params['temperature']}")
                    print("==========================\n")
                    
                    response = self.client.responses.create(**clean_api_params)
                    return self._process_response(response, message_list)
                    
                except Exception as e:
                    print(f"\n=== API CALL FAILED ===")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    
                    # Try to get more detailed error information
                    if hasattr(e, 'response'):
                        try:
                            if hasattr(e.response, 'text'):
                                print(f"Response text: {e.response.text}")
                            if hasattr(e.response, 'status_code'):
                                print(f"Status code: {e.response.status_code}")
                        except Exception as e2:
                            print(f"Error getting response details: {str(e2)}")
                    
                    # Create a minimal error response
                    error_resp = SamplerResponse(
                        response_text=f"Error: {str(e)[:200]}",
                        response_metadata={"usage": None, "error": str(e)},
                        actual_queried_message_list=message_list,
                    )
                    setattr(error_resp, "tool_calls", None)
                    setattr(error_resp, "output", None)
                    return error_resp
                
            except openai.RateLimitError as e:
                # Handle rate limiting with exponential backoff
                exception_backoff = 2 ** trial  # exponential backoff
                print(f"Rate limit exception, waiting {exception_backoff} seconds before retry {trial+1}...")
                time.sleep(exception_backoff)
                trial += 1
                continue
                
            except Exception as e:
                # For other errors, re-raise the exception
                raise

    def _process_response(self, response, message_list):
        """Process the API response and create a SamplerResponse"""
        try:
            # Try to get output_text, handling different response formats
            if hasattr(response, 'output_text'):
                response_text = response.output_text
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.get('content', '')
            else:
                response_text = str(response)
                
            # Create the sampler response
            sampler_resp = SamplerResponse(
                response_text=response_text,
                response_metadata={"usage": getattr(response, "usage", {})},
                actual_queried_message_list=message_list,
            )
                
            # Debug: Print the full response structure
            print("\n=== FULL RESPONSE STRUCTURE ===")
            try:
                # Try to convert response to dict safely
                if hasattr(response, 'to_dict'):
                    try:
                        print(json.dumps(response.to_dict(), indent=2, default=str))
                    except (TypeError, ValueError) as e:
                        print(f"Could not serialize response: {str(e)}")
                        print(f"Response type: {type(response)}")
                        if hasattr(response, 'output_text'):
                            print(f"Output text: {response.output_text[:500]}..." if len(response.output_text) > 500 else f"Output text: {response.output_text}")
                else:
                    print("Response does not have to_dict() method")
                    print(f"Response type: {type(response)}")
                    if hasattr(response, 'output_text'):
                        print(f"Output text: {response.output_text[:500]}..." if len(response.output_text) > 500 else f"Output text: {response.output_text}")
            except Exception as e:
                print(f"Error printing response: {str(e)}")
                print(f"Response type: {type(response)}")
                self._print_response_attrs(response)
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
            
        except Exception as e:
            print(f"Error in _process_response: {str(e)}")
            # Return a minimal response with the error
            error_resp = SamplerResponse(
                response_text=f"Error processing response: {str(e)[:200]}",
                response_metadata={"usage": None, "error": str(e)},
                actual_queried_message_list=message_list,
            )
            setattr(error_resp, "tool_calls", None)
            setattr(error_resp, "output", None)
            return error_resp
