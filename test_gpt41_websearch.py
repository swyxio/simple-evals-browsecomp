#!/usr/bin/env python3
"""
Standalone script to test GPT-4.1 with web search tool.

This script demonstrates how to:
1. Initialize the ResponsesSampler with web search enabled
2. Send a query that would benefit from web search
3. Inspect the raw response to see tool calls and results
4. Print the final response

Make sure to set OPENAI_API_KEY environment variable before running.
"""
import os
import json
from typing import Dict, Any, List, Optional

from sampler.responses_sampler import ResponsesSampler
from custom_types import MessageList


def format_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """Format tool calls for pretty printing."""
    if not tool_calls:
        return "No tool calls"
    
    formatted = []
    for i, call in enumerate(tool_calls, 1):
        formatted.append(f"Tool Call {i}:")
        formatted.append(f"  ID: {call.get('id', 'N/A')}")
        formatted.append(f"  Type: {call.get('type', 'N/A')}")
        
        if 'function' in call:
            func = call['function']
            formatted.append(f"  Function: {func.get('name', 'N/A')}")
            formatted.append(f"  Arguments: {func.get('arguments', 'N/A')}")
        
        if 'output' in call:
            output = call['output']
            if isinstance(output, dict) or isinstance(output, list):
                output = json.dumps(output, indent=2)
            formatted.append(f"  Output: {output}")
    
    return "\n".join(formatted)


def test_gpt41_websearch():
    # Initialize the sampler with web search enabled
    sampler = ResponsesSampler(
        # model="gpt-4.1",
        model="o4-mini-deep-research-2025-06-26",
        enable_web_search=True,
        system_message="You are a helpful assistant with web search capabilities."
    )
    
    # Example query that would benefit from web search
    query = "What are the latest developments in AI as of June 2024?"
    
    print(f"Sending query: {query}")
    
    # Create the message list
    messages = [{"role": "user", "content": query}]
    
    # Get the response
    try:
        response = sampler.client.responses.create(
            model=sampler.model,
            input=messages,
            # temperature=0.7,
            tools=[{"type": "web_search_preview"}]
        )
        
        # Print the raw response for inspection
        print("\n=== RAW RESPONSE ===")
        print(json.dumps(response.to_dict(), indent=2))
        
        # Print the final response text
        print("\n=== FINAL RESPONSE ===")
        print(response.output_text)
        
        # Show tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("\n=== TOOL CALLS ===")
            print(format_tool_calls(response.tool_calls))
        
        # Show usage information
        if hasattr(response, 'usage') and response.usage:
            print("\n=== USAGE ===")
            usage = response.usage
            if hasattr(usage, 'input_tokens'):
                print(f"Input tokens: {usage.input_tokens}")
            if hasattr(usage, 'output_tokens'):
                print(f"Output tokens: {usage.output_tokens}")
            if hasattr(usage, 'total_tokens'):
                print(f"Total tokens: {usage.total_tokens}")
            # Print all available usage attributes for debugging
            print("\n=== USAGE DETAILS ===")
            print(json.dumps(usage.to_dict(), indent=2))
    
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable")
        exit(1)
        
    test_gpt41_websearch()
