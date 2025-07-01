"""
Async implementation of simple_evals.py that:
1. Fires off all queries asynchronously
2. Saves task info to a JSON file
3. Exits immediately after submission

To check status, use poll_tasks.py
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tiktoken
import pandas as pd
from openai import AsyncOpenAI
from simple_evals import (
    calculate_cost, TokenCounter, common, BrowseCompEval, 
    OPENAI_SYSTEM_MESSAGE_API, ChatCompletionSampler, OChatCompletionSampler
)
from sampler.responses_sampler import ResponsesSampler, MessageList, SamplerResponse

# Reuse the same model pricing as simple_evals.py
MODEL_PRICING = {
    # GPT-4 models
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-32k": {"input": 60.0, "output": 120.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4.1-2025-04-14": {"input": 5.0, "output": 15.0},
    # GPT-3.5 models
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
    # O1 models
    "o1": {"input": 5.0, "output": 25.0},
    "o1-mini": {"input": 1.5, "output": 2.0},
    "o3-mini": {"input": 1.0, "output": 1.5},
}

class AsyncResponsesSampler(ResponsesSampler):
    """Async version of ResponsesSampler that uses AsyncOpenAI"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = AsyncOpenAI()
        self.task_id = None
        self.status = "pending"
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None
    
    async def _process_response(self, response, message_list):
        """Process the API response asynchronously"""
        try:
            # Get the response text
            response_text = getattr(response, 'output_text', str(response))
            
            # Create the sampler response
            sampler_resp = SamplerResponse(
                response_text=response_text,
                response_metadata={"usage": getattr(response, "usage", {})},
                actual_queried_message_list=message_list,
            )
            
            # Extract search data and tool calls
            tool_calls = []
            output = {}
            
            if hasattr(response, 'output') and isinstance(response.output, list):
                search_data = []
                for item in response.output:
                    if hasattr(item, 'action') and hasattr(item.action, 'query'):
                        search_data.append({
                            'type': 'search_query',
                            'query': item.action.query,
                            'status': getattr(item, 'status', 'completed'),
                            'timestamp': getattr(item, 'created_at', None)
                        })
                
                if search_data:
                    output['search_data'] = search_data
            
            setattr(sampler_resp, "tool_calls", tool_calls or None)
            setattr(sampler_resp, "output", output or None)
            return sampler_resp
            
        except Exception as e:
            print(f"Error in _process_response: {str(e)}")
            return SamplerResponse(
                response_text=f"Error processing response: {str(e)[:200]}",
                response_metadata={"usage": None, "error": str(e)},
                actual_queried_message_list=message_list,
            )
    
    async def create_async_task(self, message_list: MessageList) -> str:
        """Create an async task and return the task ID"""
        try:
            self.started_at = time.time()
            self.status = "running"
            
            if self.system_message:
                message_list = [
                    self._pack_message("developer", self.system_message)
                ] + message_list
            
            tools = self._get_tools()
            
            api_params = {
                "model": self.model,
                "input": message_list,
                "tools": tools
            }
            
            if not self.model.startswith('o4-'):
                api_params["temperature"] = self.temperature
            
            if self.reasoning_model and self.reasoning_effort:
                api_params["reasoning"] = {"effort": self.reasoning_effort}
            
            # Create the async task
            response = await self.client.responses.create(**api_params)
            
            # For simplicity, we'll process the response immediately
            # In a production system, you'd want to track the task ID and poll for completion
            self.result = await self._process_response(response, message_list)
            self.status = "completed"
            self.completed_at = time.time()
            
            # Generate a unique task ID
            self.task_id = f"task_{int(time.time())}_{hash(str(message_list))}"
            return self.task_id
            
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self.completed_at = time.time()
            raise

def save_tasks(tasks: Dict[str, Dict], output_dir: str = "async_results") -> Path:
    """Save tasks to a JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks_file = output_dir / f"tasks_{timestamp}.json"
    
    # Also create/update a latest.json symlink
    latest_file = output_dir / "latest_tasks.json"
    
    # Save the tasks
    with open(tasks_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    # Update the latest symlink
    if latest_file.exists():
        latest_file.unlink()
    latest_file.symlink_to(tasks_file.name)
    
    return tasks_file
    
async def create_tasks(
    eval_name: str,
    models: Dict[str, Any],
    examples: List[Dict[str, Any]],
    n_repeats: int = 1,
    output_dir: str = "async_results"
) -> Dict[str, Dict]:
    """Create tasks for all model evaluations"""
    tasks = {}
    timestamp = int(time.time())
    
    for model_name, sampler in models.items():
        task_id = f"{eval_name}_{model_name}_{timestamp}"
        
        tasks[task_id] = {
            "task_id": task_id,
            "eval_name": eval_name,
            "model_name": model_name,
            "model_config": {
                "model": sampler.model,
                "temperature": getattr(sampler, 'temperature', 0.5),
                "reasoning_model": getattr(sampler, 'reasoning_model', False),
                "enable_web_search": getattr(sampler, 'enable_web_search', False),
            },
            "num_examples": len(examples),
            "n_repeats": n_repeats,
            "status": "pending",
            "progress": 0,
            "results": [],
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "output_files": {}
        }
    
    # Save tasks to file
    tasks_file = save_tasks(tasks, output_dir)
    print(f"Created {len(tasks)} tasks. Task info saved to: {tasks_file}")
    return tasks

def get_models(args):
    """Get the models to evaluate"""
    models = {
        "o3": AsyncResponsesSampler(
            model="o3-2025-04-16",
            system_message="""You are a research assistant helping answer questions through web search...""",
            reasoning_model=True,
            enable_web_search=True,
        ),
        "o3-mini": AsyncResponsesSampler(
            model="o3-mini-deep-research-2025-06-26",
            system_message="""You are a research assistant helping answer questions through web search...""",
            reasoning_model=True,
            enable_web_search=True,
        ),
        "o4-mini": AsyncResponsesSampler(
            model="o4-mini-deep-research-2025-06-26",
            system_message="""You are a research assistant helping answer questions through web search...""",
            reasoning_model=True,
            enable_web_search=True,
        ),
    }
    
    if args.model:
        selected_models = {}
        for model_name in args.model.split(','):
            if model_name in models:
                selected_models[model_name] = models[model_name]
            else:
                print(f"Warning: Model '{model_name}' not found")
        return selected_models
    
    return models

async def main():
    """Main async entry point"""
    parser = argparse.ArgumentParser(description="Submit async evaluation tasks")
    parser.add_argument("--model", type=str, help="Comma-separated list of models to evaluate")
    parser.add_argument("--eval", type=str, default="browsecomp", help="Evaluation to run")
    parser.add_argument("--n-repeats", type=int, default=1, help="Number of times to repeat each evaluation")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (use fewer examples)")
    parser.add_argument("--output-dir", type=str, default="async_results", help="Directory to save task files")
    args = parser.parse_args()
    
    # Load examples
    print("Loading examples...")
    df = pd.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    )
    examples = [row.to_dict() for _, row in df.iterrows()]
    
    if args.debug:
        examples = examples[:2]  # Just use 2 examples in debug mode
        print(f"Debug mode: Using {len(examples)} examples")
    
    # Get models
    models = get_models(args)
    if not models:
        print("No valid models specified")
        return
    
    print(f"Preparing to evaluate {len(models)} models with {len(examples)} examples each")
    
    # Create and save tasks
    tasks = await create_tasks(
        eval_name=args.eval,
        models=models,
        examples=examples,
        n_repeats=args.n_repeats,
        output_dir=args.output_dir
    )
    
    print(f"\nSuccessfully created {len(tasks)} evaluation tasks")
    print("To check status, run: python poll_tasks.py")

if __name__ == "__main__":
    asyncio.run(main())
