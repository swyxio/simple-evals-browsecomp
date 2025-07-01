import argparse
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
import tiktoken

# Pricing per 1M tokens (as of 2024-06-29)
MODEL_PRICING = {
    # GPT-4 models
    "gpt-4": {"input": 30.0, "output": 60.0},  # $30/M input, $60/M output
    "gpt-4-32k": {"input": 60.0, "output": 120.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4.1-2025-04-14": {"input": 5.0, "output": 15.0},  # Assuming same as gpt-4o
    
    # GPT-3.5 models
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
    
    # O1 models (example pricing, adjust as needed)
    "o1": {"input": 5.0, "output": 25.0},
    "o1-mini": {"input": 1.5, "output": 2.0},
    "o3-mini": {"input": 1.0, "output": 1.5},
}

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on token usage and model pricing."""
    model_key = next((k for k in MODEL_PRICING if k in model_name.lower()), "gpt-4")
    pricing = MODEL_PRICING[model_key]
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost

class TokenCounter:
    """Helper class to count tokens and track costs."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Default for most models
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def add_input(self, text: str) -> int:
        """Add input text and return token count."""
        count = self.count_tokens(text)
        self.input_tokens += count
        return count
    
    def add_output(self, text: str) -> int:
        """Add output text and return token count."""
        count = self.count_tokens(text)
        self.output_tokens += count
        return count
    
    def get_usage(self) -> Dict[str, Any]:
        """Get token usage and cost information."""
        cost = calculate_cost(self.model_name, self.input_tokens, self.output_tokens)
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_cost": cost,
            "model": self.model_name,
        }

import pandas as pd

import common
from browsecomp_eval import BrowseCompEval
from sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.responses_sampler import ResponsesSampler


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Comma-separated list of models to evaluate (default: all)",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="browsecomp",
        help="Comma-separated list of evaluations to run",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of times to repeat each evaluation (default: 1)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads to use for parallel evaluation (default: 120)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (fewer examples, more verbose output)",
    )
    parser.add_argument(
        "--track-tokens",
        action="store_true",
        help="Track token usage and estimate costs",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: 5 for debug, all for non-debug)",
    )

    args = parser.parse_args()

    models = {
        # # Reasoning Models
        # "o3": ResponsesSampler(
        #     model="o3-2025-04-16",
        #     reasoning_model=True,
        # ),
#         "o3": ResponsesSampler(
#             model="o3-2025-04-16",
#             system_message="""You are a research assistant helping answer questions through web search. Follow these guidelines:

# 1. Break down complex questions into focused, atomic search queries (10-12 words max)
# 2. Use clear, simple keywords instead of full sentences
# 3. Avoid multiple concepts in a single query
# 4. Use specific, concrete terms over vague ones
# 5. Start with broad searches and narrow down based on results
# 6. Use quotes for exact phrases when needed
# 7. Include date ranges when relevant
# 8. Use site: operator for known reliable sources
# 9. Search for specific facts one at a time
# 10. Verify information from multiple sources

# For example, instead of:
# "two narratives origins potato human awareness stone tools rituals divine medicinal remedy poverty rural 1940s study body composition controlled intake 12 weeks documented use first person article"

# Search for:
# - "origin of potato cultivation"
# - "potato history archaeological evidence"
# - "medicinal uses of potatoes history"
# - "potato nutritional studies 20th century"

# Be precise, focused, and methodical in your search strategy.""",
#             reasoning_model=True,
#             enable_web_search=True,  # Enable web search for this model
#         ),
        "o3-dr": ResponsesSampler(
            model="o3-deep-research-2025-06-26",
            system_message="""You are a research assistant helping answer questions through web search. Follow these guidelines:

1. Break down complex questions into focused, atomic search queries (5-10 words max)
2. Use clear, simple keywords instead of full sentences
3. Avoid multiple concepts in a single query
4. Use specific, concrete terms over vague ones
5. Start with broad theories for what you are looking for and only slowly narrow down based on results
6. Use quotes for exact phrases only when needed
7. Include date ranges when relevant
8. Use site: operator for known reliable sources
9. Search for specific facts one at a time

For example, instead of:
"two narratives origins potato human awareness stone tools rituals divine medicinal remedy poverty rural 1940s study body composition controlled intake 12 weeks documented use first person article"

Search for:
- "origin of potato cultivation"
- "potato history archaeological evidence"
- "medicinal uses of potatoes history"
- "potato nutritional studies 20th century"

Be precise, focused, and methodical in your search strategy.""",
            reasoning_model=True,
            enable_web_search=True,  # Enable web search for this model
        ),
        # # "o3-temp-1": ResponsesSampler(
        # #     model="o3-2025-04-16",
        # #     reasoning_model=True,
        # #     temperature=1.0,
        # # ),
        # "o3_high": ResponsesSampler(
        #     model="o3-2025-04-16",
        #     reasoning_model=True,
        #     reasoning_effort="high",
        # ),
        # "o3_low": ResponsesSampler(
        #     model="o3-2025-04-16",
        #     reasoning_model=True,
        #     reasoning_effort="low",
        # ),
#         # # # Default == Medium
#         "o4-mini": ResponsesSampler(
#             model="o4-mini-2025-04-16",
#             system_message="""You are a research assistant helping answer questions through web search. Follow these guidelines:

# 1. Break down complex questions into focused, atomic search queries (10-12 words max)
# 2. Use clear, simple keywords instead of full sentences
# 3. Avoid multiple concepts in a single query
# 4. Use specific, concrete terms over vague ones
# 5. Start with broad searches and narrow down based on results
# 6. Use quotes for exact phrases when needed
# 7. Include date ranges when relevant
# 8. Use site: operator for known reliable sources
# 9. Search for specific facts one at a time
# 10. Verify information from multiple sources

# For example, instead of:
# "two narratives origins potato human awareness stone tools rituals divine medicinal remedy poverty rural 1940s study body composition controlled intake 12 weeks documented use first person article"

# Search for:
# - "origin of potato cultivation"
# - "potato history archaeological evidence"
# - "medicinal uses of potatoes history"
# - "potato nutritional studies 20th century"

# Be precise, focused, and methodical in your search strategy.""",
#             reasoning_model=True,
#             enable_web_search=True,  # Enable web search for this model
#         ),
#         # # # Default == Medium
#         "o4-mini": ResponsesSampler(
#             model="o4-mini-deep-research-2025-06-26",
#             system_message="""You are a research assistant helping answer questions through web search. Follow these guidelines:

# 1. Break down complex questions into focused, atomic search queries (10-12 words max)
# 2. Use clear, simple keywords instead of full sentences
# 3. Avoid multiple concepts in a single query
# 4. Use specific, concrete terms over vague ones
# 5. Start with broad searches and narrow down based on results
# 6. Use quotes for exact phrases when needed
# 7. Include date ranges when relevant
# 8. Use site: operator for known reliable sources
# 9. Search for specific facts one at a time
# 10. Verify information from multiple sources

# For example, instead of:
# "two narratives origins potato human awareness stone tools rituals divine medicinal remedy poverty rural 1940s study body composition controlled intake 12 weeks documented use first person article"

# Search for:
# - "origin of potato cultivation"
# - "potato history archaeological evidence"
# - "medicinal uses of potatoes history"
# - "potato nutritional studies 20th century"

# Be precise, focused, and methodical in your search strategy.""",
#             reasoning_model=True,
#             enable_web_search=True,  # Enable web search for this model
#         ),
        # "o4-mini_high": ResponsesSampler(
        #     model="o4-mini-2025-04-16",
        #     reasoning_model=True,
        #     reasoning_effort="high",
        # ),
        # "o4-mini_low": ResponsesSampler(
        #     model="o4-mini-2025-04-16",
        #     reasoning_model=True,
        #     reasoning_effort="low",
        # ),
        # "o1-pro": ResponsesSampler(
        #     model="o1-pro",
        #     reasoning_model=True,
        # ),
        # "o1": OChatCompletionSampler(
        #     model="o1",
        # ),
        # "o1_high": OChatCompletionSampler(
        #     model="o1",
        #     reasoning_effort="high",
        # ),
        # "o1_low": OChatCompletionSampler(
        #     model="o1",
        #     reasoning_effort="low",
        # ),
        # "o1-preview": OChatCompletionSampler(
        #     model="o1-preview",
        # ),
        # "o1-mini": OChatCompletionSampler(
        #     model="o1-mini",
        # ),
        # # Default == Medium
        # "o3-mini": OChatCompletionSampler(
        #     model="o3-mini",
        # ),
        # "o3-mini_high": OChatCompletionSampler(
        #     model="o3-mini",
        #     reasoning_effort="high",
        # ),
        # "o3-mini_low": OChatCompletionSampler(
        #     model="o3-mini",
        #     reasoning_effort="low",
        # ),
        # # GPT-4.1 models
        # "gpt-4.1": ResponsesSampler(
        #     model="gpt-4.1-2025-04-14",
        #     enable_web_search=True,
        #     system_message="You are a helpful assistant answering trivia questions you don't have the direct answers to in your knowledge. Always use the web search tool you have, but think step by step and break down the problem so that you dont make too complex of a search. search in stages and gather information as you go.",
        #     # max_tokens=2048,
        # ),
        # "gpt-4.1": ChatCompletionSampler(
        #     model="gpt-4.1-2025-04-14",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        #     enable_web_search=True,
        # ),
        # "gpt-4.1-temp-1": ChatCompletionSampler(
        #     model="gpt-4.1-2025-04-14",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        #     temperature=1.0,
        # ),
        # "gpt-4.1-mini": ResponsesSampler(
        #     model="gpt-4.1-mini-2025-04-14",
        #     enable_web_search=True,
        #     system_message="You are a helpful assistant answering trivia questions you don't have the direct answers to in your knowledge. Always use the web search tool you have, but think step by step and break down the problem so that you dont make too complex of a search. search in stages and gather information as you go.",
        #     # max_tokens=2048,
        # ),
        # "gpt-4.1-nano": ChatCompletionSampler(
        #     model="gpt-4.1-nano-2025-04-14",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # # GPT-4o models
        # "gpt-4o": ChatCompletionSampler(
        #     model="gpt-4o",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-2024-11-20": ChatCompletionSampler(
        #     model="gpt-4o-2024-11-20",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-2024-08-06": ChatCompletionSampler(
        #     model="gpt-4o-2024-08-06",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-2024-08-06-temp-1": ChatCompletionSampler(
        #     model="gpt-4o-2024-08-06",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        #     temperature=1.0,
        # ),
        # "gpt-4o-2024-05-13": ChatCompletionSampler(
        #     model="gpt-4o-2024-05-13",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-mini": ChatCompletionSampler(
        #     model="gpt-4o-mini-2024-07-18",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # # GPT-4.5 model
        # "gpt-4.5-preview": ChatCompletionSampler(
        #     model="gpt-4.5-preview-2025-02-27",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # # GPT-4-turbo model
        # "gpt-4-turbo-2024-04-09": ChatCompletionSampler(
        #     model="gpt-4-turbo-2024-04-09",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        # ),
        # # GPT-4 model
        # "gpt-4-0613": ChatCompletionSampler(
        #     model="gpt-4-0613",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        # ),
        # # GPT-3.5 Turbo model
        # "gpt-3.5-turbo-0125": ChatCompletionSampler(
        #     model="gpt-3.5-turbo-0125",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        # ),
        # "gpt-3.5-turbo-0125-temp-1": ChatCompletionSampler(
        #     model="gpt-3.5-turbo-0125",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     temperature=1.0,
        # ),
        # # Chatgpt models:
        # "chatgpt-4o-latest": ChatCompletionSampler(
        #     model="chatgpt-4o-latest",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        #     max_tokens=2048,
        # ),
        # "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
        #     model="gpt-4-turbo-2024-04-09",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        # ),
        # # Claude models:
        # "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
        #     model="claude-3-opus-20240229",
        #     system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        # ),
        # "claude-3-7-sonnet-20250219": ClaudeCompletionSampler(
        #     model="claude-3-7-sonnet-20250219",
        #     system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        # ),
        # "claude-3-haiku-20240307": ClaudeCompletionSampler(
        #     model="claude-3-haiku-20240307",
        # ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        models_chosen = args.model.split(",")
        for model_name in models_chosen:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                return
        models = {model_name: models[model_name] for model_name in models_chosen}

    print(f"Running with args {args}")

    grading_sampler = ChatCompletionSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "browsecomp":
                return BrowseCompEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    # num_examples=2 if debug_mode else num_examples,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    if args.eval:
        evals_list = args.eval.split(",")
        evals = {}
        for eval_name in evals_list:
            try:
                evals[eval_name] = get_evals(eval_name, args.debug)
            except Exception as e:
                print(f"Error loading eval '{eval_name}': {e}")
                return
    else:
        evals = {
            "browsecomp": get_evals("browsecomp", args.debug)
        }

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    mergekey2resultpath = {}
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    
    for eval_name, eval_fn in evals.items():
        for model_name, sampler in models.items():
            print(f"\n=== Running {eval_name} for {model_name} ===")
            
            # Enable token tracking if requested
            if args.track_tokens and hasattr(sampler, 'track_tokens'):
                sampler.track_tokens = True
                print("\n=== TOKEN TRACKING ENABLED ===")
                print(f"Model: {model_name}")
                if hasattr(sampler, 'system_message'):
                    print(f"System message: {sampler.system_message[:100]}...")
                print("=============================\n")
            
            result = eval_fn(sampler)
            file_stem = f"{eval_name}_{model_name}_{date_str}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            
            # Add token usage to metrics if available
            if hasattr(sampler, 'token_counter'):
                token_usage = sampler.token_counter.get_usage()
                metrics.update({
                    "token_usage": token_usage,
                    "estimated_cost": token_usage["estimated_cost"]
                })
            
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print("\n=== Evaluation Metrics ===")
            print(json.dumps(metrics, indent=2))
            
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"\nWriting results to {result_filename}")
            
            full_result_filename = f"/tmp/{file_stem}{debug_suffix}_allresults.json"
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")
                
            mergekey2resultpath[f"{eval_name}_{model_name}"] = result_filename

    # Aggregate and display results
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            with open(result_filename, "r") as f:
                result = json.load(f)
            # Get the accuracy metric from the results
            metric_value = result.get("is_correct", 0.0)  # Default to 0.0 if not found
            eval_name = eval_model_name[:eval_model_name.rfind("_")]
            model_name = eval_model_name[eval_model_name.rfind("_") + 1:]
            merge_metrics.append(
                {"eval_name": eval_name, "model_name": model_name, "metric": metric_value}
            )
        except Exception as e:
            print(f"Error processing {result_filename}: {e}")
            continue
    
    if merge_metrics:
        merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
            index=["model_name"], columns="eval_name"
        )
        print("\n=== Final Results Summary ===")
        print(merge_metrics_df.to_markdown())
    else:
        print("\nNo results to display.")
    
    return merge_metrics


if __name__ == "__main__":
    main()
