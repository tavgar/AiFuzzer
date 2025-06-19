#!/usr/bin/env python3
"""
Integration test script to check if the real Claude API is consistently returning empty responses.

This script sends a series of test prompts to the Claude API and records whether
the responses are empty or not. It helps diagnose if there's a consistent issue
with the Claude API returning empty responses.

IMPORTANT: This test requires a valid Claude API key to be provided either in the
config file or as an environment variable CLAUDE_API_KEY.
"""

import os
import sys
import asyncio
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.model_client import ClaudeClient
from src.utils.logger import setup_logging
from src.utils.config import load_config

# Test prompts to send to Claude
TEST_PROMPTS = [
    "Hello, how are you today?",
    "What is the capital of France?",
    "Write a short poem about AI.",
    "Explain how a computer works in simple terms.",
    "What are the benefits of regular exercise?",
    "Can you tell me about the history of the internet?",
    "What is machine learning?",
    "Tell me a joke.",
    "What's the weather like today?",  # This should trigger a refusal since Claude doesn't know current weather
    "Provide a recipe for chocolate chip cookies.",
]

async def test_claude_responses(api_key, model="claude-3-opus-20240229", num_tests=3):
    """
    Test Claude API responses to see if they're empty.
    
    Args:
        api_key: Claude API key
        model: Claude model to use
        num_tests: Number of times to run each test prompt
        
    Returns:
        Results dictionary with statistics
    """
    # Initialize the Claude client
    claude_client = ClaudeClient(api_key=api_key, model=model)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info(f"Testing Claude API responses with model: {model}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "total_requests": 0,
        "empty_responses": 0,
        "non_empty_responses": 0,
        "error_responses": 0,
        "prompt_results": {},
    }
    
    # Test each prompt multiple times
    for prompt in TEST_PROMPTS:
        prompt_result = {
            "prompt": prompt,
            "tests": [],
            "empty_count": 0,
            "non_empty_count": 0,
            "error_count": 0,
        }
        
        # Run the test multiple times for each prompt
        for i in range(num_tests):
            try:
                logger.info(f"Test {i+1}/{num_tests} for prompt: {prompt[:50]}...")
                
                # Generate response from Claude
                response = await claude_client.generate(prompt, temperature=0.7)
                
                # Check if the response is empty
                is_empty = not response.strip()
                
                # Record the result
                test_result = {
                    "attempt": i + 1,
                    "is_empty": is_empty,
                    "response_length": len(response),
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
                }
                
                prompt_result["tests"].append(test_result)
                
                if is_empty:
                    prompt_result["empty_count"] += 1
                    results["empty_responses"] += 1
                    logger.warning(f"Empty response received for prompt: {prompt[:50]}...")
                else:
                    prompt_result["non_empty_count"] += 1
                    results["non_empty_responses"] += 1
                    logger.info(f"Non-empty response ({len(response)} chars) received")
                
                results["total_requests"] += 1
                
                # Small delay between requests to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error during test: {e}")
                test_result = {
                    "attempt": i + 1,
                    "error": str(e)
                }
                prompt_result["tests"].append(test_result)
                prompt_result["error_count"] += 1
                results["error_responses"] += 1
                results["total_requests"] += 1
        
        # Add the prompt result to the overall results
        results["prompt_results"][prompt] = prompt_result
    
    return results

def print_summary(results):
    """Print a summary of the test results."""
    print("\n==== CLAUDE API RESPONSE TEST SUMMARY ====")
    print(f"Model: {results['model']}")
    print(f"Total requests: {results['total_requests']}")
    print(f"Empty responses: {results['empty_responses']} ({results['empty_responses']/results['total_requests']*100:.1f}%)")
    print(f"Non-empty responses: {results['non_empty_responses']} ({results['non_empty_responses']/results['total_requests']*100:.1f}%)")
    print(f"Errors: {results['error_responses']} ({results['error_responses']/results['total_requests']*100:.1f}%)")
    
    print("\nResults by prompt:")
    for prompt, result in results["prompt_results"].items():
        print(f"\nPrompt: {prompt[:50]}..." if len(prompt) > 50 else f"\nPrompt: {prompt}")
        print(f"  Empty: {result['empty_count']}/{len(result['tests'])} ({result['empty_count']/len(result['tests'])*100:.1f}%)")
        print(f"  Non-empty: {result['non_empty_count']}/{len(result['tests'])} ({result['non_empty_count']/len(result['tests'])*100:.1f}%)")
        print(f"  Errors: {result['error_count']}/{len(result['tests'])} ({result['error_count']/len(result['tests'])*100:.1f}%)")

async def main():
    """Run the Claude API response test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Claude API for empty responses")
    parser.add_argument("--config", default="examples/config.json", 
                      help="Path to config file (default: examples/config.json)")
    parser.add_argument("--model", default="claude-3-opus-20240229",
                      help="Claude model to use (default: claude-3-opus-20240229)")
    parser.add_argument("--num-tests", type=int, default=3,
                      help="Number of times to test each prompt (default: 3)")
    parser.add_argument("--output", default="output/claude_test_results.json",
                      help="Path to save test results (default: output/claude_test_results.json)")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(verbose=True)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Claude API empty response test")
    
    # Get the API key - first try from config, then environment variable
    api_key = None
    try:
        config = load_config(args.config)
        api_key = config.claude_api_key
    except Exception as e:
        logger.warning(f"Could not load config from {args.config}: {e}")
    
    # If API key not in config, try environment variable
    if not api_key:
        api_key = os.environ.get("CLAUDE_API_KEY")
    
    # Check if we have an API key
    if not api_key:
        logger.error("No Claude API key found in config or environment variables")
        print("Error: Claude API key not found.")
        print("Please provide it in config.json or set the CLAUDE_API_KEY environment variable.")
        return 1
    
    # Run the tests
    logger.info(f"Running Claude API tests with model {args.model}")
    results = await test_claude_responses(api_key, args.model, args.num_tests)
    
    # Print summary
    print_summary(results)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save results to file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {output_path}")
    
    # Determine overall result and return appropriate exit code
    if results["empty_responses"] > 0:
        print("\nWARNING: Empty responses were detected!")
        print(f"Claude returned empty responses for {results['empty_responses']} out of {results['total_requests']} requests.")
        return 2  # Return a specific code for empty responses
    else:
        print("\nSUCCESS: No empty responses detected!")
        return 0

if __name__ == "__main__":
    asyncio.run(main())
