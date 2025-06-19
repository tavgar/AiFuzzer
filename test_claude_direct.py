#!/usr/bin/env python3
"""
A direct test of the Claude API using our updated client implementation.
This script sends a simple request to Claude and prints the response.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.model_client import ClaudeClient
from src.utils.logger import setup_logging
from src.utils.config import load_config

async def test_claude_direct():
    """Test the Claude API directly with a simple request."""
    # Set up logging
    setup_logging(verbose=True)
    logger = logging.getLogger(__name__)
    
    print("\n===== DIRECT CLAUDE API TEST =====")
    print("Starting direct Claude API test with verbose output")
    logger.info("Starting direct Claude API test")
    
    # Get the API key - first try from config, then environment variable
    api_key = None
    try:
        config = load_config("examples/config.json")
        api_key = config.claude_api_key
        print(f"Loaded API key from config file. Key starts with: {api_key[:8]}...")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        print(f"Could not load config: {e}")
    
    # If API key not in config, try environment variable
    if not api_key:
        api_key = os.environ.get("CLAUDE_API_KEY")
    
    # Check if we have an API key
    if not api_key:
        logger.error("No Claude API key found in config or environment variables")
        print("Error: Claude API key not found.")
        print("Please provide it in config.json or set the CLAUDE_API_KEY environment variable.")
        return 1
    
    # Initialize the Claude client
    logger.info("Initializing Claude client")
    print(f"Initializing Claude client with API key: {api_key[:8]}...")
    client = ClaudeClient(api_key=api_key)
    
    # Test prompt
    prompt = "Hello, can you tell me what the capital of France is?"
    print(f"Using test prompt: '{prompt}'")
    
    try:
        # Generate response from Claude
        logger.info(f"Sending test prompt: {prompt}")
        print("Sending request to Claude API...")
        print("Request details:")
        print(f"  - Model: {client.model}")
        print(f"  - API URL: {client.api_url}")
        
        # Test with curl command first
        print("\nEquivalent curl command:")
        masked_key = f"{api_key[:8]}...{api_key[-5:]}"
        curl_cmd = f"""curl {client.api_url} \\
  -H "x-api-key: {masked_key}" \\
  -H "anthropic-version: 2023-06-01" \\
  -H "content-type: application/json" \\
  -d '{{
    "model": "{client.model}",
    "max_tokens": 1024,
    "temperature": 0.7,
    "system": "You are a helpful assistant that provides clear, accurate information.",
    "messages": [
      {{
        "role": "user",
        "content": [
          {{
            "type": "text",
            "text": "{prompt}"
          }}
        ]
      }}
    ],
    "thinking": {{
      "type": "enabled",
      "budget_tokens": 800
    }}
  }}'"""
        print(curl_cmd)
        print("\nSending request via Python client...")
        
        response = await client.generate(prompt)
        
        # Print the response
        print("\n=== CLAUDE RESPONSE ===")
        print(response)
        print("======================\n")
        
        # Check if the response is empty
        if not response.strip():
            logger.warning("Received empty response from Claude API")
            print("WARNING: Empty response received!")
            return 1
        else:
            logger.info(f"Received non-empty response ({len(response)} chars)")
            print(f"SUCCESS: Received {len(response)} characters in response")
            return 0
            
    except Exception as e:
        logger.error(f"Error testing Claude API: {e}")
        print(f"ERROR: {e}")
        import traceback
        print("\nDetailed error traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(test_claude_direct())
    sys.exit(exit_code)
