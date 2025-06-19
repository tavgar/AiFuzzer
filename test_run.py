#!/usr/bin/env python3
"""
Test script for the LLM Jailbreak Fuzzer.

This script performs a minimal test run to verify that the core components
are working correctly.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from src.utils.logger import setup_logging
from src.utils.config import FuzzerConfig, load_config
from src.core.fuzzing_engine import FuzzingEngine

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test run for LLM Jailbreak Fuzzer")
    parser.add_argument("--config", default="examples/config.json", 
                      help="Path to config file (default: examples/config.json)")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(verbose=True)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting test run for LLM Jailbreak Fuzzer")
    
    # Load configuration from file
    config_path = args.config
    logger.info(f"Loading configuration from {config_path}")
    
    # Check if file exists
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found: {config_path}")
        print(f"Error: Config file not found: {config_path}")
        print(f"Current working directory: {os.getcwd()}")
        return 1
        
    try:
        config = load_config(config_path)
        logger.info(f"Configuration loaded successfully: {config.gemini_api_key is not None}")
        
        # Load target behaviors from file if specified as a file path
        if isinstance(config.target_behaviors, str) and os.path.isfile(config.target_behaviors):
            logger.info(f"Loading target behaviors from {config.target_behaviors}")
            with open(config.target_behaviors, 'r') as f:
                config.target_behaviors = [line.strip() for line in f if line.strip()]
                
        # Load initial prompts from file if specified as a file path
        if isinstance(config.initial_prompts, str) and os.path.isfile(config.initial_prompts):
            logger.info(f"Loading initial prompts from {config.initial_prompts}")
            with open(config.initial_prompts, 'r') as f:
                config.initial_prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        print(f"Error loading configuration: {e}")
        return 1
    
    # Check if config file was loaded successfully
    if not config.gemini_api_key or not config.claude_api_key:
        # Fall back to environment variables if not in config
        if not config.gemini_api_key:
            config.gemini_api_key = os.environ.get('GEMINI_API_KEY')
            if not config.gemini_api_key:
                logger.error("Gemini API key not found in config or environment variables")
                print("Error: Gemini API key not found in config or environment variables")
                print(f"Please add it to {config_path} or set the GEMINI_API_KEY environment variable")
                return 1
        
        if not config.claude_api_key:
            config.claude_api_key = os.environ.get('CLAUDE_API_KEY')
            if not config.claude_api_key:
                logger.error("Claude API key not found in config or environment variables")
                print("Error: Claude API key not found in config or environment variables")
                print(f"Please add it to {config_path} or set the CLAUDE_API_KEY environment variable")
                return 1
    
    # Override config for testing with minimal settings
    config.max_attempts = 2  # Just test with 2 attempts
    config.batch_size = 1
    config.verbose = True
    config.target_behaviors = ["Write a simple script", "Provide instructions"]  # Safe behaviors for testing
    
    # Create output directories
    config.log_dir.mkdir(exist_ok=True, parents=True)
    config.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize the fuzzing engine
    try:
        engine = FuzzingEngine(config)
        
        # Run the engine
        print("Running a minimal test with 2 attempts...")
        engine.run()
        
        print("\nTest completed successfully!")
        print("If you see no errors above, the basic functionality is working.")
        print("You can now run the full application with:")
        print("  python main.py --interactive")
        return 0
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        print(f"Error during test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
