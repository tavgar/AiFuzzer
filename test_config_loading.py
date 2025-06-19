#!/usr/bin/env python3
"""
Test script to verify configuration loading from config file.
"""

import sys
import logging
from src.utils.config import initialize_config
from src.cli.cli_handler import parse_args

def main():
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Simulate command line arguments
    sys.argv = [
        "test_config_loading.py",
        "--config", "examples/config.json",
        "--initial-prompts", "examples/initial_prompts.txt",
        "--target-behaviors", "examples/target_behaviors.txt"
    ]
    
    # Parse arguments and initialize config
    args = parse_args()
    config = initialize_config(args)
    
    # Print the configuration info
    logger.info("=== Configuration Loaded ===")
    logger.info(f"Gemini Model: {config.gemini_model}")
    logger.info(f"Claude Model: {config.claude_model}")
    logger.info(f"Max Attempts: {config.max_attempts}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info(f"Memory Size: {config.memory_size}")
    logger.info("===========================")
    
    # Verify model names match what's in the config file
    from pathlib import Path
    import json
    
    with open(Path("examples/config.json"), "r") as f:
        config_data = json.load(f)
    
    gemini_match = config.gemini_model == config_data.get("gemini_model")
    claude_match = config.claude_model == config_data.get("claude_model")
    
    logger.info(f"Gemini model matches config file: {gemini_match}")
    logger.info(f"Claude model matches config file: {claude_match}")
    
    if gemini_match and claude_match:
        logger.info("✅ SUCCESS: Configuration is loading correctly!")
    else:
        logger.error("❌ ERROR: Configuration is not loading correctly!")

if __name__ == "__main__":
    main()
