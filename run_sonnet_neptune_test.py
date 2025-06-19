#!/usr/bin/env python3
"""
Helper script to run Claude-to-Claude fuzzing with:
- Claude-3-7-Sonnet as the generator (tester)
- Claude-Neptune as the target model being tested

This script simplifies the process of running the fuzzer with the specified models.
"""

import os
import sys
import argparse
from pathlib import Path

from src.utils.config import FuzzerConfig
from run_claude_to_claude_fuzzer import ClaudeToClaudeFuzzer, setup_logging

def main():
    """Run the Claude-to-Claude fuzzer with default settings."""
    # Setup logging
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Claude-3-7-Sonnet to test Claude-Neptune",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # API key
    parser.add_argument(
        "--claude-api-key", 
        help="API key for Anthropic's Claude models (can also use CLAUDE_API_KEY env var)"
    )
    
    # Fuzzing parameters
    parser.add_argument(
        "--max-attempts", 
        type=int, 
        default=20,
        help="Maximum number of fuzzing attempts"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=5,
        help="Number of prompts to generate in each batch"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = FuzzerConfig()
    
    # Set the models
    config.generator_model = "claude-3-7-sonnet-20250219"  # Generator model
    config.claude_model = "claude-neptune"                # Target model
    
    # Set API key from command line or environment variable
    if args.claude_api_key:
        config.claude_api_key = args.claude_api_key
    elif os.environ.get('CLAUDE_API_KEY'):
        config.claude_api_key = os.environ.get('CLAUDE_API_KEY')
    else:
        print("ERROR: Claude API key not provided. Use --claude-api-key or set CLAUDE_API_KEY environment variable.")
        return 1
    
    # Set fuzzing parameters
    config.max_attempts = args.max_attempts
    config.batch_size = args.batch_size
    config.verbose = args.verbose
    
    # Setup directories
    config.log_dir = Path("logs")
    config.output_dir = Path("output")
    config.log_dir.mkdir(exist_ok=True, parents=True)
    config.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Enable advanced techniques
    config.use_advanced_techniques = True
    config.save_technique_analytics = True
    
    # Set target behaviors and initial prompts
    config.initial_prompts = "examples/initial_prompts.txt"
    config.target_behaviors = "examples/target_behaviors.txt"
    
    # Run the fuzzer
    print(f"\n{'='*60}")
    print(f"🔍 Starting Fuzzing Test:")
    print(f"   🧪 Generator Model: {config.generator_model}")
    print(f"   🎯 Target Model: {config.claude_model}")
    print(f"   🔄 Max Attempts: {config.max_attempts}")
    print(f"{'='*60}\n")
    
    engine = ClaudeToClaudeFuzzer(config)
    engine.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
