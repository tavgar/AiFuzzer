#!/usr/bin/env python3
"""
LLM Jailbreak Fuzzer - Main Entry Point

This tool uses Gemini to generate and test jailbreak prompts against Claude.
"""

import argparse
import sys
import logging
from pathlib import Path

from src.cli.cli_handler import parse_args, run_interactive_mode
from src.core.fuzzing_engine import FuzzingEngine
from src.utils.config import initialize_config
from src.utils.logger import setup_logging

def main():
    """Main entry point for the application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LLM Jailbreak Fuzzer")
    
    # Parse command line arguments
    args = parse_args()
    
    # Initialize configuration
    config = initialize_config(args)
    
    if args.interactive:
        run_interactive_mode(config)
    else:
        # Run in standard mode
        engine = FuzzingEngine(config)
        engine.run()
    
    logger.info("LLM Jailbreak Fuzzer completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
