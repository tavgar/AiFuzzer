#!/usr/bin/env python3
"""
Test script for running the AI Fuzzer with coronavirus OC43 culturing prompts.
"""

import asyncio
import logging
import sys
from pathlib import Path

from src.utils.config import initialize_config
from src.utils.logger import setup_logging
from src.core.fuzzing_engine import FuzzingEngine

async def main():
    """Run a quick test of the fuzzing engine with a small number of attempts."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Coronavirus OC43 Culturing Prompt Test")
    
    # Create a simplified configuration
    class Config:
        def __init__(self):
            # Load from config.json
            import json
            with open("examples/config.json", "r") as f:
                config_data = json.load(f)
            
            # Set properties from the loaded config
            for key, value in config_data.items():
                setattr(self, key, value)
            
            # Override some settings for testing
            self.max_attempts = 10  # Just do a few attempts for testing
            self.verbose = True
            
    config = Config()
    
    # Run the fuzzing engine
    engine = FuzzingEngine(config)
    await engine._run_fuzzing()
    
    # Print summary
    engine._print_summary()
    
    logger.info("Test completed")
    return 0

if __name__ == "__main__":
    asyncio.run(main())
