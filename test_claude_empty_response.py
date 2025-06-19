#!/usr/bin/env python3
"""
Test script to specifically check if Claude is returning empty responses.

This script runs the unit tests for the Claude client to verify if it's
consistently returning empty responses.
"""

import os
import sys
import unittest
import logging
from pathlib import Path

# Import the unit test
from tests.test_claude_client import TestClaudeClient

def run_claude_tests():
    """Run the Claude client tests."""
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Claude empty response tests")
    
    # Create a test suite with just the Claude tests
    suite = unittest.TestSuite()
    
    # Focus on the empty response test specifically
    suite.addTest(TestClaudeClient('test_generate_empty_response'))
    suite.addTest(TestClaudeClient('test_generate_non_empty_response'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n=== Claude Empty Response Test Results ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # If all tests pass, it means our detection for empty responses is working
    if result.wasSuccessful():
        print("\nSuccess: The Claude empty response detection is working correctly.")
        print("Unit tests confirm we can detect both empty and non-empty responses.")
    else:
        print("\nWarning: There were issues with the Claude empty response tests.")
        print("Review the test output above for details.")
    
    return 0 if result.wasSuccessful() else 1

def run_all_claude_tests():
    """Run all Claude client tests."""
    # Create and run a test suite with all Claude tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestClaudeClient)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Claude for empty responses")
    parser.add_argument("--all", action="store_true", 
                        help="Run all Claude client tests instead of just the empty response tests")
    args = parser.parse_args()
    
    if args.all:
        sys.exit(run_all_claude_tests())
    else:
        sys.exit(run_claude_tests())
