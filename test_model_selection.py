"""
Test script to verify model selection functionality.
"""

import asyncio
import os
from src.utils.config import FuzzerConfig
from src.models.model_client import GeminiClient, ClaudeClient

async def test_models():
    """Test Gemini and Claude with different models."""
    
    # Load API keys from environment
    gemini_key = os.environ.get('GEMINI_API_KEY', 'your_gemini_key_here')
    claude_key = os.environ.get('CLAUDE_API_KEY', 'your_claude_key_here')
    
    print("Testing Gemini models...")
    
    # Test with different Gemini models
    gemini_models = [
        "gemini-1.5-pro",
        "gemini-2.5-pro-preview-0506",
        "gemini-2.5-pro"
    ]
    
    for model in gemini_models:
        print(f"\nInitializing Gemini with model: {model}")
        try:
            client = GeminiClient(api_key=gemini_key, model=model)
            print(f"✓ Successfully initialized {model}")
        except Exception as e:
            print(f"✗ Failed to initialize {model}: {e}")
    
    print("\nTesting Claude models...")
    
    # Test with different Claude models
    claude_models = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-7-sonnet-20250219"
    ]
    
    for model in claude_models:
        print(f"\nInitializing Claude with model: {model}")
        try:
            client = ClaudeClient(api_key=claude_key, model=model)
            print(f"✓ Successfully initialized {model}")
        except Exception as e:
            print(f"✗ Failed to initialize {model}: {e}")
    
    # Test type detection for models
    print("\nTesting model type detection...")
    config = FuzzerConfig()
    
    for model in gemini_models + claude_models:
        family = config.get_model_family(model)
        is_claude = config.is_claude_model(model)
        is_gemini = config.is_gemini_model(model)
        print(f"Model: {model}")
        print(f"  Family: {family}")
        print(f"  Is Claude: {is_claude}")
        print(f"  Is Gemini: {is_gemini}")

if __name__ == "__main__":
    asyncio.run(test_models())
