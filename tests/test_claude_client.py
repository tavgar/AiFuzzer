#!/usr/bin/env python3
"""
Unit tests for the Claude client to verify response handling.
This test specifically checks if the Claude client is returning empty responses.
"""

import unittest
import asyncio
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_client import ClaudeClient, ModelClientError


class TestClaudeClient(unittest.TestCase):
    """Test cases for the Claude client."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a dummy API key for testing
        self.api_key = "dummy_api_key"
        self.model = "claude-3-opus-20240229"
        
        # Mock the Anthropic client
        self.anthropic_patcher = patch('src.models.model_client.Anthropic')
        self.mock_anthropic = self.anthropic_patcher.start()
        
        # Create a mock Anthropic client instance
        self.mock_client = MagicMock()
        self.mock_anthropic.return_value = self.mock_client
        
        # Create a mock messages object
        self.mock_messages = MagicMock()
        self.mock_client.messages = self.mock_messages
        
        # Initialize the Claude client
        self.claude_client = ClaudeClient(
            api_key=self.api_key,
            model=self.model
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.anthropic_patcher.stop()

    def test_initialize_client(self):
        """Test that the client initializes correctly."""
        self.assertEqual(self.claude_client.api_key, self.api_key)
        self.assertEqual(self.claude_client.model, self.model)
        self.assertEqual(self.claude_client.api_url, "https://api.anthropic.com/v1/messages")
        # Verify that the Anthropic client was created with the correct API key
        self.mock_anthropic.assert_called_once_with(api_key=self.api_key)

    def test_generate_empty_response(self):
        """Test that we can detect when Claude returns an empty response."""
        # Configure the mock to return an empty content response
        mock_message = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = ""  # Empty text response
        mock_message.content = [mock_content_block]
        self.mock_messages.create.return_value = mock_message

        # Run the generate method and get the result
        result = asyncio.run(self.claude_client.generate("Test prompt"))
        
        # Verify the result is empty
        self.assertEqual(result, "")
        
        # Verify the API was called with the right parameters
        self.mock_messages.create.assert_called_once()
        call_args = self.mock_messages.create.call_args[1]
        self.assertEqual(call_args["model"], self.model)
        self.assertEqual(call_args["messages"][0]["content"], "Test prompt")

    def test_generate_non_empty_response(self):
        """Test that we can receive a non-empty response from Claude."""
        # Configure the mock to return a non-empty content response
        mock_message = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "This is a test response"
        mock_message.content = [mock_content_block]
        self.mock_messages.create.return_value = mock_message

        # Run the generate method and get the result
        result = asyncio.run(self.claude_client.generate("Test prompt"))
        
        # Verify the result is not empty
        self.assertEqual(result, "This is a test response")

    def test_generate_multiple_content_blocks(self):
        """Test handling of multiple content blocks in the response."""
        # Configure the mock to return multiple content blocks
        mock_message = MagicMock()
        mock_content_block1 = MagicMock()
        mock_content_block1.type = "text"
        mock_content_block1.text = "First part. "
        
        mock_content_block2 = MagicMock()
        mock_content_block2.type = "text"
        mock_content_block2.text = "Second part."
        
        mock_message.content = [mock_content_block1, mock_content_block2]
        self.mock_messages.create.return_value = mock_message

        # Run the generate method and get the result
        result = asyncio.run(self.claude_client.generate("Test prompt"))
        
        # Verify the result combines all text blocks
        self.assertEqual(result, "First part. Second part.")

    def test_generate_no_content_attribute(self):
        """Test handling when the response has no content attribute."""
        # Configure the mock to return a response with no content attribute
        mock_message = MagicMock()
        # Remove the content attribute
        del mock_message.content
        # Make string representation return something predictable
        mock_message.__str__.return_value = "Message with no content"
        self.mock_messages.create.return_value = mock_message

        # Run the generate method and get the result
        result = asyncio.run(self.claude_client.generate("Test prompt"))
        
        # Verify the result is the string representation of the message
        self.assertEqual(result, "Message with no content")

    def test_api_error_handling(self):
        """Test that API errors are properly handled."""
        # Make the API call raise an exception
        self.mock_messages.create.side_effect = Exception("API error")
        
        # Verify that the exception is caught and re-raised as a ModelClientError
        with self.assertRaises(ModelClientError) as context:
            asyncio.run(self.claude_client.generate("Test prompt"))
        
        self.assertIn("Error generating text with Claude", str(context.exception))

    def test_retry_mechanism(self):
        """Test that the retry mechanism works as expected."""
        # Configure the mock to raise an exception on first call, then succeed
        self.mock_messages.create.side_effect = [
            Exception("Temporary error"),  # First call fails
            MagicMock(  # Second call succeeds
                content=[MagicMock(type="text", text="Success after retry")]
            )
        ]
        
        # Run the generate method and get the result
        result = asyncio.run(self.claude_client.generate("Test prompt"))
        
        # Verify the result is from the successful retry
        self.assertEqual(result, "Success after retry")
        
        # Verify the API was called twice
        self.assertEqual(self.mock_messages.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
