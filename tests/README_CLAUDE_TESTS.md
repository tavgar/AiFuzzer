# Claude Empty Response Testing

This directory contains tests specifically designed to check if Claude is consistently returning empty responses.

## Background

If you're experiencing issues where Claude is returning empty responses, these tests can help diagnose the problem by:

1. Testing if the empty response detection is working correctly (unit tests)
2. Running integration tests against the actual Claude API to determine the rate of empty responses

## Test Files

- `tests/test_claude_client.py`: Unit tests for the Claude client, including specific tests for empty response detection
- `test_claude_empty_response.py`: A simple runner for the Claude unit tests
- `test_claude_integration.py`: A comprehensive integration test that sends various prompts to the real Claude API and reports on empty responses

## Running the Tests

### Unit Tests

To run the unit tests for the Claude client:

```bash
# Run just the empty response tests
python test_claude_empty_response.py

# Run all Claude client tests
python test_claude_empty_response.py --all
```

These tests use mocks and don't require a real API key.

### Integration Tests

To run integration tests against the real Claude API:

```bash
# Run with default settings (3 tests per prompt)
python test_claude_integration.py

# Run with a specific model
python test_claude_integration.py --model claude-3-sonnet-20240229

# Run more tests per prompt
python test_claude_integration.py --num-tests 5

# Specify a custom output file
python test_claude_integration.py --output custom_results.json
```

**Note:** Integration tests require a valid Claude API key, which should be in your `examples/config.json` file or set as the `CLAUDE_API_KEY` environment variable.

## Interpreting Results

### Unit Test Results

The unit tests will confirm that the code can properly detect both empty and non-empty responses from Claude.

### Integration Test Results

The integration tests will provide:

1. Overall statistics on empty vs. non-empty responses
2. Breakdown by prompt type
3. Detailed logs for each test run
4. A JSON file with comprehensive results

If you're seeing a high rate of empty responses, consider:

- Checking your API key validity
- Trying a different Claude model
- Checking for rate limiting
- Contacting Anthropic support

## Exit Codes

- `0`: Tests passed successfully, no empty responses detected in integration tests
- `1`: Configuration error (e.g., missing API key)
- `2`: Empty responses were detected in integration tests
