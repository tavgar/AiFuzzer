# Claude-to-Claude Fuzzing

This feature allows you to use one Claude model (Claude-3-7-Sonnet) to generate jailbreak prompts and test another Claude model (Claude-Neptune).

## Overview

The standard aifuzzer setup uses Gemini to generate prompts that are tested against Claude. This new feature replaces Gemini's role with a Claude model, allowing for an all-Claude testing environment.

### Key Components:

- **Generator Model**: Claude-3-7-Sonnet (creates jailbreak prompts)
- **Target Model**: Claude-Neptune (model being tested)

## Quick Start

The easiest way to run the Claude-to-Claude fuzzer is to use the helper script:

```bash
# Set your Claude API key as an environment variable
export CLAUDE_API_KEY="your_api_key_here"

# Run the helper script
python aifuzzer/run_sonnet_neptune_test.py
```

## Command Line Options

The helper script supports several command-line options:

```
--claude-api-key TEXT  API key for Anthropic's Claude models
--max-attempts INT     Maximum number of fuzzing attempts (default: 20)
--batch-size INT       Number of prompts per batch (default: 5)
--verbose, -v          Enable verbose output
```

Example with options:

```bash
python aifuzzer/run_sonnet_neptune_test.py --max-attempts 50 --batch-size 10 --verbose
```

## Advanced Usage

For more control, you can use the main Claude-to-Claude fuzzer directly with custom parameters:

```bash
python aifuzzer/run_claude_to_claude_fuzzer.py \
  --claude-api-key "your_api_key" \
  --generator-model "claude-3-7-sonnet-20250219" \
  --target-model "claude-neptune" \
  --max-attempts 30 \
  --batch-size 5 \
  --temperature 0.9 \
  --verbose
```

Available options:

```
--claude-api-key TEXT       API key for Anthropic's Claude models
--generator-model TEXT      Claude model for generating prompts (default: claude-3-7-sonnet-20250219)
--target-model TEXT         Claude model to test against (default: claude-neptune)
--max-attempts INT          Maximum number of fuzzing attempts (default: 50)
--batch-size INT            Number of prompts per batch (default: 5)
--temperature FLOAT         Temperature for generation (default: 0.9)
--timeout INT               Timeout in seconds for API calls (default: 30)
--initial-prompts TEXT      Path to initial prompts file (default: examples/initial_prompts.txt)
--target-behaviors TEXT     Path to target behaviors file (default: examples/target_behaviors.txt)
--output-dir TEXT           Directory to store output files (default: output)
--log-dir TEXT              Directory to store log files (default: logs)
--config TEXT               Path to configuration file
--verbose, -v               Enable verbose output
```

## Understanding the Results

The fuzzer will display:

1. Progress bar showing fuzzing attempts
2. Details about successful jailbreaks (if any)
3. A summary at the end with:
   - Generator and target models used
   - Total attempts and successful jailbreaks
   - Success rate
   - Examples of successful jailbreaks
   - Analysis of effective techniques

Output files are saved to the `output` directory:
- `results_YYYYMMDD_HHMMSS.jsonl`: All fuzzing attempts
- `successful_YYYYMMDD_HHMMSS.jsonl`: Only successful jailbreaks
- `technique_analytics_YYYYMMDD_HHMMSS.json`: Analysis of which techniques were most effective

## How It Works

1. **Generation**: Claude-3-7-Sonnet generates jailbreak prompts based on patterns that have proven effective against Claude models
2. **Testing**: These prompts are sent to Claude-Neptune
3. **Evaluation**: Claude-3-7-Sonnet evaluates Claude-Neptune's responses to determine if the jailbreak was successful
4. **Learning**: The system learns from results and improves its prompt generation over time

This approach leverages Claude-3-7-Sonnet's understanding of language models to create more effective jailbreak attempts against Claude-Neptune.
