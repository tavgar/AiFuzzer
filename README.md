# LLM Jailbreak Fuzzer

A command-line tool for testing LLM safety measures by automatically generating and evaluating jailbreak prompts. This tool uses Google's Gemini to generate potential jailbreak prompts and tests them against Anthropic's Claude model.

## Features

- Automatic generation of diverse jailbreak prompts using Gemini 2.5 Pro Preview
- Automatic evaluation of responses to determine if jailbreak was successful
- Learning mechanism that improves prompts based on previous attempts
- Detailed logging and statistics
- Interactive and batch modes
- Configurable targeting of specific behaviors

## Advanced Techniques

The fuzzer includes sophisticated techniques to improve jailbreak effectiveness:

### Genetic Algorithms
- Evolves prompts through selection, crossover, and mutation operations
- Tracks "prompt lineage" to understand which evolutionary paths are most effective
- Uses fitness functions that consider both success rate and pattern effectiveness

### Token Manipulation
- Character substitution (using similar-looking characters)
- Zero-width space insertion in sensitive words
- Homoglyph substitution (replacing characters with visually similar ones)
- Word splitting to bypass token-based filters

### Context Manipulation
- Multi-layered attack strategies (nested contexts and role-playing)
- Hypothetical frameworks ("in a universe where...")
- Advanced misdirection techniques
- Cognitive exploits (reasoning chains that justify harmful outputs)

### Enhanced Learning
- Pattern analysis across successful and failed attempts
- Clustering of effective techniques
- Automatic enhancement of prompts based on learned patterns

## Installation

### Prerequisites

- Python 3.8 or higher
- API keys for Google's Gemini and Anthropic's Claude

### Install from source

```bash
git clone https://github.com/yourusername/aifuzzer.git
cd aifuzzer
pip install -e .
```

## Configuration Setup

### 🔒 API Key Security

**IMPORTANT**: Never commit API keys to version control. This repository includes security measures to prevent accidental exposure.

### Setting Up Configuration

1. **Copy the template configuration file:**
   ```bash
   cp examples/config.template.json examples/config.json
   ```

2. **Edit the configuration file:**
   ```bash
   # Open examples/config.json and replace the placeholder values:
   # "YOUR_GEMINI_API_KEY_HERE" -> your actual Gemini API key
   # "YOUR_CLAUDE_API_KEY_HERE" -> your actual Claude API key
   ```

3. **Alternatively, use environment variables:**
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key"
   export CLAUDE_API_KEY="your_claude_api_key"
   ```

### Getting API Keys

- **Gemini API Key**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Claude API Key**: Get from [Anthropic Console](https://console.anthropic.com/)

### Configuration File Structure

The `examples/config.json` file (ignored by git) contains:
- API credentials
- Model selection (Gemini 2.5 Pro Preview, Claude 3.7 Sonnet, etc.)
- Fuzzing parameters (batch size, temperature, max attempts)
- Advanced technique settings
- Output directory configuration

**Note**: The `examples/config.json` file is automatically ignored by git to prevent accidental commit of API keys.

## Usage

### Basic Usage

```bash
# Run with API keys as environment variables
export GEMINI_API_KEY="your_gemini_api_key"
export CLAUDE_API_KEY="your_claude_api_key"
aifuzzer --max-attempts 20
```

### Interactive Mode

```bash
aifuzzer --interactive
```

### Using Configuration Files

```bash
# Create a configuration file with your settings
aifuzzer --interactive --save-config config.json

# Use the saved configuration
aifuzzer --config config.json
```

### Advanced Options

```bash
# Target specific behaviors
aifuzzer --target-behaviors examples/target_behaviors.txt

# Start with initial prompts
aifuzzer --initial-prompts examples/initial_prompts.txt

# Customize model selection
aifuzzer --gemini-model "gemini-2.5-pro-preview-0506" --claude-model "claude-3-opus-20240229"

# Adjust fuzzing parameters
aifuzzer --batch-size 10 --temperature 0.8 --max-attempts 100
```

### Advanced Techniques Options

```bash
# Enable or disable advanced techniques
aifuzzer --use-advanced-techniques  # Enable (default)
aifuzzer --no-advanced-techniques   # Disable

# Configure genetic algorithm parameters
aifuzzer --genetic-algorithm-population 30 --genetic-algorithm-mutation-rate 0.4 --genetic-algorithm-crossover-rate 0.8

# Configure token and context manipulation
aifuzzer --token-manipulation-intensity 0.7 --context-manipulation-probability 0.5

# Save technique analytics
aifuzzer --save-technique-analytics
```

## Output

The tool outputs results in several formats:

1. **Console output** - Shows real-time progress and results
2. **Log files** - Detailed logging of the tool's operation
3. **Result files** - JSONL files containing all attempted prompts and responses
4. **Success files** - JSONL files containing only successful jailbreak attempts

Results are stored in the `output` directory by default.

## Ethical Considerations

This tool is intended for responsible security research and improvement of AI safety measures. Usage should comply with:

1. Terms of service for both Google's and Anthropic's APIs
2. Responsible disclosure practices
3. Ethical research guidelines

Do not use this tool to deploy jailbreak prompts in production environments or to harm users.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
