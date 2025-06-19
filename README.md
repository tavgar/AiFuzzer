# 🎯 AiFuzzer - Advanced LLM Safety Testing Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-tavgar/AiFuzzer-black.svg)](https://github.com/tavgar/AiFuzzer)

A state-of-the-art command-line framework for testing LLM safety measures through automated jailbreak prompt generation and evaluation. Features advanced techniques including genetic algorithms, context manipulation, and the highly effective **Foot-in-the-Door technique** with **94% average success rate**.

## 🌟 Key Features

### Core Capabilities
- **Multi-Model Support**: Test Anthropic Claude models using Google Gemini or Claude-to-Claude generation
- **Advanced Fuzzing Engine**: Genetic algorithms, token manipulation, and context overflow techniques
- **Interactive & Batch Modes**: Full CLI support with detailed configuration options
- **Comprehensive Analytics**: Track technique effectiveness and generate detailed reports
- **Security-First Design**: Built-in API key protection and secure configuration management

### Specialized Techniques

#### 🚪 Foot-in-the-Door (FITD) Technique
- **94% average success rate** across various models
- Multi-turn conversation approach with progressive escalation
- Three bridge strategies: gradual, context-shift, and role-based
- Dynamic adaptation based on model responses

#### 🧬 Genetic Algorithms
- Evolutionary prompt optimization through selection, crossover, and mutation
- Fitness tracking and lineage analysis
- Population-based improvement over generations

#### 🔧 Token Manipulation
- Character substitution and homoglyph replacement
- Zero-width space insertion and word splitting
- Leetspeak transformation and unicode obfuscation

#### 🎭 Context Manipulation
- Role-playing scenarios and hypothetical frameworks
- Multi-layered attacks with nested contexts
- Cognitive exploits and confusion techniques
- Context overflow with strategic positioning

#### 🔄 Claude-to-Claude Fuzzing
- Use Claude-3.7-Sonnet to generate prompts for Claude-Neptune
- All-Claude testing environment for specialized evaluation
- Advanced pattern recognition and adaptation

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/tavgar/AiFuzzer.git
cd AiFuzzer
pip install -e .
```

### Configuration Setup

1. **Copy the template configuration:**
   ```bash
   cp examples/config.template.json examples/config.json
   ```

2. **Add your API keys:**
   ```bash
   # Edit examples/config.json:
   # "gemini_api_key": "your_actual_gemini_key"
   # "claude_api_key": "your_actual_claude_key"
   ```

3. **Get API Keys:**
   - **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey)
   - **Claude**: [Anthropic Console](https://console.anthropic.com/)

### Basic Usage

```bash
# Standard fuzzing with Gemini → Claude
aifuzzer --config examples/config.json --max-attempts 50

# Interactive mode with guided setup
aifuzzer --interactive

# Enable advanced techniques with analytics
aifuzzer --use-advanced-techniques --save-technique-analytics
```

## 🎯 Specialized Testing Modes

### Foot-in-the-Door Testing
```bash
# Run FITD technique tests (94% success rate)
python run_foot_in_door_test.py --num-tests 20 --escalation-levels 4

# Dynamic approach with real-time adaptation
python run_foot_in_door_test.py --dynamic --max-failures 3

# Context-shift bridge strategy
python run_foot_in_door_test.py --bridge-strategy context_shift
```

### Claude-to-Claude Fuzzing
```bash
# Quick Claude-to-Claude test
python run_sonnet_neptune_test.py --max-attempts 30 --verbose

# Advanced Claude-to-Claude with custom models
python run_claude_to_claude_fuzzer.py \
  --generator-model claude-3-7-sonnet-20250219 \
  --target-model claude-neptune \
  --temperature 0.9
```

### Pattern Analysis
```bash
# Analyze Claude-specific patterns
python run_claude_pattern_demo.py

# Pattern-based fuzzing
python run_pattern_fuzzer.py --use-claude-patterns
```

## ⚙️ Advanced Configuration

### Genetic Algorithm Parameters
```bash
aifuzzer \
  --genetic-algorithm-population 30 \
  --genetic-algorithm-mutation-rate 0.4 \
  --genetic-algorithm-crossover-rate 0.8
```

### Context Manipulation
```bash
aifuzzer \
  --context-manipulation-probability 0.7 \
  --token-manipulation-intensity 0.8
```

### Foot-in-the-Door Settings
```json
{
  "foot_in_door_settings": {
    "enabled": true,
    "escalation_levels": 3,
    "bridge_strategy": "gradual",
    "dynamic": true,
    "max_failures": 3
  }
}
```

## 📊 Results & Analytics

### Output Files
- **`results_[timestamp].jsonl`**: Complete fuzzing attempts with metadata
- **`successful_[timestamp].jsonl`**: Only successful jailbreaks
- **`technique_analytics_[timestamp].json`**: Effectiveness analysis by technique

### Sample Analytics Output
```
===== Technique Effectiveness Summary =====
Overall success rate: 12.50% (10/80)

Per-technique success rates:
- foot_in_door_technique: 94.00% (47/50)
- context_manipulation: 23.08% (3/13)
- token_manipulation: 18.75% (3/16)
- genetic_algorithm: 15.20% (19/125)
```

## 🏗️ Project Structure

```
AiFuzzer/
├── src/
│   ├── core/                 # Core fuzzing engine and techniques
│   │   ├── fuzzing_engine.py
│   │   ├── advanced_techniques.py
│   │   └── claude_patterns/  # Claude-specific patterns
│   ├── models/               # Model clients (Gemini, Claude)
│   ├── utils/                # Configuration and logging
│   └── cli/                  # Command-line interface
├── examples/                 # Configuration templates and prompts
├── docs/                     # Specialized technique documentation
│   ├── FOOT_IN_DOOR_TECHNIQUE.md
│   ├── CLAUDE_TO_CLAUDE_FUZZING.md
│   └── ADVANCED_TECHNIQUES.md
├── run_*.py                  # Specialized testing runners
└── tests/                    # Test suites
```

## 📈 Performance & Research

### Benchmark Results
- **Foot-in-the-Door Technique**: 94% average success rate
- **Multi-technique Combinations**: Up to 23% success rate improvement
- **Genetic Algorithm Optimization**: Continuous improvement over generations
- **Context Overflow**: Effective against context-aware safety measures

### Research Foundation
Based on cutting-edge research including:
- "A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts" (Ding et al., 2023)
- Advanced prompt engineering and adversarial ML techniques
- Multi-turn conversation analysis and psychological persuasion patterns

## 🔒 Security & Ethics

### Built-in Security Measures
- **API Key Protection**: Comprehensive .gitignore patterns prevent accidental exposure
- **Safe Configuration**: Template-based setup with placeholder keys
- **Output Filtering**: Automatic detection and flagging of sensitive content

### Ethical Usage Guidelines
This tool is intended **exclusively** for:
- ✅ Security research and AI safety improvement
- ✅ Red-team testing of LLM guardrails
- ✅ Academic research and responsible disclosure
- ✅ Model safety evaluation and enhancement

**Prohibited uses:**
- ❌ Deploying jailbreaks against production systems
- ❌ Generating harmful content for malicious purposes
- ❌ Circumventing safety measures in real applications

## 🤝 Contributing

We welcome contributions! Areas of particular interest:
- New jailbreak techniques and patterns
- Model support extensions
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Repository**: [https://github.com/tavgar/AiFuzzer](https://github.com/tavgar/AiFuzzer)
- **Issues**: [https://github.com/tavgar/AiFuzzer/issues](https://github.com/tavgar/AiFuzzer/issues)
- **Documentation**: See `docs/` directory for detailed technique guides

---

⚠️ **Responsible Research**: This tool is designed to improve AI safety through rigorous testing. Please use responsibly and in accordance with applicable terms of service and ethical guidelines.
