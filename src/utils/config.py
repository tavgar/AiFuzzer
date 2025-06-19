"""
Configuration management for the LLM Jailbreak Fuzzer.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class FuzzerConfig:
    """Configuration class for the fuzzer."""
    # API keys
    gemini_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None
    
    # Models
    gemini_model: str = "gemini-2.5-pro-preview-0506"
    claude_model: str = "claude-3-7-sonnet-20250219"
    
    # Fuzzing parameters
    max_attempts: int = 50
    batch_size: int = 5
    temperature: float = 0.9
    timeout: int = 30
    
    # Paths
    log_dir: Path = Path("logs")
    output_dir: Path = Path("output")
    
    # Learning parameters
    learning_rate: float = 0.1
    memory_size: int = 100
    
    # Advanced techniques parameters
    use_advanced_techniques: bool = True
    genetic_algorithm_population: int = 20
    genetic_algorithm_mutation_rate: float = 0.3
    genetic_algorithm_crossover_rate: float = 0.7
    token_manipulation_intensity: float = 0.5
    context_manipulation_probability: float = 0.6
    
    # Context Overflow parameters
    use_context_overflow: bool = False
    context_overflow_intensity: float = 0.7
    context_overflow_strategy: str = "distributed"
    
    # Foot-in-the-door parameters
    use_foot_in_door: bool = False
    foot_in_door_escalation_levels: int = 3
    foot_in_door_bridge_strategy: str = "gradual"
    foot_in_door_persistence: float = 0.7
    foot_in_door_dynamic: bool = False
    foot_in_door_max_failures: int = 3
    foot_in_door_simplification_factor: float = 0.5
    
    model_context_sizes: Dict[str, int] = field(default_factory=lambda: {
        # Gemini models
        "gemini-1.5-pro": 1048576,
        "gemini-1.5-flash": 1048576,
        "gemini-2.5-pro-preview-0506": 32768,
        "gemini-2.5-pro": 1048576,
        "gemini-2.5-flash": 1048576,
        
        # Claude models
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-7-sonnet-20250219": 200000,
        "claude-neptune": 200000
    })
    
    def is_claude_model(self, model_name: str) -> bool:
        """Determine if a model is from Anthropic's Claude family."""
        return model_name.startswith('claude-')
    
    def is_gemini_model(self, model_name: str) -> bool:
        """Determine if a model is from Google's Gemini family."""
        return model_name.startswith('gemini-')
    
    def get_model_family(self, model_name: str) -> str:
        """Get the family name for a model."""
        if self.is_claude_model(model_name):
            return "claude"
        elif self.is_gemini_model(model_name):
            return "gemini"
        else:
            return "unknown"
    
    # Additional settings
    verbose: bool = False
    save_all_prompts: bool = True
    save_technique_analytics: bool = True
    
    # Runtime settings
    initial_prompts: list = field(default_factory=list)
    target_behaviors: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FuzzerConfig':
        """Create config from dictionary."""
        # Convert string paths back to Path objects
        for key in ['log_dir', 'output_dir']:
            if key in config_dict and isinstance(config_dict[key], str):
                config_dict[key] = Path(config_dict[key])
        
        return cls(**{k: v for k, v in config_dict.items() 
                    if k in cls.__annotations__})


def load_config(config_path: Optional[str] = None) -> FuzzerConfig:
    """Load configuration from file."""
    if not config_path:
        return FuzzerConfig()
    
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return FuzzerConfig.from_dict(config_dict)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return FuzzerConfig()


def save_config(config: FuzzerConfig, config_path: str) -> None:
    """Save configuration to file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")


def initialize_config(args) -> FuzzerConfig:
    """Initialize configuration from arguments and/or config file."""
    # Track which arguments were explicitly provided on the command line
    parser_defaults = getattr(args, '_defaults', {})
    explicitly_provided = {k: v for k, v in vars(args).items() if k not in parser_defaults}
    
    # First load from config file if specified
    if hasattr(args, 'config') and args.config:
        config = load_config(args.config)
    else:
        config = FuzzerConfig()
    
    # Override with environment variables
    if os.environ.get('GEMINI_API_KEY'):
        config.gemini_api_key = os.environ.get('GEMINI_API_KEY')
    
    if os.environ.get('CLAUDE_API_KEY'):
        config.claude_api_key = os.environ.get('CLAUDE_API_KEY')
    
    # Override only with explicitly provided command line arguments
    for key, value in explicitly_provided.items():
        config_key = key.replace('-', '_')  # Convert dashes to underscores
        if value is not None and hasattr(config, config_key):
            setattr(config, config_key, value)
    
    # Ensure directory paths are Path objects, not strings
    if isinstance(config.log_dir, str):
        config.log_dir = Path(config.log_dir)
    if isinstance(config.output_dir, str):
        config.output_dir = Path(config.output_dir)
    
    # Create directories if they don't exist
    config.log_dir.mkdir(exist_ok=True, parents=True)
    config.output_dir.mkdir(exist_ok=True, parents=True)
    
    return config
