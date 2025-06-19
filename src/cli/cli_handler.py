"""
Command-line interface handler for the LLM Jailbreak Fuzzer.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rich_print

logger = logging.getLogger(__name__)
console = Console()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Jailbreak Fuzzer - A tool to test LLM safety measures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Keep track of default values to identify explicitly provided args later
    defaults = {}
    
    # API keys
    parser.add_argument(
        "--gemini-api-key", 
        help="API key for Google's Gemini model (can also use GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--claude-api-key", 
        help="API key for Anthropic's Claude model (can also use CLAUDE_API_KEY env var)"
    )
    
    # Model settings
    parser.add_argument(
        "--gemini-model", 
        help="Gemini model version to use"
    )
    parser.add_argument(
        "--claude-model", 
        help="Claude model version to use"
    )
    
    # Fuzzing parameters
    parser.add_argument(
        "--max-attempts", 
        type=int, 
        default=50,
        help="Maximum number of fuzzing attempts"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=5,
        help="Number of prompts to generate in each batch"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.9,
        help="Temperature for generation (higher = more random)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30,
        help="Timeout in seconds for API calls"
    )
    
    # Advanced techniques
    parser.add_argument(
        "--use-advanced-techniques",
        action="store_true",
        default=True,
        help="Enable advanced fuzzing techniques"
    )
    parser.add_argument(
        "--no-advanced-techniques",
        action="store_false",
        dest="use_advanced_techniques",
        help="Disable advanced fuzzing techniques"
    )
    parser.add_argument(
        "--genetic-algorithm-population",
        type=int,
        default=20,
        help="Population size for genetic algorithm"
    )
    parser.add_argument(
        "--genetic-algorithm-mutation-rate",
        type=float,
        default=0.3,
        help="Mutation rate for genetic algorithm (0.0-1.0)"
    )
    parser.add_argument(
        "--genetic-algorithm-crossover-rate",
        type=float,
        default=0.7,
        help="Crossover rate for genetic algorithm (0.0-1.0)"
    )
    parser.add_argument(
        "--token-manipulation-intensity",
        type=float,
        default=0.5,
        help="Intensity of token manipulation techniques (0.0-1.0)"
    )
    parser.add_argument(
        "--context-manipulation-probability",
        type=float,
        default=0.6,
        help="Probability of applying context manipulation (0.0-1.0)"
    )
    
    # Context Overflow technique
    parser.add_argument(
        "--use-context-overflow",
        action="store_true",
        help="Enable context overflow technique"
    )
    parser.add_argument(
        "--context-overflow-intensity",
        type=float,
        default=0.7,
        help="Intensity of context overflow (how much of context window to fill) (0.0-1.0)"
    )
    parser.add_argument(
        "--context-overflow-strategy",
        type=str,
        default="distributed",
        choices=["beginning", "end", "distributed", "sandwich"],
        help="Strategy for positioning overflow content"
    )
    parser.add_argument(
        "--model-context-size",
        type=int,
        help="Override the context size for the target model (in tokens)"
    )
    
    parser.add_argument(
        "--save-technique-analytics",
        action="store_true",
        default=True,
        help="Save analytics about which techniques were most effective"
    )
    
    # Input/output
    parser.add_argument(
        "--initial-prompts", 
        type=str,
        help="Path to a file containing initial prompts to start with"
    )
    parser.add_argument(
        "--target-behaviors", 
        type=str,
        help="Path to a file containing target behaviors to test for"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory to store output files"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="Directory to store log files"
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    # Runtime options
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--save-all-prompts", 
        action="store_true",
        help="Save all prompts to the output directory"
    )
    # Store all default values from all arguments
    for action in parser._actions:
        if action.default is not None and action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
    
    # Parse arguments
    args = parser.parse_args()
    
    # Attach defaults dictionary to the args object
    args._defaults = defaults
    
    return args


def run_interactive_mode(config: Any) -> None:
    """Run the fuzzer in interactive mode."""
    logger.info("Starting interactive mode")
    console.print("[bold green]LLM Jailbreak Fuzzer - Interactive Mode[/bold green]")
    console.print()
    
    # Check for API keys
    if not config.gemini_api_key:
        config.gemini_api_key = Prompt.ask(
            "Enter your Gemini API key", 
            password=True,
            default=""
        )
    
    if not config.claude_api_key:
        config.claude_api_key = Prompt.ask(
            "Enter your Claude API key", 
            password=True,
            default=""
        )
    
    # Configure fuzzing parameters interactively
    console.print("[bold]Configure Fuzzing Parameters[/bold]")
    
    config.max_attempts = int(Prompt.ask(
        "Maximum number of fuzzing attempts",
        default=str(config.max_attempts)
    ))
    
    config.batch_size = int(Prompt.ask(
        "Batch size (prompts per iteration)",
        default=str(config.batch_size)
    ))
    
    config.temperature = float(Prompt.ask(
        "Temperature for generation",
        default=str(config.temperature)
    ))
    
    # Configure advanced techniques
    console.print("\n[bold]Advanced Techniques[/bold]")
    
    use_advanced = Confirm.ask(
        "Enable advanced fuzzing techniques?",
        default=config.use_advanced_techniques
    )
    config.use_advanced_techniques = use_advanced
    
    if use_advanced:
        console.print("[bold cyan]Configuring advanced techniques:[/bold cyan]")
        
        config.genetic_algorithm_population = int(Prompt.ask(
            "Genetic algorithm population size",
            default=str(config.genetic_algorithm_population)
        ))
        
        config.genetic_algorithm_mutation_rate = float(Prompt.ask(
            "Genetic algorithm mutation rate (0.0-1.0)",
            default=str(config.genetic_algorithm_mutation_rate)
        ))
        
        config.genetic_algorithm_crossover_rate = float(Prompt.ask(
            "Genetic algorithm crossover rate (0.0-1.0)",
            default=str(config.genetic_algorithm_crossover_rate)
        ))
        
        config.token_manipulation_intensity = float(Prompt.ask(
            "Token manipulation intensity (0.0-1.0)",
            default=str(config.token_manipulation_intensity)
        ))
        
        config.context_manipulation_probability = float(Prompt.ask(
            "Context manipulation probability (0.0-1.0)",
            default=str(config.context_manipulation_probability)
        ))
        
        # Context Overflow configuration
        config.use_context_overflow = Confirm.ask(
            "Enable context overflow technique?",
            default=config.use_context_overflow
        )
        
        if config.use_context_overflow:
            config.context_overflow_intensity = float(Prompt.ask(
                "Context overflow intensity (0.0-1.0)",
                default=str(config.context_overflow_intensity)
            ))
            
            strategies = ["beginning", "end", "distributed", "sandwich"]
            strategy_idx = Prompt.ask(
                "Context overflow strategy",
                choices=strategies,
                default=config.context_overflow_strategy
            )
            config.context_overflow_strategy = strategy_idx
        
        config.save_technique_analytics = Confirm.ask(
            "Save analytics about technique effectiveness?",
            default=config.save_technique_analytics
        )
    else:
        console.print("[yellow]Advanced techniques disabled. Using basic fuzzing only.[/yellow]")
    
    # Ask for target behaviors
    console.print("[bold]Target Behaviors[/bold]")
    console.print("Enter behaviors you want to test for (one per line, empty line to finish):")
    
    behaviors = []
    while True:
        behavior = console.input("> ")
        if not behavior:
            break
        behaviors.append(behavior)
    
    if behaviors:
        config.target_behaviors = behaviors
    
    # Start fuzzing
    console.print()
    start = Confirm.ask("Start fuzzing now?")
    
    if start:
        from src.core.fuzzing_engine import FuzzingEngine
        engine = FuzzingEngine(config)
        engine.run()
    else:
        console.print("Exiting...")
        sys.exit(0)


def display_results(results: List[Dict[str, Any]]) -> None:
    """Display fuzzing results in a formatted table."""
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return
    
    # Create a table for results
    table = Table(title="Fuzzing Results")
    table.add_column("ID", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Prompt Sample", style="blue", no_wrap=False)
    table.add_column("Response Sample", style="yellow", no_wrap=False)
    
    for result in results:
        # Truncate long prompt and response for display
        prompt_sample = (result.get("prompt", "")[:50] + "...") if len(result.get("prompt", "")) > 50 else result.get("prompt", "")
        response_sample = (result.get("response", "")[:50] + "...") if len(result.get("response", "")) > 50 else result.get("response", "")
        
        success_str = "✓" if result.get("success", False) else "✗"
        success_style = "green" if result.get("success", False) else "red"
        
        table.add_row(
            str(result.get("id", "")),
            f"[{success_style}]{success_str}[/{success_style}]",
            prompt_sample,
            response_sample
        )
    
    console.print(table)
    
    # Summary statistics
    successful = sum(1 for r in results if r.get("success", False))
    console.print(f"[bold]Summary:[/bold] {successful}/{len(results)} successful jailbreaks")
