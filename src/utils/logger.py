"""
Logging configuration for the LLM Jailbreak Fuzzer.
"""

import logging
import sys
import datetime
from pathlib import Path
from typing import Optional

def configure_logging(log_dir: Optional[Path] = None,
                     level: int = logging.INFO,
                     log_filename: Optional[str] = None) -> None:
    """
    Configure logging for the application with specific log level.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_filename: Custom log filename (default: timestamped)
    """
    # Call setup_logging with the appropriate parameters
    verbose = level <= logging.DEBUG
    setup_logging(log_dir, verbose, log_filename)


def setup_logging(log_dir: Optional[Path] = None, 
                  verbose: bool = False, 
                  log_filename: Optional[str] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_dir: Directory to store log files
        verbose: Whether to enable verbose logging
        log_filename: Custom log filename (default: timestamped)
    """
    # Create a timestamped log file name if not provided
    if log_dir is None:
        log_dir = Path("logs")
    
    log_dir.mkdir(exist_ok=True, parents=True)
    
    if log_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"aifuzzer_{timestamp}.log"
    
    log_file = log_dir / log_filename
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure file handler
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Configure console handler
    console_formatter = logging.Formatter(
        "%(levelname)s: %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    # Log initial message
    logging.info(f"Logging initialized. Log file: {log_file}")
    if verbose:
        logging.debug("Verbose logging enabled")


class ResultLogger:
    """
    Specialized logger for fuzzing results.
    
    This logger records the results of each fuzzing attempt, including:
    - The prompt used
    - The target model's response
    - The evaluation result
    - Metadata about the attempt
    - Analytics about technique effectiveness (if enabled)
    """
    
    def __init__(self, output_dir: Path, save_technique_analytics: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"results_{self.session_timestamp}.jsonl"
        self.success_file = self.output_dir / f"successful_{self.session_timestamp}.jsonl"
        self.analytics_file = self.output_dir / f"technique_analytics_{self.session_timestamp}.json"
        self.logger = logging.getLogger(__name__)
        self.save_technique_analytics = save_technique_analytics
        
        # Technique effectiveness tracking
        self.technique_stats = {
            "genetic_algorithm": {"attempts": 0, "successes": 0},
            "token_manipulation": {"attempts": 0, "successes": 0},
            "context_manipulation": {"attempts": 0, "successes": 0},
            "multi_layered_attack": {"attempts": 0, "successes": 0},
            "homoglyph_substitution": {"attempts": 0, "successes": 0},
            "zero_width_spaces": {"attempts": 0, "successes": 0},
            "role_playing": {"attempts": 0, "successes": 0},
            "hypothetical_framework": {"attempts": 0, "successes": 0},
            "confusion_technique": {"attempts": 0, "successes": 0},
            "foot_in_door": {"attempts": 0, "successes": 0},
            "basic_prompt": {"attempts": 0, "successes": 0},
        }
        
        # Initialize files with headers if needed
        for file_path in [self.results_file, self.success_file]:
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    pass  # Just create empty file
        
        self.logger.info(f"Result logger initialized. Results file: {self.results_file}")
        self.logger.info(f"Successful attempts will be saved to: {self.success_file}")
        if self.save_technique_analytics:
            self.logger.info(f"Technique analytics will be saved to: {self.analytics_file}")
    
    def log_attempt(self, attempt_data: dict) -> None:
        """
        Log a fuzzing attempt.
        
        Args:
            attempt_data: Dictionary containing attempt details
        """
        import json
        
        try:
            # Update technique statistics if analytics are enabled
            if self.save_technique_analytics and "techniques" in attempt_data:
                self._update_technique_stats(attempt_data)
            
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(attempt_data) + '\n')
            
            # If this was a successful attempt, also log to success file
            if attempt_data.get('success', False):
                with open(self.success_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(attempt_data) + '\n')
                self.logger.info(f"Successful jailbreak found! ID: {attempt_data.get('id', 'unknown')}")
                
                # Save updated analytics after each successful attempt
                if self.save_technique_analytics:
                    self._save_analytics()
                    
        except Exception as e:
            self.logger.error(f"Error logging attempt: {e}")
    
    def _update_technique_stats(self, attempt_data: dict) -> None:
        """
        Update technique statistics based on attempt data.
        
        Args:
            attempt_data: Dictionary containing attempt details
        """
        success = attempt_data.get('success', False)
        techniques = attempt_data.get('techniques', [])
        
        # If no specific techniques are listed, count as basic prompt
        if not techniques:
            self.technique_stats['basic_prompt']['attempts'] += 1
            if success:
                self.technique_stats['basic_prompt']['successes'] += 1
            return
            
        # Update stats for each technique used
        for technique in techniques:
            if technique in self.technique_stats:
                self.technique_stats[technique]['attempts'] += 1
                if success:
                    self.technique_stats[technique]['successes'] += 1
    
    def _save_analytics(self) -> None:
        """Save technique analytics to file."""
        import json
        
        try:
            # Calculate success rates and additional metrics
            analytics = {
                "technique_stats": self.technique_stats,
                "summary": {
                    "total_attempts": sum(tech['attempts'] for tech in self.technique_stats.values()),
                    "total_successes": sum(tech['successes'] for tech in self.technique_stats.values()),
                    "technique_success_rates": {}
                }
            }
            
            # Calculate success rate for each technique
            for technique, stats in self.technique_stats.items():
                if stats['attempts'] > 0:
                    success_rate = (stats['successes'] / stats['attempts']) * 100
                    analytics['summary']['technique_success_rates'][technique] = f"{success_rate:.2f}%"
                else:
                    analytics['summary']['technique_success_rates'][technique] = "N/A"
            
            # Add timestamp
            analytics['timestamp'] = datetime.datetime.now().isoformat()
            
            # Write to file
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving technique analytics: {e}")
    
    def log_technique_usage(self, technique_name: str, success: bool) -> None:
        """
        Log usage of a specific technique.
        
        Args:
            technique_name: Name of the technique used
            success: Whether the attempt was successful
        """
        if not self.save_technique_analytics:
            return
            
        if technique_name in self.technique_stats:
            self.technique_stats[technique_name]['attempts'] += 1
            if success:
                self.technique_stats[technique_name]['successes'] += 1
        
        # Save updated analytics occasionally
        if sum(tech['attempts'] for tech in self.technique_stats.values()) % 10 == 0:
            self._save_analytics()
    
    def get_technique_summary(self) -> str:
        """
        Get a summary of technique effectiveness.
        
        Returns:
            A formatted string containing technique effectiveness summary
        """
        if not self.save_technique_analytics:
            return "Technique analytics not enabled"
            
        summary = "\n===== Technique Effectiveness Summary =====\n"
        
        total_attempts = sum(tech['attempts'] for tech in self.technique_stats.values())
        total_successes = sum(tech['successes'] for tech in self.technique_stats.values())
        
        if total_attempts == 0:
            return summary + "No data available yet."
            
        overall_success_rate = (total_successes / total_attempts) * 100
        summary += f"Overall success rate: {overall_success_rate:.2f}% ({total_successes}/{total_attempts})\n\n"
        
        summary += "Per-technique success rates:\n"
        for technique, stats in self.technique_stats.items():
            if stats['attempts'] > 0:
                success_rate = (stats['successes'] / stats['attempts']) * 100
                summary += f"- {technique}: {success_rate:.2f}% ({stats['successes']}/{stats['attempts']})\n"
        
        return summary
