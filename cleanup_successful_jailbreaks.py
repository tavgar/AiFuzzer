#!/usr/bin/env python3
"""
Cleanup script to remove incorrectly classified successful jailbreaks where the response is empty.

This script fixes the issue where empty Claude responses were incorrectly marked as successful jailbreaks.
"""

import json
import sys
import os
from pathlib import Path
import argparse
import logging

def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def cleanup_successful_jailbreaks(file_path: str) -> None:
    """
    Clean up successful jailbreaks file by removing entries with empty responses.
    
    Args:
        file_path: Path to the successful jailbreaks file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Cleaning up file: {file_path}")
    
    # Ensure the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    # Read the current file
    entries = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    entries.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return
    
    # Count the total entries
    total_entries = len(entries)
    logger.info(f"Found {total_entries} entries in the file")
    
    # Filter out entries where response is empty
    corrected_entries = []
    removed_entries = []
    for entry in entries:
        response = entry.get("response", "")
        if response and response.strip():
            # Response is not empty, keep this entry
            corrected_entries.append(entry)
        else:
            # Response is empty, this was incorrectly classified
            removed_entries.append(entry)
    
    # Log the results
    logger.info(f"Filtered out {len(removed_entries)} entries with empty responses")
    logger.info(f"Keeping {len(corrected_entries)} entries with non-empty responses")
    
    # Create a backup of the original file
    backup_file = file_path + ".bak"
    try:
        with open(backup_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Created backup of original file: {backup_file}")
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return
    
    # Write the corrected entries back to the file
    try:
        with open(file_path, 'w') as f:
            for entry in corrected_entries:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Successfully updated file with corrected entries: {file_path}")
    except Exception as e:
        logger.error(f"Error writing corrected entries: {e}")
        return
    
    # Write removed entries to a separate file for reference
    removed_file = file_path + ".removed"
    try:
        with open(removed_file, 'w') as f:
            for entry in removed_entries:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Saved removed entries to: {removed_file}")
    except Exception as e:
        logger.error(f"Error writing removed entries: {e}")
    
    # Print a summary
    logger.info("Summary:")
    logger.info(f"  Original entries: {total_entries}")
    logger.info(f"  Entries removed:  {len(removed_entries)}")
    logger.info(f"  Entries kept:     {len(corrected_entries)}")
    logger.info(f"  Success rate:     {len(corrected_entries)/total_entries*100:.2f}% (corrected from {100:.2f}%)")

def main():
    """Main entry point for the application."""
    # Set up logging
    logger = setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Clean up successful jailbreaks file by removing entries with empty responses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the successful jailbreaks file",
        nargs="?",
        default="output/successful_20250518_174617.jsonl"
    )
    
    args = parser.parse_args()
    
    # Clean up the file
    cleanup_successful_jailbreaks(args.file_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
