"""
Poll the status of async evaluation tasks.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime, timedelta

def load_tasks(tasks_file: Optional[str] = None) -> Tuple[Dict, Path]:
    """Load tasks from a JSON file"""
    if tasks_file is None:
        # Try to find the latest tasks file
        output_dir = Path("async_results")
        if not output_dir.exists():
            print(f"Error: Directory {output_dir} does not exist")
            sys.exit(1)
            
        # Look for latest_tasks.json symlink
        latest_file = output_dir / "latest_tasks.json"
        if latest_file.exists():
            tasks_file = latest_file.resolve()
        else:
            # Find the most recent tasks file
            task_files = list(output_dir.glob("tasks_*.json"))
            if not task_files:
                print("No task files found in", output_dir)
                sys.exit(1)
            tasks_file = max(task_files, key=os.path.getmtime)
    else:
        tasks_file = Path(tasks_file)
    
    try:
        with open(tasks_file, 'r') as f:
            tasks = json.load(f)
        return tasks, tasks_file
    except json.JSONDecodeError as e:
        print(f"Error reading {tasks_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string"""
    if seconds is None:
        return "N/A"
    return str(timedelta(seconds=int(seconds)))

def print_task_summary(tasks: Dict, tasks_file: Path):
    """Print a summary of all tasks"""
    print(f"\n=== Task Summary ({len(tasks)} tasks) ===")
    print(f"Task file: {tasks_file}\n")
    
    # Group tasks by status
    status_counts = {}
    for task in tasks.values():
        status = task.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Print status summary
    print("Status Summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    # Print task details
    print("\nTask Details:")
    for task_id, task in tasks.items():
        print(f"\nTask ID: {task_id}")
        print(f"  Model: {task.get('model_name', 'N/A')}")
        print(f"  Status: {task.get('status', 'N/A')}")
        
        if 'progress' in task:
            print(f"  Progress: {task['progress']:.1f}%")
        
        if 'created_at' in task:
            created = datetime.fromtimestamp(task['created_at']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  Created: {created}")
        
        if 'started_at' in task and task['started_at']:
            started = datetime.fromtimestamp(task['started_at']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  Started: {started}")
        
        if 'completed_at' in task and task['completed_at']:
            completed = datetime.fromtimestamp(task['completed_at']).strftime('%Y-%m-%d %H:%M:%S')
            duration = task['completed_at'] - task.get('started_at', task['completed_at'])
            print(f"  Completed: {completed} (duration: {format_duration(duration)})")
        
        if 'error' in task and task['error']:
            print(f"  Error: {task['error']}")

def watch_tasks(tasks_file: Optional[str] = None, interval: int = 5):
    """Watch task status with auto-refresh"""
    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Load and display tasks
            tasks, current_file = load_tasks(tasks_file)
            print_task_summary(tasks, current_file)
            
            # Check if all tasks are done
            all_done = all(
                task.get('status') in ('completed', 'failed') 
                for task in tasks.values()
            )
            
            if all_done:
                print("\nAll tasks completed!")
                break
                
            print(f"\nRefreshing in {interval} seconds (Ctrl+C to stop)...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopped watching tasks.")

def main():
    parser = argparse.ArgumentParser(description="Poll status of async evaluation tasks")
    parser.add_argument(
        "--file", 
        type=str, 
        help="Path to tasks JSON file (default: find latest in async_results/)"
    )
    parser.add_argument(
        "--watch", 
        action="store_true",
        help="Watch task status with auto-refresh"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )
    args = parser.parse_args()
    
    if args.watch:
        watch_tasks(args.file, args.interval)
    else:
        tasks, tasks_file = load_tasks(args.file)
        print_task_summary(tasks, tasks_file)
        print("\nUse '--watch' flag to auto-refresh status.")

if __name__ == "__main__":
    main()
