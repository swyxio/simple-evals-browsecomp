"""
Run async evaluation tasks in the background.
"""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('async_runner.log')
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

class AsyncTaskRunner:
    """Run async evaluation tasks"""
    
    def __init__(self, tasks_file: str, max_concurrent: int = 5):
        self.tasks_file = Path(tasks_file)
        self.max_concurrent = max_concurrent
        self.client = AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks_data = {}
        self.running_tasks = {}
    
    async def load_tasks(self) -> Dict:
        """Load tasks from the tasks file"""
        try:
            with open(self.tasks_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading tasks from {self.tasks_file}: {e}")
            return {}
    
    async def save_tasks(self):
        """Save updated tasks back to the file"""
        try:
            temp_file = self.tasks_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.tasks_data, f, indent=2)
            
            # Atomic write
            temp_file.replace(self.tasks_file)
            logger.debug(f"Saved tasks to {self.tasks_file}")
        except Exception as e:
            logger.error(f"Error saving tasks to {self.tasks_file}: {e}")
    
    async def update_task(self, task_id: str, updates: Dict):
        """Update task with new status/progress"""
        if task_id not in self.tasks_data:
            logger.warning(f"Task {task_id} not found in tasks data")
            return
        
        # Update task data
        self.tasks_data[task_id].update(updates)
        
        # Mark timestamps
        if 'status' in updates:
            status = updates['status']
            if status == 'running' and 'started_at' not in self.tasks_data[task_id]:
                self.tasks_data[task_id]['started_at'] = datetime.now().timestamp()
            elif status in ('completed', 'failed') and 'completed_at' not in self.tasks_data[task_id]:
                self.tasks_data[task_id]['completed_at'] = datetime.now().timestamp()
        
        # Periodically save to file
        if len(self.running_tasks) % 5 == 0:
            await self.save_tasks()
    
    async def run_evaluation(self, task_id: str, task_data: Dict):
        """Run a single evaluation task"""
        try:
            # Mark as running
            await self.update_task(task_id, {
                'status': 'running',
                'progress': 0
            })
            
            # Simulate work - replace with actual API calls
            total_steps = 10
            for i in range(total_steps):
                if shutdown_requested:
                    logger.info(f"Shutdown requested, cancelling task {task_id}")
                    await self.update_task(task_id, {
                        'status': 'cancelled',
                        'error': 'Shutdown requested'
                    })
                    return
                
                # Simulate work
                await asyncio.sleep(1)
                
                # Update progress
                progress = (i + 1) / total_steps * 100
                await self.update_task(task_id, {
                    'progress': progress
                })
            
            # Mark as completed
            await self.update_task(task_id, {
                'status': 'completed',
                'progress': 100,
                'results': {'sample_result': 'success'}
            })
            
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            await self.update_task(task_id, {
                'status': 'cancelled',
                'error': 'Task was cancelled'
            })
        except Exception as e:
            logger.error(f"Error in task {task_id}: {str(e)}", exc_info=True)
            await self.update_task(task_id, {
                'status': 'failed',
                'error': str(e)
            })
        finally:
            # Clean up
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            # Save final state
            await self.save_tasks()
    
    async def run(self):
        """Main task runner loop"""
        # Load tasks
        self.tasks_data = await self.load_tasks()
        if not self.tasks_data:
            logger.error("No tasks found to run")
            return
        
        logger.info(f"Loaded {len(self.tasks_data)} tasks")
        
        # Filter tasks that need to be run
        pending_tasks = {
            task_id: task for task_id, task in self.tasks_data.items()
            if task.get('status') not in ('completed', 'failed', 'cancelled')
        }
        
        if not pending_tasks:
            logger.info("No pending tasks to run")
            return
        
        logger.info(f"Starting {len(pending_tasks)} pending tasks (max {self.max_concurrent} concurrent)")
        
        # Start tasks with concurrency control
        for task_id, task_data in pending_tasks.items():
            if shutdown_requested:
                break
                
            # Wait for semaphore
            await self.semaphore.acquire()
            
            # Create and run task
            task = asyncio.create_task(self.run_evaluation(task_id, task_data))
            self.running_tasks[task_id] = task
            
            # Add callback to release semaphore when done
            def release_sem(task_id=task_id, task=task):
                self.semaphore.release()
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            
            task.add_done_callback(lambda _: release_sem())
        
        # Wait for all tasks to complete
        if self.running_tasks:
            logger.info("Waiting for tasks to complete...")
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        logger.info("All tasks completed")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global shutdown_requested
    logger.info("Shutdown signal received, waiting for tasks to complete...")
    shutdown_requested = True

async def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run async evaluation tasks")
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="async_results/latest_tasks.json",
        help="Path to tasks JSON file"
    )
    parser.add_argument(
        "--concurrent", 
        type=int, 
        default=5,
        help="Maximum concurrent tasks (default: 5)"
    )
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run task runner
    runner = AsyncTaskRunner(args.tasks, args.concurrent)
    await runner.run()

if __name__ == "__main__":
    asyncio.run(main())
