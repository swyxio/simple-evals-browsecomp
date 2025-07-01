# Async Evaluation System

This directory contains scripts for running evaluations asynchronously using OpenAI's API.

## Overview

The async evaluation system consists of three main components:

1. `async_simple_evals.py` - Submits evaluation tasks and saves their metadata
2. `run_async_tasks.py` - Runs the actual evaluations in the background
3. `poll_tasks.py` - Monitors the status of running evaluations

## Quick Start

1. **Submit Evaluation Tasks**
   ```bash
   python async_simple_evals.py --eval browsecomp --models gpt-4o,gpt-3.5-turbo --limit 10
   ```
   This will create tasks in `async_results/tasks_TIMESTAMP.json` and a symlink at `async_results/latest_tasks.json`.

2. **Run the Evaluations**
   ```bash
   python run_async_tasks.py --tasks async_results/latest_tasks.json --concurrent 5
   ```
   This will process up to 5 evaluations concurrently.

3. **Monitor Progress**
   ```bash
   # View current status
   python poll_tasks.py
   
   # Watch for updates
   python poll_tasks.py --watch --interval 5
   ```

## Detailed Usage

### async_simple_evals.py

Submit evaluation tasks to be run asynchronously.

```
usage: async_simple_evals.py [-h] [--eval EVAL] [--models MODELS] [--limit LIMIT] [--output-dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --eval EVAL           Evaluation to run (default: browsecomp)
  --models MODELS       Comma-separated list of models to evaluate (default: gpt-4o,gpt-3.5-turbo)
  --limit LIMIT         Maximum number of examples to evaluate per model (default: 10)
  --output-dir OUTPUT_DIR
                        Directory to save results (default: async_results)
```

### run_async_tasks.py

Run the actual evaluation tasks in the background.

```
usage: run_async_tasks.py [-h] [--tasks TASKS] [--concurrent CONCURRENT]

optional arguments:
  -h, --help            show this help message and exit
  --tasks TASKS         Path to tasks JSON file (default: async_results/latest_tasks.json)
  --concurrent CONCURRENT
                        Maximum concurrent tasks (default: 5)
```

### poll_tasks.py

Monitor the status of running evaluations.

```
usage: poll_tasks.py [-h] [--file FILE] [--watch] [--interval INTERVAL]

optional arguments:
  -h, --help         show this help message and exit
  --file FILE        Path to tasks JSON file (default: find latest in async_results/)
  --watch            Watch task status with auto-refresh
  --interval INTERVAL
                     Refresh interval in seconds (default: 5)
```

## Example Workflow

1. **Submit a batch of evaluations**
   ```bash
   python async_simple_evals.py --eval browsecomp --models gpt-4o,gpt-3.5-turbo --limit 20
   ```

2. **Start the task runner in a terminal session**
   ```bash
   # Run in the background with nohup
   nohup python run_async_tasks.py --concurrent 10 > async_runner.log 2>&1 &
   ```

3. **Monitor progress in another terminal**
   ```bash
   watch -n 5 python poll_tasks.py
   # Or with auto-refresh
   python poll_tasks.py --watch --interval 5
   ```

4. **Check logs**
   ```bash
   tail -f async_runner.log
   ```

## Implementation Notes

- Tasks are saved with metadata including model, status, timestamps, and progress
- The system supports graceful shutdown (SIGINT/SIGTERM)
- Concurrent execution is controlled by a semaphore
- Progress is periodically saved to disk
- Detailed logs are written to `async_runner.log`

## Troubleshooting

- If tasks get stuck, check the log file for errors
- Make sure you have sufficient API rate limits for concurrent requests
- The system is designed to be resilient to restarts - running the task runner again will pick up where it left off
