# Simple Evals - BrowseComp

This is a simplified version of the `simple-evals` repository containing only the BrowseComp evaluation.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

To run the BrowseComp evaluation:

```bash
python simple_evals.py --eval=browsecomp --model=<your-model-name>
```

## Available Models

To list available models:

```bash
python simple_evals.py --list-models
```

## Example

```bash
python simple_evals.py --eval=browsecomp --model=gpt-4
```

## Environment Variables

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-api-key-here
```
