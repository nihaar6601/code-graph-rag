# Code Graph RAG

A Python project for asking questions about Java code using its class and method relationships. It parses a repository with Tree-sitter, builds a code graph, and retrieves relevant methods along with their dependencies before sending context to an LLM.

The goal is to compare graph-based retrieval with traditional RAG and see how much useful context can fit within a token budget.

## Features

- Extracts classes, nested classes, methods, constructors, parameters, and source locations.
- Indexes multiple Java files and resolves supported calls across files.
- Builds `CONTAINS` and `CALLS` relationships, keeping ambiguous or external calls unresolved.
- Supports keyword search, local embeddings, hybrid search, and graph expansion.
- Selects complete code snippets within a configurable context token budget.
- Generates optional LLM answers with file and line citations.
- Exports graphs, retrieved context, answers, and evaluation reports.

## How it works

1. Tree-sitter parses the Java files and extracts code entities.
2. NetworkX stores the entities and their relationships as a graph.
3. Retrieval finds methods related to the question and can follow their calls.
4. Context optimization selects snippets that fit the token budget.
5. An optional LLM answers using the selected code and cites its sources.

There are five retrieval strategies: `traditional` (embedded text chunks), `structure` (keyword search over methods), `graph` (structure search with call expansion), `hybrid` (keyword + embeddings), and `hybrid_graph` (hybrid search with call expansion).

## Tech stack

Python 3.11+, Tree-sitter Java, NetworkX, Sentence Transformers (`all-MiniLM-L6-v2`), NumPy, and tiktoken. Answer generation supports Groq, OpenAI, and compatible endpoints. The project runs through a CLI; no database or frontend is required.

## Setup

```bash
git clone https://github.com/nihaar6601/code-graph-rag.git
cd code-graph-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For embedding-based strategies, download the model once:

```bash
python embeddings.py --download-model
```

Embeddings run locally after download. The `structure` and `graph` strategies do not need the embedding model. The tokenizer may download its vocabulary on first use.

## Usage

### Build a Java code graph

```bash
python main.py examples/repository/ --json graph.json
```

Replace the example directory with your Java repository or pass individual `.java` files. Add `--verbose` to see extracted entities and calls.

### Retrieve code for a question

```bash
python main.py examples/shop/ \
  --query "How does checkout handle a rejected payment?" \
  --strategy graph --max-tokens 2000 \
  --output-dir output/checkout
```

This saves `context.txt` and `retrieval.json`. Use `--strategy hybrid_graph` to include embedding search, or `--optimization relevance` to try relevance-based context selection. `--depth`, `--top-k`, and `--include-callers` control graph retrieval.

### Generate an answer with Groq

Set `GROQ_API_KEY` in your shell environment, then run:

```bash
python main.py examples/shop/ \
  --query "How does checkout handle a rejected payment?" \
  --strategy graph --answer \
  --provider groq --llm-model openai/gpt-oss-120b \
  --max-tokens 2000 --max-output-tokens 512 \
  --output-dir output/answer
```

`--answer` enables API requests and may incur costs. API keys are read from environment variables; `.env` files are not loaded automatically. Without `--answer`, indexing and retrieval make no LLM requests.

To try the answer workflow without an API key, replace the Groq options with:

```bash
--provider mock --mock-response examples/mock_checkout_answer.json
```

The mock uses a fixed response, so it only checks the workflow. Answer output includes `answer.txt` and `answer.json`, alongside the retrieved context. For connection issues, `python llm_transport.py` shows local TLS/proxy diagnostics without making network requests.

## Tests and evaluation

```bash
python -m unittest discover -s tests -v

python evaluate.py examples/repository/ \
  --benchmark benchmarks/payment_queries.json \
  --max-tokens 2000 --top-k 1 --depth 1 \
  --json output/evaluation.json
```

Tests cover Java parsing, cross-file resolution, retrieval, context budgets, answer validation, and mocked HTTP failures. `evaluate.py` compares retrieval; `evaluate_answers.py` handles answer benchmarks and budget controls. See [the answer benchmark notes](benchmarks/ANSWER_BENCHMARK.md) and use `--help` for additional options.

## Current limitations

- Java only. Call resolution is based on supported static patterns, not a full Java compiler. Inheritance, runtime dispatch, wildcard/static imports, and ambiguous overloads are not fully resolved.
- Constructors are extracted from source; implicit default constructors are not synthesized.
- Whole-method selection and bounded graph traversal can leave out relevant code.
- Context token counts are estimates for the configured tokenizer, not exact provider billing counts.
- Citation checks validate source references, not whether an answer is factually correct. Mock runs and automated grading do not establish real LLM answer quality or cost savings.

Example repositories are in `examples/`, benchmark questions are in `benchmarks/`, and generated files and model caches are ignored by Git.
