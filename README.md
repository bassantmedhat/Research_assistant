# Research Assistant Agent

A multi-agent research assistant that writes literature reviews. You give it a topic, and it plans the review, researches each section in parallel using arXiv (and other tools), checks the drafts, and puts them together into one Markdown report.

It uses an orchestrator–worker pattern built with **LangGraph**. Tools are exposed through **MCP** (Model Context Protocol).

## Architecture

```
                    ┌──────────────┐
      topic ──────▶ │   Planner    │  splits the topic into sections
                    └──────┬───────┘
                           │ sections
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
 ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
 │    Junior     │ │    Junior     │ │    Junior     │  research one section each
 │  Researcher   │ │  Researcher   │ │  Researcher   │  (tool-calling loop)
 └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
         └─────────────────┼─────────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Senior Researcher │  verifies citations and claims,
                 │    (verifier)     │  sends weak sections back
                 └─────────┬─────────┘
                           ▼
                    ┌──────────────┐
                    │ Synthesizer  │  merges sections into the final report
                    └──────────────┘

   Tools (through MCP): arXiv search · RAG retriever · memory cache
```

### Agents

| Agent | Role |
|---|---|
| **Planner** (orchestrator) | Turns the topic into a structured plan (a list of sections with a name and scope) using structured output. |
| **Junior Researchers** (workers) | One per section, running in parallel. Each runs a tool-calling loop (up to `MAX_STEPS`) to find papers, then writes its section with inline citations. |
| **Senior Researcher** (verifier) | Reviews each section: checks that citations are real and support the claims, and flags gaps. Sections that fail go back to a junior researcher. |
| **Synthesizer** | Joins the approved sections into one Markdown literature review. |

### Tools

| Tool | Purpose |
|---|---|
| **arXiv search** | Searches arXiv and returns titles, authors, dates and abstracts. |
| **RAG** | Retrieves relevant chunks from indexed papers or local documents so sections are based on full text, not just abstracts. |
| **Memory cache** | Caches search results and past findings so repeated queries don't hit the APIs again. |

## Status

| Component | Status |
|---|---|
| LangGraph orchestrator–worker pipeline | ✅ Implemented |
| Planner | ✅ Implemented |
| Junior researchers (parallel workers) | ✅ Implemented |
| arXiv search tool | ✅ Implemented |
| Synthesizer | ✅ Implemented |
| Senior researcher (verifier) | 🚧 Planned |
| MCP tool server | 🚧 Planned |
| RAG | 🚧 Planned |
| Memory cache | 🚧 Planned |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with the `qwen2.5:3b` model (the default local LLM)
- Python packages:

```bash
pip install langgraph langchain langchain-core langchain-community langchain-ollama langchain-tavily arxiv pydantic
```

## Usage

### Run locally

```bash
# 1. Start Ollama on the port the script expects
OLLAMA_HOST=127.0.0.1:11500 ollama serve &
ollama pull qwen2.5:3b

# 2. Run the pipeline
python orchestrator.py
```

The topic is set at the bottom of `orchestrator.py`:

```python
report = orchestrator_worker.invoke("Create a report on LLM medical imaging in echocardiography")
```

### Run on a SLURM cluster

`run.sh` requests a GPU, starts Ollama, pulls the model and runs the pipeline:

```bash
sbatch run.sh
```

Output goes to `output.log`, and Ollama's server log goes to `ollama.log`.

## Configuration

| Setting | Where | Default |
|---|---|---|
| LLM model | `init_chat_model(...)` in `orchestrator.py` | `qwen2.5:3b` via Ollama |
| Ollama address | `base_url` / `OLLAMA_HOST` | `127.0.0.1:11500` |
| Max tool steps per section | `MAX_STEPS` | `6` |
| arXiv results per query | `ArxivAPIWrapper(top_k_results=...)` | `5` |

To use Claude instead of the local model, uncomment the `ChatAnthropic` block in `orchestrator.py` and put your API keys in `.env`. If you enable web search, `TAVILY_API_KEY` goes in `.env` too.

## Project structure

```
Research_assistant/
├── orchestrator.py   # LangGraph pipeline: planner, workers, synthesizer
├── run.sh            # SLURM job script (Ollama + pipeline)
├── .env              # API keys (git-ignored)
└── README.md
```
