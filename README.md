# story_teller

Keep long stories consistent.

story_teller reads a manuscript and builds a **Story Bible**: the characters, places, objects, events and relationships in the book, each linked to the sentence it came from and placed on the story's own timeline. It then flags continuity errors, such as a character whose eye colour changes, someone acting after their death, or two scenes that can't both be true. It understands flashbacks and dreams, so non-linear stories don't produce false alarms.

On top of that, the **Writers' Room** lets one or more LLMs write a long novel (30+ chapters) from an outline, using the Story Bible as shared memory and the continuity checker as a quality gate.

All content is family-friendly (children and teen ratings).

## How it works

| Component | What it does |
|---|---|
| **Archivist** | A fine-tuned open model that extracts Story Bible facts from each scene |
| **Scripty** | Checks the Story Bible for contradictions, using story time rather than chapter order |
| **Story Bible** | The story's knowledge graph, stored in Postgres with pgvector |
| **Brainstormer** | Turns an outline into a chapter plan, a planned Story Bible and a style sheet |
| **Showrunner** | Runs the writing loop: assigns chapters to writer models, then checks and revises each one |
| **Context Builder** | Chooses what each writer model sees from the Story Bible and recent chapters |
| **Editor** | Fixes flagged problems with targeted revisions |
| **Censor** | Enforces the content rating on everything uploaded or generated |

Archivist runs in a full MLOps lifecycle:
- versioned data (DVC)
- tracked training and a model registry (MLflow)
- evaluation gates in CI
- training on Apple Silicon (MLX) or CUDA (PEFT)
- orchestration with Airflow
- serving with llama.cpp or vLLM
- monitoring with Prometheus, Grafana and Evidently
- retraining when drift is detected

Agents and LLM clients can query a Story Bible through an MCP server.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design: data model, training and label strategy, evaluation, guardrails, security and user journeys.

## Repository layout

```
data/              DVC-tracked datasets: raw/ interim/ processed/ eval/
pipelines/         ingest, labeling, contradictions (planted eval cases)
training/          fine-tuning code and configs/
evaluation/        eval suites and the promotion gate
services/
  api/             FastAPI: projects, chapters, Story Bible queries
  extractor/       Archivist inference
  checker/         Scripty
  mcp/             MCP server
store/             Story Bible storage (Postgres + pgvector)
writers_room/      brainstormer/ showrunner/ context/ editor/
guardrails/        Censor and content policies
security/          auth, access control, rate limits, audit log
web/               UI: Story Bible, timeline, flag review
monitoring/        Prometheus, Grafana, drift reports
infra/             Docker Compose, Kubernetes, Terraform
tests/             unit/ integration/
notebooks/         exploration only
docs/              experiment write-ups and security docs
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env            # then set POSTGRES_PASSWORD
uv sync
pre-commit install
docker compose -f infra/docker/compose.yaml --env-file .env up -d   # Postgres, MLflow, Airflow
uv run pytest
```

Ingest the starter corpus (27 public-domain novels from Project Gutenberg) into scenes:

```bash
uv run python -m pipelines.ingest.run      # or trigger the ingest_gutenberg DAG in Airflow (localhost:8080)
uv run dvc add data/raw/gutenberg data/processed && uv run dvc push
```

No shared DVC remote is configured yet. Set your own before pushing or pulling data, e.g.
`uv run dvc remote add --local -d storage /path/to/dvc-storage`.

## License

MIT. See [LICENSE](LICENSE).

Gold entity labels come from [LitBank](https://github.com/dbamman/litbank) (Bamman et al., 2019–2020),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Novel texts are from Project Gutenberg
and are in the public domain.
