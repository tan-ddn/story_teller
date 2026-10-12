# story_teller

A story-continuity engine for long-form fiction. It extracts a character/location/event graph from manuscripts and flags continuity errors. The extraction model is fine-tuned in-house and runs inside a full MLOps lifecycle: versioned data, tracked training, CI eval gates, serving, monitoring and drift-triggered retraining.

See [ARCHITECTURE.md](ARCHITECTURE.md).

## Layout
```
data/            DVC-tracked: raw/ interim/ processed/ eval/ (gold set, frozen per version)
pipelines/       Prefect flows: ingest, labeling (silver labels), contradictions (planted eval)
training/        LoRA fine-tuning + configs/
evaluation/      eval suites, champion-vs-challenger, promotion gate
services/
  api/           FastAPI: projects, chapters, graph queries
  extractor/     Archivist: model inference (vLLM / ONNX), loads model@prod from MLflow
  checker/       Scripty: continuity rules + model-scored contradictions
  mcp/           MCP server exposing story-graph tools
store/           GraphStore interface over Postgres + pgvector (Story Bible)
writers_room/    phase 2: brainstormer/ showrunner/ context/ editor/
security/        authN/Z, row-level security helpers, rate limits, audit log
guardrails/      Censor: family-friendly only (children/teen); policies/*.yaml
web/             thin UI (graph, timeline, flag review)
monitoring/      prometheus/ grafana/ drift/ (Evidently)
infra/           docker/ (compose), k8s/ (Helm), terraform/
.github/workflows CI: lint, tests, tiny-model eval gate, image build
tests/           unit/ integration/
notebooks/       exploration only, nothing imported from here
docs/            results, experiment write-ups; security/ (threat model, incident runbook)
```

## Security
Content is family-friendly only (children/teen ratings). User manuscripts are used for training
only with per-project opt-in. See ARCHITECTURE.md §11 (guardrails) and §12 (security and privacy).

- Never commit secrets. Local config goes in `.env`, which is git-ignored. Copy `.env.example`.
- Install the pre-commit hooks (gitleaks, ruff) before your first commit: `pre-commit install`.
- Model weights: `safetensors` only, pinned revisions, no `trust_remote_code`.

## License
MIT. See [LICENSE](LICENSE).
