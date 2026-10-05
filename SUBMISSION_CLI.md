## Executive summary (read this first)

The active participant image accepts analyze and writes answer.json. This page documents its local interface; the [versioned official runtime](https://github.com/Agenthon-2026/track4-analysis-public/blob/1c744e1d6725340643a533f436517d72b53ca0e1/SUBMISSION_CLI.md) governs evaluated execution. Read the [Development runtime](https://github.com/Agenthon-2026/Agenthon2026-public/blob/bd01548e34d21fd660d88fd06157078fb25ece4e/docs/DEVELOPMENT-RUNTIME.md) for applied limits and House injection. Local historical cards and scorer3.1.0 do not certify current hosted behavior. No local diagnostic run establishes official quality or rank.

## Command

```bash
python -m baselines.strong_rag_baseline.cli analyze \
  --task /input/task.json --corpus /input/corpus --out /output/answer.json
```

Inputs are read-only. The CLI loads unit manifest admission from the task's parent directory. On success it writes UTF8 JSON and exits0. Errors remain errors; production output is not replaced with mock values. The runtime source, [Dockerfile](baselines/Dockerfile.strong-rag) and current shared answer schema define the interface. Required schema/roster, output-tree safety and actual per-unit resource/request/deadline limits remain binding.

### Container environment contract

The organizer injects the model route and identity during evaluated runs. Source: [House route](https://github.com/Agenthon-2026/Agenthon2026-public/blob/bd01548e34d21fd660d88fd06157078fb25ece4e/docs/HOUSE-MODEL.md).

| Variable | Consumed by this participant |
|---|---|
| `MODEL_ENDPOINT` | House route origin; client appends /v1/chat/completions |
| `MODEL_NAME` | Supplied model identity |
| `MODEL_TOKEN` | Per-unit bearer credential |

### Local settings and diagnostic mode

[config.py](baselines/strong_rag_baseline/config.py) is the single implementation source for local seed, retrieval, temperature, socket timeout and attempt settings. These defaults are engineering choices. The client conservatively caps attempted calls; the official service counts admitted generation requests. A one-attempt default or60second socket timeout does not prove full-roster deadline safety.

Use --mock only for offline wiring tests. Network-free mock runs are separate from evaluated model-API access. For current publication fields and actual output-tree restrictions, use the official runtime and shared descriptor schema. Keep credentials out of logs and public files; publication/upload requires the user's applicable authority.
