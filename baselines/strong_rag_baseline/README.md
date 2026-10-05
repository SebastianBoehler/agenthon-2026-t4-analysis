## Executive summary (read this first)

This participant uses BM25, a lexical relevance score, to retrieve bounded pre-cutoff excerpts for each entity. Manifest labels determine document admission. The supplied model produces predictions and quotes; spans are located in original text. Missing quote matches are dropped, and invalid labels or intervals produce an error rather than an invented value. Exact matching protects citation lineage but can reduce completeness; it does not prove factual support. Model quality and total House latency remain unmeasured locally.

Start from the [root entrypoint](../../README.md) and [local interface](../../SUBMISSION_CLI.md). The task prompt determines units, horizon and ranking direction. Document names add no admission filter beyond the verified manifest. The implementation does not emit optional submitted_reasons.

## Source map

| Module | Role |
|---|---|
| admission.py | Manifest labels, paths and source hashes |
| indexer.py / bounded_spans.py | Full source text and overlapping bounded offsets |
| retriever.py | BM25 and pre-cutoff filtering |
| prompts.py | Supplied task semantics and per-entity evidence |
| client.py / config.py | Model route and explicit local request settings |
| span_finder.py / agent.py | Exact source binding and model value validation |
| formatter.py / cli.py | Complete answer assembly and analyze command |

Top-K, chunk size, exact matching, serial scheduling and the conservative attempt cap are engineering choices. They are not mandatory method-family gates. Claim character and citation limits guide this implementation's evidence filtering; current official checker/judge findings must be reported separately. An unchanged mock answer is useful regression evidence, not proof of a forecast mechanism.

## Offline checks

```bash
python -m pytest baselines/strong_rag_baseline -q
```

--mock produces synthetic test values and a real source quote. It is never a production fallback. The current configuration is read from config.py; defaults are not copied into another environment table. A fixed seed and temperature do not guarantee deterministic hosted responses. Keep full-roster failures, uncertainty and weak cohorts in prospective comparisons.
