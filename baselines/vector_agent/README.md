## Executive summary (read this first)

`vector_agent` is the Track 4 `analyze` agent: it forecasts every roster row from sampled House replies, with claims that quote admitted pre-cutoff documents word for word. Every forecast comes from the model. Robustness comes from concurrency, retries and redundant prompts, never from substitute predictions. Local runs check wiring, timing and reply format. They do not estimate official scores.

## How a unit is answered

- **Full-context mode.** Used when the admitted corpus is 110k characters or less. Every document goes into the prompt whole, each followed by an arithmetic summary of its pipe tables from `tables.py`:
  - last value, mean and spread;
  - step-to-step change;
  - change sd over the unit's horizon under a random walk;
  - revision sizes per release for vintage tables.

  The model forecasts groups of up to 12 entities per reply, and each group is sampled until every entity has 9 forecasts.
- **Retrieval mode** (large filings). Each entity gets up to 44k characters of BM25 and finance-anchor excerpts. Rosters of up to 10 rows are split into at most 5 prompts of one or two entities, each sampled about 5 times; larger rosters use at most 10 grouped prompts. With single-entity prompts, leftover requests go to entities whose samples disagree.
- **Forecast method** set by the prompt:
  1. Start from the naive carry-forward value, or from the base rate for a direction or event label.
  2. For a change or growth target, forecast the level first, then compute the change against exactly the base the task names.
  3. Move by the change the evidence implies.
  4. Estimate sigma from comparable past surprises (the computed horizon sd is its floor), and set the band at about ±1.645 sigma.
- **Aggregation.**
  - Point forecast: weighted median of the samples.
  - Label: weighted vote.
  - Band: 5th and 95th percentiles of the pooled sample distributions, after removing outlying samples and runaway widths. A band edge more than 5 sigma from a sample's own point (a dropped digit, for example) is replaced by ±1.645 sigma. Probability bands are clipped to [0, 1].
- **Event labels.** For a binary vocabulary of an event and its negation (for example `credit_event` / `no_event`), three cohort calls estimate how many entities will have the event. That many entities get the event label, ranked by pooled event probability, and anything at or above 0.5 always gets it. Asked one entity at a time, the model anchors these probabilities low.
- **Event-probability bands.** When the numeric quantity is an event probability, the band is the interval-score optimum for a 0/1 outcome: [0, p] below 0.05, [p, 1] above 0.95, and [0, 1] otherwise.
- **Pure-label tasks.** When the task defines no numeric quantity, the label's probability stands as the number, with the band [0, 1].
- **Percent-growth targets.** Growth is recomputed from the model's own level forecast and the table's base value (`consistency.py`).
- **Claims.** One or two exact slices of documents the manifest admits for the entity, each at most 399 characters, taken only from documents that have a flat `text` field. The task-table row is used only when no such document is admitted.
- **Reasons.** Three driver reasons per unit. Each names one main driver, covers every entity it affects, quotes a verbatim premise and cites up to three supporting passages. `submitted_reasons` go into the answer only when they pass the shape, deny-list and byte-cap rules.
- **Oversized prompts.** A prompt rejected with HTTP 400 is retried with half, then a quarter, of its evidence.
- **Budget.**
  - At most 24 requests per unit; thinking is disabled.
  - 4 concurrent calls.
  - No new calls after 300 s, a hard stop at 440 s, and reasons by 480 s.
  - The answer is written atomically and the process then exits at once.

## Run and check

```bash
python -m pytest -p no:cases baselines/vector_agent -q
docker buildx build --platform linux/amd64 -f baselines/Dockerfile.vector -t <tag> --load .
python -m baselines.vector_agent.tests.fake_house_server --delay 30 --fault-every 7   # synthetic HTTP stand-in
python -m baselines.vector_agent.tests.run_local --out <dir> [unit ...]               # hosted Nemotron via OpenRouter; key in git-ignored .env
```

Validate answers with the organizer's current checker: `check_claim_rules` in `baselines/guardrails_example/citation_rail.py` (scorer 5.2.2), the toolkit schema, and `qfbench2-smoke --track analysis`.
