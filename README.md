## Executive summary (read this first)

This personal fork contains a participant agent for Track 4: predict a label, value or ordering from a table and frozen text evidence. Start here for local work. Current official scoring is5.2.2; the retained local scorer package is a historical3.1.0 reference. Local mock checks measure wiring and evidence handling, not predictive quality or official rank. Public units carry no outcomes, naive answers or Final judge feedback. Current challenge rules come from the versioned organizer sources linked below.

## Current challenge contract

Read the [official participant guide](https://github.com/Agenthon-2026/track4-analysis-public/blob/1c744e1d6725340643a533f436517d72b53ca0e1/README.md), [submission runtime](https://github.com/Agenthon-2026/track4-analysis-public/blob/1c744e1d6725340643a533f436517d72b53ca0e1/SUBMISSION_CLI.md), [training policy](https://github.com/Agenthon-2026/track4-analysis-public/blob/1c744e1d6725340643a533f436517d72b53ca0e1/docs/TRAINING-POLICY.md), [House guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/bd01548e34d21fd660d88fd06157078fb25ece4e/docs/HOUSE-MODEL.md) and [Development runtime](https://github.com/Agenthon-2026/Agenthon2026-public/blob/bd01548e34d21fd660d88fd06157078fb25ece4e/docs/DEVELOPMENT-RUNTIME.md). Do not replace those rules with the historical local scorer's formulas or old faithfulness admission threshold. Current faithfulness uses per-claim penalties. The [organizer clarification](https://github.com/Agenthon-2026/track4-analysis-public/issues/23#issuecomment-5985154828) explains task semantics, available feedback and cutoff-valid training labels.

The mounted task prompt, target, label assertions and entity table define the requested units, horizon and ranking direction. Do not infer direction from a generic rank convention. The public units are examples, not a representative evaluation population. A model response, schema check or exact quotation does not establish forecast quality. Keep complete-roster failures and weak cohorts visible when comparing candidates.

## Participant implementation

The active agent is [strong_rag_baseline](baselines/strong_rag_baseline/README.md). It loads admitted pre-cutoff documents, retrieves bounded excerpts, asks the supplied model and emits model values with exact citation spans. Invalid labels or interval bounds are errors; they are not replaced by guessed predictions. Unmatched quotes are dropped. That exact-quote policy has a completeness tradeoff and is not a challenge-wide ban on other grounded methods.

```bash
python -m baselines.strong_rag_baseline.cli analyze \
  --task units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out /tmp/t4-answer.json --mock
```

The mock returns synthetic values for offline tests. House access is injected during evaluated runs; a local credential cannot enable it. Local diagnostic servers are independent of official House evidence. The [local interface](SUBMISSION_CLI.md) and [configuration source](baselines/strong_rag_baseline/config.py) describe implementation settings. Build context must contain runtime sources and license only; tests, bytecode, outputs and private workspace files do not belong in a candidate image.

## Checks and exploration

```bash
python -m pytest baselines -q
python .github/validate_units.py analysis --stdlib-only
```

Import shared toolkit checks and scoring primitives; never duplicate scoring math. Use the current official checker when assessing current claim rules. The retained local `qfbench2_track_analysis` and `faithfulness` packages preserve earlier reproducibility; their smoke scores are historical diagnostics. They are not a production gate for current candidates.

Correct schema/roster, eligible data, provenance, public safety, owned-job isolation and actual resource/request/output limits are validity controls. Top-K, chunk size, quote policy, local timeout/attempt cap, exact-output parity and no-family-regression are engineering or study choices. Do not turn them into unexplained hard vetoes on a distinct informative experiment. Freeze meaningful comparisons and assess mean, failures, weak cohorts and uncertainty. Publication, submission and registry access follow the user's explicit authority.
