## Executive summary (read this first)

This is the personal Track 4 participant repository. Start from [README.md](README.md). Preserve the public/private firewall and unrelated work. Current published organizer scoring is5.3.0 (11852111d268766fef7772ce5edfaccc1718c763), effective9October2026 22:11UTC; reconfirm before Final freeze. Local scorer3.1.0 is retained historical reference code. Use current official contracts before changing active behavior. Keep modules small and diffs tied to the user's task.

## Current sources and boundaries

[Official guide](https://github.com/Agenthon-2026/track4-analysis-public/blob/11852111d268766fef7772ce5edfaccc1718c763/README.md), [runtime](https://github.com/Agenthon-2026/track4-analysis-public/blob/11852111d268766fef7772ce5edfaccc1718c763/SUBMISSION_CLI.md), [artifact policy](https://github.com/Agenthon-2026/track4-analysis-public/blob/11852111d268766fef7772ce5edfaccc1718c763/docs/ARTIFACT-POLICY.md), [training policy](https://github.com/Agenthon-2026/track4-analysis-public/blob/11852111d268766fef7772ce5edfaccc1718c763/docs/TRAINING-POLICY.md). Import shared qfbench2-common primitives rather than rewriting scoring. Do not edit organizer scoring math to improve a participant's score.

The [9October selection ruling](https://github.com/Agenthon-2026/track4-analysis-public/issues/24#issuecomment-6090644107) also covers keeping or dropping code after outcome scores. Final selection may use the ten practice units' first-published outcomes with source/retrieval records, or labels predating every applicable cutoff. Hidden Final cutoffs are unpublished. Otherwise choose on disclosed rationale, code correctness and resources; private diagnostic scores cannot change the submission. Keep [ARTIFACT_PROVENANCE.md](ARTIFACT_PROVENANCE.md) with the build source. Preserve historical outcome-informed choices honestly; no organizer signoff is required.

Never commit realized outcomes, private answer keys, adversarial variants, credentials, private experiment logs or frozen submission records. Corpus documents and scientific/reproducibility assets are preserved. Reconcile public unit metadata only from a verified organizer source, retain provenance and run schema/checksum/embargo checks after changes. Keep local synthetic tests explicit.

Use the published analyze verb, schema, exact entity roster and target types classification/regression/ranking. Verify task meaning from the mounted prompt, target fields, label assertions and entity table. Manifest entity admission and cutoff rules protect evidence integrity; filename conventions are not additional admission rules. Read actual worker limits and account quotas from the current contract, not historical sample cards.

Every new Markdown document opens with an executive summary. Link authoritative artifacts instead of copying long field lists and formulas. Remove obsolete active instructions; preserve dated historical evidence with an explicit superseded label. Do not install fallback models, fabricate answers or add speculative configuration. The one user-approved exception is degraded recovery: an entity with no House forecast after every attempt gets the declared carry-forward row from `baselines/vector_agent/degraded.py`, and the answer's evidence_trace says so.

## Participant gates

Keep meaningful correctness, schema/data integrity, public-safety, provenance, quota/resource and owned-job checks. Mock parity, universal no-unit-regression, a particular retrieval/quote policy, arbitrary variant/sample counts and proof of a hypothesis before testing are not organizer rules. Treat tradeoffs and uncertain local measurements as diagnostics. Freeze actual comparison/stop rules before observing results; historical negative results do not ban a method family.

Run the narrow affected tests. This repo's legacy scorer tests validate retained legacy implementation only. They neither certify current5.3.0 behavior nor authorize production judging, publication or uploads. Preserve existing explicit permissions and consumed quotas when handing off.

## Git and cleanup

Inspect tracked and untracked changes. Stage only reviewed public-safe task groups; preserve other work. Personal remote is `personal` (SebastianBoehler/agenthon-2026-t4-analysis); `origin` is organizer-owned and is never a push target for personal participant work. Verify branch, exact remote URL, ancestry and resulting remote commit before reporting a push. Never force-push or silently reset unrelated files.

Back up task-owned scratch deletions and check actual references/imports. Do not delete corpus, frozen receipts, negative results, qualified images or another owner's data. Do not claim broad cleanup from pointers or cache removals alone. Private current run state stays outside this public repo.
