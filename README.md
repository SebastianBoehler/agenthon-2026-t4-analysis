# Agenthon 2026, Track 4 (Explainability): Proof of Alpha

My participant code for **Proof of Alpha** (a one-person team) in [Agenthon 2026](https://www.agenthon.net/) Track 4.
The task gives an agent a table of entities and a set of dated, pre-cutoff documents (filings, macro
releases, auction and positioning reports). For each entity, the agent forecasts a value, a label and
an interval, and backs them with claims quoted word for word from the documents and short reasons.

**Final result: 2nd of 63 teams in Track 4** (Final score 0.4857,
[official leaderboard](https://www.agenthon.net/#leaderboard)).

The winning agent is [`baselines/vector_agent`](baselines/vector_agent/README.md). Everything else in
this repository comes from the organisers' public starter kit
([Agenthon-2026/track4-analysis-public](https://github.com/Agenthon-2026/track4-analysis-public)):
public example units, the reference baselines, the local scorer and the checkers.

## How the agent works

1. **Evidence.** Small corpora go into the prompt whole; large filings are cut to BM25 and
   finance-anchored excerpts per entity. Each document also gets an arithmetic summary of its tables
   (last value, changes, random-walk spread over the horizon, revision sizes).
2. **Repeated forecasts.** The organiser-provided model (the "House", Nemotron 3 Super) is sampled
   several times per entity. The prompt starts from the naive carry-forward value, forecasts levels
   before changes and estimates a band from comparable past surprises.
3. **Deterministic aggregation.** Points are weighted medians of the samples, labels are weighted votes,
   and intervals come from the pooled sample distribution. Task-specific rules handle event counts,
   threshold labels, revision directions, growth targets and basis-point changes.
4. **Grounded explanations.** Claims are exact slices of admitted documents with verified offsets.
   Reasons each quote a verbatim premise and cite up to three supporting passages, and they pass
   shape and deny-list checks before going into the answer.
5. **Time budget.** Requests, concurrency and deadlines are scheduled to fit the 600-second card. An
   entity with no model reply gets an explicit, disclosed carry-forward fallback.

The [agent README](baselines/vector_agent/README.md) gives every rule in detail.
[ARTIFACT_PROVENANCE.md](baselines/vector_agent/ARTIFACT_PROVENANCE.md) records how each constant was
chosen and what data each choice used.

## Reproducing the submission

The image is built from `main` with [`baselines/Dockerfile.vector`](baselines/Dockerfile.vector). It contains only the
agent's Python sources (standard library only), the provenance record and the licence. It bundles no
model, weights or data.

**Build and inspect the image**

```bash
git clone https://github.com/SebastianBoehler/agenthon-2026-t4-analysis
cd agenthon-2026-t4-analysis
docker buildx build --platform linux/amd64 -f baselines/Dockerfile.vector -t t4-vector --load .
docker run --rm t4-vector            # prints the analyze interface
```

**Run it on a unit.** Create an output folder first (`mkdir -p out`). The agent calls an OpenAI-compatible chat endpoint. In the official evaluation
the organisers inject the House route. To run it yourself, point it at any endpoint serving the same
model, for example NVIDIA Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) on OpenRouter:

```bash
docker run --rm \
  -e MODEL_ENDPOINT=https://openrouter.ai/api \
  -e MODEL_NAME=nvidia/nemotron-3-super-120b-a12b \
  -e MODEL_TOKEN=$OPENROUTER_API_KEY \
  -v "$PWD/units/t4-credit-event-2023:/input:ro" -v "$PWD/out:/output" \
  t4-vector analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json
```

The agent sends `POST $MODEL_ENDPOINT/v1/chat/completions`. Without a reachable endpoint it still
runs end to end: every entity gets the disclosed carry-forward fallback with quoted claims, which is a
quick way to check the build and mounts. A hosted model differs from the House in
provider, quantisation, latency and limits, so local answers will not match the official ones exactly,
and the Final units are sealed. Local runs reproduce the method, not the official score.

**Tests.** The tests need the organisers' shared toolkit (`qfbench2_common`):

```bash
pip install pytest "git+https://github.com/Agenthon-2026/Agenthon2026-public#subdirectory=common"
python -m pytest -p no:cases baselines/vector_agent -q
```

All 190 tests pass. They use synthetic entities and a fake House server, and need
no network access or credentials.

## Data and licences

- My code is under the [MIT licence](LICENSE).
- The example units, starter code and scorer are the organisers'. See
  [DATA-LICENSE.md](DATA-LICENSE.md) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
- No credentials, protected outcomes, Final answers or private experiment records are in this
  repository.
- The model is not ours and is not redistributed here. Use it under its own licence and your
  provider's terms.

## Further reading

- [docs/WORKING-NOTES.md](docs/WORKING-NOTES.md): the working README from the competition, kept unchanged.
- [SUBMISSION_CLI.md](SUBMISSION_CLI.md): the container interface.
- Organiser rules and runtime:
  [participant guide](https://github.com/Agenthon-2026/track4-analysis-public/blob/11852111d268766fef7772ce5edfaccc1718c763/README.md),
  [Development runtime](https://github.com/Agenthon-2026/Agenthon2026-public/blob/bd01548e34d21fd660d88fd06157078fb25ece4e/docs/DEVELOPMENT-RUNTIME.md).
