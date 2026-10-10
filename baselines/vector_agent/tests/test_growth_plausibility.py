"""Growth samples whose level forecast carries another period's growth are dropped."""
from baselines.vector_agent.consistency import plausible_growth_samples
from baselines.vector_agent.parse import Sample, parse_predictions
from baselines.vector_agent.prompts import unit_prompt
from baselines.vector_agent.unit import Unit


def sample(level, latest, point=0.0):
    return Sample("GS", point, point - 10, point + 10, None, None, None, "", level=level, base=5.47, latest=latest)


def test_level_far_from_latest_reported_level_is_dropped():
    good = [sample(8.3, 8.62, 51.7), sample(8.62, 8.62, 57.6)]
    bad = sample(15.27, 8.62, 179.0)
    assert plausible_growth_samples(good + [bad]) == good


def test_reference_is_the_median_of_the_latest_level_replies():
    samples = [sample(8.3, 8.62), sample(8.5, 8.7), sample(15.3, 15.3)]  # one reply misstates the latest level
    assert plausible_growth_samples(samples) == samples[:2]


def test_without_reference_or_level_nothing_is_dropped():
    assert plausible_growth_samples([sample(15.27, None)]) == [sample(15.27, None)]
    kept = [sample(None, 8.62, 179.0), sample(8.3, 8.62)]
    assert plausible_growth_samples(kept) == kept


def test_all_implausible_keeps_all():
    samples = [sample(15.27, 8.62), sample(16.0, 8.62)]
    assert plausible_growth_samples(samples) == samples


def test_latest_level_is_parsed_and_requested_only_for_growth_targets():
    growth = {"target": {"name": "eps_yoy_growth_pct", "type": "regression"},
              "entities": [{"entity_id": "GS", "prior_year_q_eps": 5.47}]}
    level = {"target": {"name": "eps_level", "type": "regression"},
             "entities": [{"entity_id": "GS", "prior_year_q_eps": 5.47}]}
    assert "latest_level" in unit_prompt(Unit(growth, {}), [])
    assert "latest_level" not in unit_prompt(Unit(level, {}), [])
    reply = '{"predictions": [{"entity_id": "GS", "point_forecast": 179, "lo": 100, "hi": 200, "level_forecast": 15.27, "base_value": 5.47, "latest_level": "8.62"}]}'
    got = parse_predictions(reply, target_type="regression", labels=[], entity_ids={"GS"}, weight=1.0, source="x")
    assert got[0].latest == 8.62
