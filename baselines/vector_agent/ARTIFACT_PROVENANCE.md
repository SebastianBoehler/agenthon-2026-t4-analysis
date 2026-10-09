## Executive summary (read this first)

`vector_agent` includes deterministic aggregation and three calibrated band-width constants, with no fitted model weights or stored task answers. House supplies forecasts during evaluation; the user-approved degraded recovery uses mounted task values when usable forecasts are absent. The classification multiplier was also selected using public-data outcomes collected for a private evaluation set. Its eligibility for unknown Final cutoffs is **unresolved**. This recovered source is not a Final-qualified image.

## Calibrated artifacts

| Constant in `agent.py` | Calibration source | Availability and retrieval | Cutoff status |
|---|---|---|---|
| `BAND_SCALE = 1.5` | Six published practice units: Treasury auction results, CFTC positions, first-print CPI components, bank earnings and two Treasury-yield windows. | Source observations span October 2022–December 2024; retrieved 7 October 2026. Exact source and retrieval records remain in the private calibration archive. | Subject to the explicit practice-outcome exception below; no claim that these observations precede every Final cutoff. |
| `YIELD_BAND_SCALE = 2.0` | Published practice units `t4-fomc-curve-20220728` and `t4-fomc-curve-20240918`, using constant-maturity Treasury closes. A later private evaluation unit also informed retention. | Practice closes on 20 September 2022 and 6 November 2024; retrieved 7 October 2026. The private check uses June–July 2026 data. | Practice-based selection has the exception below. Any selection or retention informed by private evaluation outcomes needs a separate cutoff audit. |
| `CLASS_BAND_SCALE = 2.5` | Published EPS and earnings-reaction practice units, plus private evaluation units using SEC earnings, ALFRED vintages and market closes. | Public-data observations through October 2026; retrieved 7–8 October 2026. | The practice exception does not establish eligibility of the additional private evaluation data. Audit each source and selection date against applicable cutoffs, or reselect using eligible data, before Final freeze. |

The calibration records, realized values, private task artifacts and experiment results remain outside this repository. Constants are global numerical choices, not task-answer lookup tables.

## Authoritative data rules

The organizer's [practice-outcome ruling](https://github.com/Agenthon-2026/track4-analysis-public/issues/24#issuecomment-6029140058) permits first-published outcomes of the published practice units to inform constants and design for every Final task, regardless of its cutoff. Such measurements are explicitly in-sample. This exception does not extend to arbitrary newly collected private evaluation outcomes.

Other fitting, selection and calibration must satisfy the [artifact policy](https://github.com/Agenthon-2026/track4-analysis-public/blob/458bc07efc05983de342ccb371e2b3cd793d621f/docs/ARTIFACT-POLICY.md) and [training policy](https://github.com/Agenthon-2026/track4-analysis-public/blob/458bc07efc05983de342ccb371e2b3cd793d621f/docs/TRAINING-POLICY.md). Unknown Final cutoffs prevent a blanket eligibility claim.

Public source references include [Treasury yields](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve), [Treasury auction results](https://www.treasurydirect.gov/TA_WS/securities/search), [CFTC historical reports](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm), [ALFRED](https://alfred.stlouisfed.org/) and [SEC EDGAR](https://www.sec.gov/edgar/search/). A source link alone does not prove a particular vintage or its cutoff eligibility.

## Runtime inputs

Inference uses only the mounted task, admitted corpus and card timeout, plus organizer-injected House access. Offline outcome records and external evidence are not available to the runtime.
