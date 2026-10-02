# Evaluation framework (clinician-labelled gold set)

Goal: replace "it looks right" with measured numbers on **held-out, clinician-labelled** texts.
Everything here is dependency-free Python (stdlib) and works offline, except `--engine full`.

```
evaluation/
  labels.py            label vocabulary (7 diagnoses incl. "غير واضح حالياً", urgency, red-flag codes)
  make_sheet.py        pool of texts -> de-duplicated annotation sheet (+ leakage-safe split assignment)
  agreement.py         Cohen's kappa between two annotators + cases to adjudicate
  dataset_checks.py    duplicates / near-duplicates / train-test leakage / class balance
  run_eval.py          scores an engine (rules | full | api) or a predictions CSV, writes report + errors
  guidelines_ar.md     annotation guidelines for the dentists (DRAFT - needs clinical sign-off)
  data/                your real data (git-ignored: patient text is sensitive); template is tracked
  reports/             generated reports (git-ignored)
```

## 1. Collect candidate texts
Use real, varied patient descriptions (app submissions, clinic intake forms, WhatsApp-style
messages, voice-to-text output). Do **not** use only templated or synthetic text: the existing
2,500-row CSV has 740 distinct "normalised" patterns and cannot show real-world accuracy.
Target mix: 20-30% dialect beyond Levantine (Gulf, Egyptian, Maghrebi), short and long inputs,
typos / ASR noise, 10-15% hard negatives (looks like an abscess but is not, negated symptoms),
and non-dental or empty-of-symptoms texts (they must come out as "غير واضح").

## 2. Build the sheet and annotate
```bash
python -m evaluation.make_sheet pool.csv --out evaluation/data/annotation_sheet.csv --n 500
```
Two dentists fill `label_a/urgency_a/red_flags_a` and `label_b/...` **independently**, following
`guidelines_ar.md`. Then:
```bash
python -m evaluation.agreement evaluation/data/annotation_sheet.csv --disagreements evaluation/data/to_adjudicate.csv
```
- Cohen's kappa **>= 0.70** on both diagnosis and urgency before the set counts as ground truth
  (0.60-0.70: tighten the guidelines and re-annotate the disagreements; < 0.60: stop).
- A third senior dentist adjudicates the disagreements; the final answer goes to `label`,
  `urgency`, `red_flags` (separate multiple codes with `;`).

## 3. Split without leakage
Assign `dev` / `test` **after** adjudication. `make_sheet.assign_splits` keeps near-identical texts
and equal `group_id` (same patient / same template family) in the same split. Use `dev` to tune
thresholds and rules, and touch `test` only for the final number; once you tune on `test` it is dev.
```bash
python -m evaluation.dataset_checks evaluation/data/gold.csv   # must exit 0 (no cross-split leakage)
```

## 4. Score
```bash
python -m evaluation.run_eval evaluation/data/gold.csv --engine rules              # rule engine only
python -m evaluation.run_eval evaluation/data/gold.csv --engine api --api-url https://nlp.medismile.xn--mgbaab0cxheq.tech
python -m evaluation.run_eval evaluation/data/gold.csv --engine full               # rules + AraBERT (torch)
```
Output (`evaluation/reports/`): `report.md`, `report.json`, `errors.csv` (every miss with gold vs
predicted - this is your improvement backlog). The process exits 1 when the gate fails, so it can run in CI.

### Metrics that matter (in this order)
1. **Urgent recall** (missed urgent cases = under-triage): the safety metric. Look at the CI lower bound, not just the point value.
2. **Red-flag recall**: fever+swelling, trismus, dysphagia, breathing difficulty, trauma, uncontrolled bleeding.
3. Per-class precision/recall/F1 and **macro-F1** (accuracy hides rare classes such as abscess).
4. **Abstain rate** ("غير واضح") and accuracy when answered: a model that abstains on everything is "safe" and useless.
5. **Calibration (ECE)** and the risk-coverage table: choose the abstain threshold from this curve, not by guess.

### Suggested release targets (agree them with the clinical lead before measuring)
| Metric | Target |
|---|---|
| Urgent recall | >= 0.95, with the CI lower bound >= 0.90 |
| Red-flag recall | >= 0.98 |
| Macro-F1 | >= 0.80 |
| Abstain rate | <= 25% |
| ECE | <= 0.10 |

### Minimum sizes for numbers to mean something
`test` split: **>= 400 cases, >= 40 per diagnosis, >= 80 urgent cases**. The report prints a warning
whenever a class has fewer than 20 cases or the whole set fewer than 100: below that, a few errors
swing the result by 5-10 points.

## 5. Improve, then repeat
1. Read `errors.csv`, group the misses (negation, dialect, multi-symptom, missing label such as periodontitis).
2. Fix rules / add data on `dev`; re-run `dev` until stable.
3. Run `test` once, archive `report.md` with the model/commit id, compare with the previous release.
4. In production, log (anonymised) overrides by clinicians and add them to the next gold set.

## Evaluating a different model or an old run
`--predictions preds.csv` takes `id,pred_label,pred_urgency[,confidence]`, so any model
(a fine-tuned classifier, an LLM, the live service) is scored by exactly the same code.
