Learn it in this order—don’t try to understand the whole project at once.

## Day 1: Understand the goal and data

Read [README.md](/Users/riddhigoyal/Downloads/airproj/traceguard/README.md) and [data-card.md](/Users/riddhigoyal/Downloads/airproj/traceguard/docs/data-card.md).

Then open [data.py](/Users/riddhigoyal/Downloads/airproj/traceguard/app/data.py).

Focus only on:

- What an `incident` looks like
- What `events` are
- The labels: `normal`, `database`, `network`, `application`
- How the generator adds log messages

Run:

```bash
python scripts/generate_demo_data.py
```

Open `data/generated/incidents.jsonl` and read 3–5 incident records.

## Day 2: Understand training

Read [ml.py](/Users/riddhigoyal/Downloads/airproj/traceguard/app/ml.py).

Follow these functions in order:

1. `incident_text()`  
   Combines all logs of one incident into one text string.

2. `train_models()`  
   Trains two models:
   - anomaly model: normal versus abnormal
   - cause model: database versus network versus application

3. `retrieve_evidence()`  
   Finds log lines that support the predicted cause.

4. `analyze()`  
   Combines prediction, evidence, confidence, and safety decision.

Run:

```bash
python scripts/train.py
python scripts/evaluate.py
```

Understand that training learns patterns from labelled examples; evaluation checks whether those patterns work on unseen examples.

## Day 3: Understand the API

Read [api.py](/Users/riddhigoyal/Downloads/airproj/traceguard/app/api.py).

Focus on three endpoints:

```text
GET  /health
GET  /incidents
GET  /incidents/{incident_id}/analyze
POST /analyze
```

Start the server:

```bash
uvicorn app.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Use Swagger UI to call `/incidents`, copy an incident ID, and call its `/analyze` endpoint.

## Day 4: Trace one incident end to end

Choose a network incident, such as `demo-network-001`.

Trace this route:

```text
JSONL incident
→ load_dataset()
→ API receives incident ID
→ analyze()
→ anomaly model
→ root-cause model
→ retrieve_evidence()
→ confidence gate
→ JSON response
```

Put a temporary `print()` inside `analyze()` to see:

```python
print(text)
print(anomaly_probability)
print(root_cause, cause_confidence)
print(evidence)
```

This is the fastest way to make the flow intuitive.

## Day 5: Understand safety and tests

Read:

- [test_ml.py](/Users/riddhigoyal/Downloads/airproj/traceguard/tests/test_ml.py)
- [test_api.py](/Users/riddhigoyal/Downloads/airproj/traceguard/tests/test_api.py)
- [architecture.md](/Users/riddhigoyal/Downloads/airproj/traceguard/docs/architecture.md)

Key idea: the system must not guess a root cause when it has weak confidence or no evidence. That is why it returns `NEEDS_HUMAN_REVIEW`.

Run:

```bash
pytest -q
```

## Day 6–7: Modify it yourself

Make one small change at a time:

1. Add a new root cause: `authentication`
2. Add log signatures such as:
   - `token validation failed`
   - `authentication service unavailable`
   - `expired session credential`
3. Update `CAUSES`, `SIGNATURES`, and evidence keywords.
4. Regenerate data, retrain, evaluate, and add a test.

When you can make this change independently, you understand the core project.

## Mental model

```text
Data.py       = creates the practice problems
ml.py         = learns patterns and makes safe decisions
api.py        = lets other systems use the intelligence
tests/        = checks that it behaves correctly
scripts/      = generate, train, and evaluate
Docker/CI     = makes it deployable and reliable
```

Start with the data, then ML, then API. That sequence will make the project much easier to understand.
