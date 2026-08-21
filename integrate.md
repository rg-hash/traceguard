# Integrating TraceGuard

TraceGuard is a read-only AIOps investigation service. It analyzes incident
logs, retrieves organization-specific evidence, and suggests approved
debugging checks. It never executes commands, restarts services, deploys code,
or changes infrastructure.

## 1. Start TraceGuard locally

TraceGuard currently uses the Python 3.11 environment named `.venv311`.

```bash
cd /Users/riddhigoyal/Desktop/learnings/airproj/traceguard
source .venv311/bin/activate
```

Create a `.env` file with a PostgreSQL connection URL:

```env
DATABASE_URL='postgresql://YOUR_USERNAME:YOUR_PASSWORD@localhost:5432/traceguard'
TRACEGUARD_PORT=8000
```

Start the API:

```bash
set -a
source .env
set +a

uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

Open these URLs:

- Dashboard: `http://127.0.0.1:8000/dashboard/`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

The first startup creates the required PostgreSQL tables automatically.

## 2. Onboard an organization

Before TraceGuard can provide product-specific investigations, an authorized
user supplies approved operational context:

- services and their dependencies;
- resolved incidents and confirmed root causes;
- approved runbooks and safe diagnostic checks;
- deployment metadata.

Use `POST /organizations/onboard` in Swagger UI, or send JSON by API.

```bash
curl -X POST http://127.0.0.1:8000/organizations/onboard \
  -H 'Content-Type: application/json' \
  --data @data/examples/acme_shop_onboarding.json
```

The example payload is available at
`data/examples/acme_shop_onboarding.json`.

A successful response looks like this:

```json
{
  "organization_id": "acme-shop",
  "knowledge_version": 1,
  "status": "onboarded"
}
```

Each update increments `knowledge_version`. The next investigation uses a new
retrieval index, preventing stale organization knowledge from being reused.

## 3. Payload format

An onboarding request has this high-level format:

```json
{
  "organization_id": "acme-shop",
  "display_name": "Acme Shop",
  "description": "E-commerce platform.",
  "services": [
    {
      "name": "checkout-api",
      "description": "Creates customer orders.",
      "owner": "commerce-team",
      "dependencies": ["payment-api", "inventory-api"]
    }
  ],
  "knowledge": [
    {
      "id": "ACME-RUNBOOK-001",
      "kind": "runbook",
      "title": "Payment database pool investigation",
      "service": "payment-api",
      "symptoms": ["connection pool exhausted"],
      "root_cause": "payment_worker_connection_leak",
      "steps": ["Check active database sessions and pool utilization."],
      "tags": ["payment", "database", "pool"]
    }
  ],
  "deployments": [
    {
      "id": "deploy-payment-41c9d",
      "service": "payment-api",
      "commit": "41c9d",
      "timestamp": "2026-08-20T10:00:00Z",
      "summary": "Updated payment retry handling."
    }
  ]
}
```

`knowledge.kind` must be either `incident` or `runbook`. Only approved data is
added to an organization's retrieval corpus.

## 4. Investigate an incident

Send incident logs to `POST /investigate` and include the organization's ID.

```bash
curl -X POST http://127.0.0.1:8000/investigate \
  -H 'Content-Type: application/json' \
  -d '{
    "organization_id": "acme-shop",
    "incident_id": "payment-incident-101",
    "triage_recommendation": "NEEDS_HUMAN_REVIEW",
    "events": [
      {
        "timestamp": "2026-08-21T10:10:00Z",
        "service": "payment-api",
        "severity": "ERROR",
        "message": "database connection timeout; connection pool exhausted"
      }
    ]
  }'
```

TraceGuard returns:

- an incident summary;
- retrieved incidents, architecture context, and runbooks;
- ranked root-cause hypotheses;
- deployment context for the affected service;
- approved diagnostic checks;
- `ENGINEER_REVIEW_REQUIRED`.

No remediation action is executed.

## 5. Use the dashboard

1. Open `http://127.0.0.1:8000/dashboard/`.
2. Enter an Incident ID and affected service.
3. Enter the onboarded Organization ID, for example `acme-shop`.
4. Paste logs, one message per line.
5. Select **Investigate unclassified logs**, or run BGL triage first and then
   select **Investigate triaged incident**.
6. Review the evidence, hypotheses, and runbook checks.
7. Submit engineer feedback after resolution.

## 6. Capture engineer feedback

Feedback creates product-specific labelled data for later evaluation. It does
not automatically retrain a model.

```bash
curl -X POST http://127.0.0.1:8000/investigations/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "organization_id": "acme-shop",
    "incident_id": "payment-incident-101",
    "triage_recommendation": "NEEDS_HUMAN_REVIEW",
    "hypothesis": "payment_worker_connection_leak",
    "hypothesis_accepted": true,
    "confirmed_root_cause": "payment_worker_connection_leak",
    "resolution": "Fixed unreleased database sessions.",
    "usefulness_rating": 5,
    "reviewer_note": "The runbook was accurate."
  }'
```

## 7. Integrate with monitoring or CI/CD

External systems should call `POST /investigate` through a small webhook
adapter. The adapter should transform an alert or failed-build log into
TraceGuard's `incident_id`, `organization_id`, and `events` fields.

```text
Grafana alert / Jenkins failure / GitHub Action failure
                         |
                         v
               Your authenticated webhook adapter
                         |
                         v
                POST /investigate to TraceGuard
                         |
                         v
       Evidence-backed plan sent to an engineer for review
```

For a public deployment, protect onboarding and investigation endpoints with
authentication, role-based access control, HTTPS, rate limits, and separate
organization credentials. Do not send secrets, access tokens, or raw customer
data in log payloads.

## 8. Current scope and next enhancement

The current onboarding API accepts structured JSON. A future dashboard feature
can add document uploads (`.md`, `.txt`, `.json`, and PDF) followed by a human
approval screen that converts extracted information into the same controlled
onboarding format.
