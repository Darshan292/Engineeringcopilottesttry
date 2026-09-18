"""One realistic sample input per tool, loadable with a single click.

These are deliberately not toy examples. Each one contains the traps the tool
is supposed to handle: the test sample has silent boundary bugs, the API sample
has an undocumented error path, the log sample has a misleading first error,
and the transcript sample is full of human names that must be anonymized.
"""

from __future__ import annotations

UNIT_TEST_SAMPLE = '''from dataclasses import dataclass
from datetime import date


@dataclass
class LineItem:
    sku: str
    unit_price_cents: int
    quantity: int


def apply_tiered_discount(
    items: list[LineItem],
    customer_tier: str,
    promo_code: str | None = None,
    today: date | None = None,
) -> dict:
    """Compute an order total with tier and promo discounts applied.

    Tiers: "standard" 0%, "silver" 5%, "gold" 10%, "platinum" 15%.
    Promo "SAVE20" adds 20% off but only for orders over $100 and only
    stacks with tiers up to "gold". Total discount is capped at 30%.
    """
    if not items:
        raise ValueError("order must contain at least one line item")

    today = today or date.today()
    subtotal = sum(i.unit_price_cents * i.quantity for i in items)

    tier_rates = {"standard": 0.0, "silver": 0.05, "gold": 0.10, "platinum": 0.15}
    if customer_tier not in tier_rates:
        raise KeyError(f"unknown tier: {customer_tier}")
    rate = tier_rates[customer_tier]

    if promo_code == "SAVE20":
        if subtotal > 10000 and customer_tier != "platinum":
            rate += 0.20
    elif promo_code is not None:
        raise ValueError(f"invalid promo code: {promo_code}")

    rate = min(rate, 0.30)
    discount_cents = int(subtotal * rate)

    return {
        "subtotal_cents": subtotal,
        "discount_cents": discount_cents,
        "total_cents": subtotal - discount_cents,
        "applied_rate": rate,
        "calculated_on": today.isoformat(),
    }
'''


API_DOC_SAMPLE = '''from fastapi import APIRouter, Depends, HTTPException, Query, Header
from pydantic import BaseModel, Field
from typing import Literal

router = APIRouter(prefix="/v1/deployments", tags=["deployments"])


class DeploymentCreate(BaseModel):
    service: str = Field(..., min_length=1, max_length=64)
    image_tag: str = Field(..., pattern=r"^[a-z0-9._-]+$")
    environment: Literal["staging", "production"]
    replicas: int = Field(default=3, ge=1, le=50)
    canary_percent: int | None = Field(default=None, ge=1, le=50)


class Deployment(BaseModel):
    id: str
    service: str
    image_tag: str
    environment: str
    replicas: int
    status: Literal["pending", "rolling", "healthy", "failed", "rolled_back"]
    created_at: str
    created_by: str


def require_token(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    return authorization.removeprefix("Bearer ")


@router.get("", response_model=list[Deployment])
def list_deployments(
    environment: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    token: str = Depends(require_token),
):
    """List deployments, newest first. Paginated via opaque cursor."""
    ...


@router.post("", response_model=Deployment, status_code=201)
def create_deployment(body: DeploymentCreate, token: str = Depends(require_token)):
    """Trigger a new deployment. Returns immediately; status is async."""
    if body.environment == "production" and body.canary_percent is None:
        raise HTTPException(status_code=422, detail="production deploys require a canary_percent")
    if _has_active_deployment(body.service, body.environment):
        raise HTTPException(status_code=409, detail="a deployment is already in progress")
    ...


@router.post("/{deployment_id}/rollback", response_model=Deployment)
def rollback(deployment_id: str, reason: str = Query(..., min_length=10), token: str = Depends(require_token)):
    """Roll a deployment back to the previously healthy image tag."""
    dep = _get(deployment_id)
    if dep is None:
        raise HTTPException(status_code=404, detail="deployment not found")
    if dep.status in ("pending", "rolled_back"):
        raise HTTPException(status_code=409, detail=f"cannot roll back a {dep.status} deployment")
    ...
'''


LOG_RCA_SAMPLE = '''2026-09-14T02:11:03.441Z INFO  checkout-api [pool] hikari-main - stats (total=40, active=6, idle=34, waiting=0)
2026-09-14T02:14:57.882Z INFO  inventory-svc [scheduler] starting nightly reindex job id=reindex-8841
2026-09-14T02:15:02.119Z WARN  inventory-svc [db] slow query 4192ms :: SELECT sku, warehouse_id, qty FROM stock_levels WHERE updated_at > $1 ORDER BY updated_at
2026-09-14T02:15:40.007Z WARN  checkout-api [pool] hikari-main - stats (total=40, active=31, idle=9, waiting=0)
2026-09-14T02:16:12.556Z WARN  inventory-svc [db] slow query 11804ms :: SELECT sku, warehouse_id, qty FROM stock_levels WHERE updated_at > $1 ORDER BY updated_at
2026-09-14T02:16:44.201Z WARN  checkout-api [pool] hikari-main - stats (total=40, active=40, idle=0, waiting=17)
2026-09-14T02:16:45.330Z ERROR checkout-api [http] POST /v1/checkout 500 in 30012ms trace=7f3a91c2
2026-09-14T02:16:45.331Z ERROR checkout-api [db] java.sql.SQLTransientConnectionException: hikari-main - Connection is not available, request timed out after 30000ms
2026-09-14T02:16:51.874Z ERROR checkout-api [http] POST /v1/checkout 500 in 30004ms trace=b1c88de0
2026-09-14T02:17:02.003Z ERROR payments-gw [upstream] checkout-api returned 500, circuit breaker half-open (failures=5/5)
2026-09-14T02:17:02.119Z WARN  payments-gw [circuit] OPEN for checkout-api, will retry in 30s
2026-09-14T02:17:30.442Z INFO  payments-gw [circuit] HALF_OPEN for checkout-api
2026-09-14T02:17:31.006Z ERROR payments-gw [circuit] probe failed, back to OPEN
2026-09-14T02:18:04.771Z ERROR checkout-api [db] java.sql.SQLTransientConnectionException: hikari-main - Connection is not available, request timed out after 30000ms
2026-09-14T02:18:15.229Z INFO  inventory-svc [scheduler] reindex job id=reindex-8841 progress 34% (rows=2841002)
2026-09-14T02:19:41.887Z ERROR checkout-api [http] POST /v1/checkout 500 in 30008ms trace=aa02f5b1
2026-09-14T02:21:10.552Z WARN  orders-svc [queue] checkout_events consumer lag 41200 messages (threshold 5000)
2026-09-14T02:23:02.010Z INFO  oncall [manual] scaled checkout-api replicas 6 -> 12
2026-09-14T02:23:48.663Z WARN  checkout-api [pool] hikari-main - stats (total=40, active=40, idle=0, waiting=39)
2026-09-14T02:24:01.447Z ERROR rds-proxy [limits] max_connections=200 reached, rejecting new connections from 10.4.2.0/24
2026-09-14T02:26:33.900Z INFO  inventory-svc [scheduler] reindex job id=reindex-8841 CANCELLED by operator
2026-09-14T02:27:12.338Z INFO  checkout-api [pool] hikari-main - stats (total=40, active=22, idle=18, waiting=0)
2026-09-14T02:28:40.115Z INFO  checkout-api [http] POST /v1/checkout 201 in 284ms trace=c9d10e44
2026-09-14T02:29:05.660Z INFO  payments-gw [circuit] CLOSED for checkout-api
2026-09-14T02:31:20.884Z WARN  orders-svc [queue] checkout_events consumer lag 12400 messages, draining
'''


POSTMORTEM_SAMPLE = '''#incident-checkout-2026-09-14

[02:18] Priya Raghavan: getting paged, checkout 500s spiking. anyone else seeing this?
[02:19] Marcus Webb: yeah dashboards are red. payments-gw circuit breaker just opened on checkout-api
[02:19] Priya Raghavan: taking IC. Marcus can you own comms?
[02:20] Marcus Webb: yep. statuspage updated, "investigating elevated checkout errors"
[02:21] Priya Raghavan: checkout-api logs are all SQLTransientConnectionException, pool exhausted. 40/40 active, 17 waiting
[02:22] Dan Oyelaran: did anything deploy? I pushed the pricing service at 23:40 yesterday but that is hours ago
[02:22] Priya Raghavan: no deploys in the window. checking db side
[02:23] Priya Raghavan: scaling checkout-api 6 -> 12 to see if it relieves anything
[02:24] Marcus Webb: that made it worse? waiting count went to 39
[02:24] Priya Raghavan: yeah more replicas = more pool connections = more pressure on rds. reverting that thought
[02:24] Dan Oyelaran: rds-proxy is logging max_connections=200 reached
[02:25] Aisha Bello: joining. I own inventory-svc. the nightly reindex job started at 02:14, that thing does a full table scan on stock_levels
[02:25] Priya Raghavan: that lines up almost exactly with the pool climbing
[02:26] Aisha Bello: it is supposed to run at 02:00 against the read replica. checking the config
[02:26] Aisha Bello: it is pointed at the primary. someone changed the DATABASE_URL in the reindex cronjob manifest
[02:26] Priya Raghavan: kill it
[02:26] Aisha Bello: cancelled reindex-8841
[02:28] Priya Raghavan: pool is draining. 22 active, 0 waiting
[02:28] Marcus Webb: first successful checkout in 12 min
[02:29] Marcus Webb: circuit breaker closed. statuspage updated to monitoring
[02:31] Dan Oyelaran: orders-svc has 12k message backlog on checkout_events, draining on its own
[02:33] Priya Raghavan: ok calling it mitigated. 02:16 to 02:28, roughly 12 minutes of checkout unavailability
[02:34] Marcus Webb: do we know how many orders we lost?
[02:35] Dan Oyelaran: rough count from the 500s, about 340 failed checkout attempts. no idea how many retried
[02:36] Priya Raghavan: we never alerted on pool saturation, we only found out because customers hit 500s. that is the real miss here
[02:37] Aisha Bello: git blame on the manifest shows the DATABASE_URL change went in 3 weeks ago in a PR titled "fix reindex connection timeout". it was approved but nobody caught the host change
[02:38] Priya Raghavan: so it has been a landmine for 3 weeks and only fired because the table finally got big enough to matter
[02:39] Marcus Webb: the reindex also has no query timeout and no rate limiting
[02:41] Priya Raghavan: ok. mitigated but not fixed, the reindex job is still misconfigured, just not running. writing this up tomorrow
'''


SAMPLES: dict[str, dict[str, str]] = {
    "unit-tests": {
        "label": "Tiered discount calculator (Python)",
        "description": "A pricing function with stacking rules, a cap, and at least two boundary bugs hiding in it.",
        "content": UNIT_TEST_SAMPLE,
    },
    "api-docs": {
        "label": "Deployments API (FastAPI router)",
        "description": "Three routes with auth, pagination, conditional validation, and 409 conflict paths.",
        "content": API_DOC_SAMPLE,
    },
    "log-rca": {
        "label": "Checkout outage, connection pool exhaustion",
        "description": "A multi-service cascade where the loudest error is a symptom and the real trigger is four minutes earlier.",
        "content": LOG_RCA_SAMPLE,
    },
    "postmortem": {
        "label": "#incident-checkout-2026-09-14 transcript",
        "description": "A raw incident channel full of real names, a wrong hypothesis, and a mitigation that is not a fix.",
        "content": POSTMORTEM_SAMPLE,
    },
}
