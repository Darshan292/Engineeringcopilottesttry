"""Parser tests.

These assert the facts the prompts are built from. If the extraction is wrong,
every downstream stage is confidently wrong, so this is the layer worth the
most test coverage.
"""

from __future__ import annotations

import pytest

from backend.parsers.code import parse_code, parse_python
from backend.parsers.logs import mask_variables, parse_logs
from backend.parsers.routes import parse_routes
from backend.parsers.transcript import extract_real_names, parse_transcript
from backend.samples import (
    API_DOC_SAMPLE,
    LOG_RCA_SAMPLE,
    POSTMORTEM_SAMPLE,
    UNIT_TEST_SAMPLE,
)

# --- logs -----------------------------------------------------------------


def test_log_fields_are_extracted():
    ir = parse_logs(LOG_RCA_SAMPLE)
    first = ir.entries[0]
    assert first.id == "L1"
    assert first.level == "INFO"
    assert first.service == "checkout-api"
    assert first.component == "pool"
    assert first.timestamp is not None
    assert ir.stats()["parse_rate"] > 0.9


def test_all_services_are_found():
    ir = parse_logs(LOG_RCA_SAMPLE)
    assert set(ir.services) == {"checkout-api", "inventory-svc", "orders-svc", "payments-gw", "rds-proxy"}


def test_earliest_anomaly_precedes_the_first_error():
    """The whole point: the loudest error is not the earliest signal."""
    ir = parse_logs(LOG_RCA_SAMPLE)
    anomaly = next(e for e in ir.entries if e.id == ir.first_anomaly_id)
    error = next(e for e in ir.entries if e.id == ir.first_error_id)
    assert anomaly.line_no < error.line_no
    assert anomaly.level == "WARN"
    assert "slow query" in anomaly.message


@pytest.mark.parametrize(
    "line, expected_substring",
    [
        ("2026-09-14T02:11:03.441Z INFO svc-a [x] msg", "2026-09-14T02:11:03.441Z"),
        ("2026-09-14 02:11:03,441 ERROR svc-b msg", "2026-09-14 02:11:03,441"),
        ("[2026-09-14 02:11:03] WARN svc-c msg", "2026-09-14 02:11:03"),
    ],
)
def test_timestamp_formats(line, expected_substring):
    ir = parse_logs(line)
    assert ir.entries[0].raw_timestamp
    assert expected_substring in ir.entries[0].raw_timestamp


def test_stack_trace_lines_attach_to_their_entry():
    log = (
        "2026-09-14T02:16:45Z ERROR svc-a [db] connection failed\n"
        "    at com.example.Pool.get(Pool.java:42)\n"
        "    at com.example.Handler.run(Handler.java:17)\n"
        "2026-09-14T02:16:46Z INFO svc-a [db] retrying\n"
    )
    ir = parse_logs(log)
    assert len(ir.entries) == 2
    assert "Pool.java:42" in ir.entries[0].message


def test_template_mining_collapses_repetition():
    log = "\n".join(
        f"2026-09-14T02:00:{i % 60:02d}Z INFO svc [http] POST /v1/x 201 in {i}ms trace={i:08x}"
        for i in range(500)
    )
    ir = parse_logs(log)
    assert len(ir.entries) == 500
    assert len(ir.templates) == 1
    assert ir.templates[0].count == 500
    # First and last occurrence are kept; the 498 in between are not.
    assert len(ir.templates[0].example_ids) == 2


def test_mask_variables_is_stable_and_specific():
    a = mask_variables("timed out after 30000ms id=abc123 from 10.0.0.1")
    b = mask_variables("timed out after 45000ms id=xyz789 from 10.0.0.2")
    assert a == b
    assert "<DUR>" in a and "<IP>" in a


def test_summary_render_caps_severe_lines():
    log = "\n".join(f"2026-09-14T02:00:{i % 60:02d}Z ERROR svc [db] failure number {i}" for i in range(5000))
    ir = parse_logs(log)
    rendered = ir.render("summary", max_severe=50)
    assert "further severe lines omitted" in rendered
    assert len(rendered) < len(log) / 10


# --- python code ----------------------------------------------------------


def test_boundary_conditions_are_extracted_verbatim():
    """These are what turn 'imagine edge cases' into 'cover these'."""
    ir = parse_code(UNIT_TEST_SAMPLE)
    func = next(f for f in ir.all_functions() if f.name == "apply_tiered_discount")
    for condition in ("subtotal > 10000", "customer_tier != 'platinum'", "promo_code == 'SAVE20'"):
        assert condition in func.comparisons


def test_raises_are_read_from_the_ast_not_guessed():
    ir = parse_code(UNIT_TEST_SAMPLE)
    func = next(f for f in ir.all_functions() if f.name == "apply_tiered_discount")
    assert "ValueError" in func.raises
    assert "KeyError" in func.raises


def test_complexity_counts_independent_paths():
    ir = parse_code(UNIT_TEST_SAMPLE)
    func = next(f for f in ir.all_functions() if f.name == "apply_tiered_discount")
    assert func.cyclomatic_complexity >= 5
    assert func.branch_count >= 4


def test_signature_and_defaults_survive():
    ir = parse_code(UNIT_TEST_SAMPLE)
    func = next(f for f in ir.all_functions() if f.name == "apply_tiered_discount")
    names = [p.name for p in func.params]
    assert names == ["items", "customer_tier", "promo_code", "today"]
    assert func.params[2].default == "None"
    assert func.returns == "dict"


def test_third_party_calls_are_flagged_for_mocking():
    ir = parse_python(
        "import requests\nimport json\n\n"
        "def fetch(url):\n"
        "    r = requests.get(url)\n"
        "    return json.loads(r.text)\n"
    )
    func = ir.functions[0]
    assert any("requests.get" in c for c in func.external_calls)
    # stdlib is not a mocking boundary
    assert not any("json.loads" in c for c in func.external_calls)
    assert "requests" in ir.third_party_imports


def test_argument_mutation_is_detected():
    ir = parse_python("def add(items, x):\n    items.append(x)\n    return items\n")
    assert ir.functions[0].mutates_arguments


def test_syntax_errors_are_reported_not_swallowed():
    ir = parse_code("def broken(:\n    pass")
    assert ir.syntax_error
    assert ir.confidence == 0.0
    assert ir.all_functions() == []


def test_methods_are_extracted_from_classes():
    ir = parse_python("class A:\n    def m(self, x: int) -> str:\n        return str(x)\n")
    assert len(ir.classes) == 1
    assert ir.classes[0].methods[0].name == "m"
    assert len(ir.all_functions()) == 1


@pytest.mark.parametrize(
    "language, code, expected_name",
    [
        ("javascript", "function addNumbers(a, b) { return a + b; }", "addNumbers"),
        ("go", "func Divide(a int, b int) (int, error) { return a / b, nil }", "Divide"),
        ("java", "public int calcTotal(int a, int b) { return a + b; }", "calcTotal"),
    ],
)
def test_non_python_fallback_extracts_signatures_with_low_confidence(language, code, expected_name):
    ir = parse_code(code, language)
    assert expected_name in [f.name for f in ir.functions]
    # Confidence is deliberately low so the prompt qualifies its claims.
    assert ir.confidence < 0.6
    assert "regex-heuristic" in ir.parser


# --- routes ---------------------------------------------------------------


def test_routes_and_prefix_are_extracted():
    ir = parse_routes(API_DOC_SAMPLE)
    assert ir.framework == "FastAPI"
    assert ir.base_prefix == "/v1/deployments"
    assert len(ir.routes) == 3
    assert {r.method for r in ir.routes} == {"GET", "POST"}


def test_error_paths_inside_handlers_are_discovered():
    """The 409 a caller hits lives in a raise statement, not the signature."""
    ir = parse_routes(API_DOC_SAMPLE)
    create = next(r for r in ir.routes if r.handler == "create_deployment")
    statuses = {resp.status for resp in create.responses}
    assert statuses == {409, 422}
    assert all(resp.from_raise for resp in create.responses)
    assert any("already in progress" in resp.description for resp in create.responses)


def test_explicit_status_code_is_used_and_absent_one_defaults_to_200():
    ir = parse_routes(API_DOC_SAMPLE)
    create = next(r for r in ir.routes if r.handler == "create_deployment")
    rollback = next(r for r in ir.routes if r.handler == "rollback")
    assert create.success_status == 201  # declared in the decorator
    assert rollback.success_status == 200  # not declared, so the framework default


def test_pydantic_constraints_reach_the_ir():
    ir = parse_routes(API_DOC_SAMPLE)
    model = next(m for m in ir.models if m.name == "DeploymentCreate")
    replicas = next(f for f in model.fields if f.name == "replicas")
    assert replicas.constraints == {"minimum": "1", "maximum": "50"}
    environment = next(f for f in model.fields if f.name == "environment")
    assert environment.constraints["enum"] == ["staging", "production"]


def test_auth_dependencies_and_path_params_are_split_out():
    ir = parse_routes(API_DOC_SAMPLE)
    rollback = next(r for r in ir.routes if r.handler == "rollback")
    assert "require_token" in rollback.auth_dependencies
    assert [p.name for p in rollback.path_params] == ["deployment_id"]
    assert "reason" in [p.name for p in rollback.query_params]


def test_express_routes_fall_back_to_regex():
    ir = parse_routes(
        "app.get('/users/:id', (req, res) => { res.status(404).json({}); });", "javascript"
    )
    assert ir.framework == "Express"
    assert ir.routes[0].method == "GET"
    assert [p.name for p in ir.routes[0].path_params] == ["id"]


# --- transcript -----------------------------------------------------------


def test_no_real_name_survives_parsing():
    """The blameless guarantee, asserted rather than requested."""
    ir = parse_transcript(POSTMORTEM_SAMPLE)
    rendered = ir.render("full")
    names = extract_real_names(POSTMORTEM_SAMPLE)
    assert names, "the fixture should contain real names to strip"

    fragments = names | {part for name in names for part in name.split() if len(part) > 2}
    leaked = sorted(t for t in fragments if t in rendered)
    assert leaked == [], f"names leaked into the IR: {leaked}"


def test_names_inside_message_bodies_are_scrubbed_too():
    """'Marcus can you own comms?' is where naive anonymization fails."""
    ir = parse_transcript(POSTMORTEM_SAMPLE)
    addressed = next(u for u in ir.utterances if "can you own comms" in u.text)
    assert "Marcus" not in addressed.text
    assert "[COMMS]" in addressed.text


def test_roles_are_inferred_from_behaviour():
    ir = parse_transcript(POSTMORTEM_SAMPLE)
    placeholders = {s.placeholder for s in ir.speakers}
    assert "[INCIDENT_COMMANDER]" in placeholders
    assert "[SERVICE_OWNER]" in placeholders
    assert "[COMMS]" in placeholders


def test_incident_markers_are_detected():
    ir = parse_transcript(POSTMORTEM_SAMPLE)
    assert ir.detection_id == "U1"
    assert ir.mitigation_id
    assert ir.resolution_id
    ids = [u.id for u in ir.utterances]
    assert ids.index(ir.detection_id) < ids.index(ir.mitigation_id) <= ids.index(ir.resolution_id)


@pytest.mark.parametrize(
    "line",
    [
        "[02:18] Priya Raghavan: getting paged",
        "02:18 Priya Raghavan: getting paged",
        "Priya Raghavan [02:18]: getting paged",
        "<priya> getting paged",
    ],
)
def test_speaker_line_formats(line):
    ir = parse_transcript(line)
    assert len(ir.utterances) == 1
    assert "Priya" not in ir.render("full")
    assert "priya" not in ir.render("full")


def test_prose_without_speakers_yields_nothing_rather_than_guessing():
    ir = parse_transcript("This is just a paragraph of text with no speakers in it at all.")
    assert ir.utterances == []
