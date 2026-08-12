"""Candidate public payload contains no private or credential material."""

import json
from pathlib import Path

from scripts.scan_public_payload import scan


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_public_payload_scan_passes():
    report = scan()

    assert report["status"] == "passed", report["findings"]
    assert report["tracked_pdocs_count"] == 0
    assert report["findings"] == []


def test_persisted_runtime_dependency_audit_has_no_findings():
    audit = json.loads(
        (ROOT / "evidence" / "dependency_audit.json").read_text(encoding="utf-8")
    )
    dependencies = audit.get("dependencies", audit)

    assert dependencies
    assert not [dependency for dependency in dependencies if dependency.get("vulns")]