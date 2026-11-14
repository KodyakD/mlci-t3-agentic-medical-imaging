"""Tests for the NIH domain-shift MCP server (read-only)."""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from t3_agentic.nih import server as nih_server


def test_nih_summary_mentions_the_gap():
    out = nih_server.nih_summary()
    assert "NIH" in out and "AUC" in out


def test_nih_findings_count_the_14_labels():
    rows = nih_server.nih_findings()
    assert len(rows) == 15   # 14 findings + multi-label marker
    for r in rows:
        if r["finding"] == "_multi_label_images":
            continue
        assert r["n_images"] >= 0


def test_nih_metrics_file_valid():
    metrics = nih_server.nih_domain_shift()
    assert "label_collapse" in metrics
    assert "per_label_ranking" in metrics
    assert metrics["n_images"] >= 100


def test_nih_case_lookup_unknown_image():
    out = nih_server.nih_case_lookup("this_image_does_not_exist.png")
    assert "error" in out
