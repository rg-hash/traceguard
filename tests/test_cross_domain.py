import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cross_domain import (
    incident_to_cross_domain_text,
    normalize_cross_domain_message,
)


def test_normalizes_hdfs_identifiers():
    raw_message = (
        "081111 042720 19510 WARN dfs.FSDataset: "
        "Unexpected error trying to delete block "
        "blk_2719230260348020339 "
        "src: /10.251.203.179:50010"
    )

    normalized = normalize_cross_domain_message(raw_message)

    assert "blk_2719230260348020339" not in normalized
    assert "10.251.203.179" not in normalized
    assert "<block_id>" in normalized
    assert "<ip_port>" in normalized
    assert "unexpected error trying to delete block" in normalized


def test_normalizes_bgl_identifiers():
    raw_message = (
        "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 "
        "2005-06-03-15.42.50.363779 "
        "R02-M1-N0-C:J12-U11 RAS KERNEL FATAL "
        "data storage interrupt"
    )

    normalized = normalize_cross_domain_message(raw_message)

    assert "1117838570" not in normalized
    assert "2005.06.03" not in normalized
    assert "r02-m1-n0-c:j12-u11" not in normalized
    assert "<date>" in normalized
    assert "<timestamp>" in normalized
    assert "<node>" in normalized
    assert "ras kernel fatal data storage interrupt" in normalized


def test_incident_text_excludes_labels_and_ids():
    incident = {
        "incident_id": "blk_2719230260348020339",
        "is_anomaly": 1,
        "root_cause": "unknown",
        "events": [
            {
                "message": (
                    "WARN dfs.FSDataset: Unexpected error trying "
                    "to delete block blk_2719230260348020339"
                )
            }
        ],
    }

    text = incident_to_cross_domain_text(incident)

    assert "blk_2719230260348020339" not in text
    assert "unknown" not in text
    assert "unexpected error trying to delete block" in text