"""Run the local WarehouseEye pipeline and assert basic tracking output."""

from __future__ import annotations

import argparse
import logging

from warehouseeye.pipeline.db import get_all_identities, init_db
from warehouseeye.pipeline.orchestrator import Orchestrator


def main() -> None:
    """Execute orchestrator on local video and validate unique tracks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True, help="Local path to warehouse CCTV video.")
    parser.add_argument("--base-dir", default="data", help="Output directory for artifacts.")
    parser.add_argument("--model-id", default="PekingU/rtdetr_v2_r50vd", help="RT-DETRv2 model id.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    db_path = Orchestrator(base_dir=args.base_dir, model_id=args.model_id).run(args.video_path)
    conn = init_db(db_path)
    identities = get_all_identities(conn)
    for row in identities:
        logging.info("identity_row", extra={"row": row})
    conn.close()

    unique_tracks = {row[0] for row in identities}
    assert len(unique_tracks) >= 3, f"Expected >=3 unique tracks, got {len(unique_tracks)}"


if __name__ == "__main__":
    main()

