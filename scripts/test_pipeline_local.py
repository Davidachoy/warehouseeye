"""Run the local WarehouseEye pipeline and assert basic tracking output."""

from __future__ import annotations

import argparse
import logging

from warehouseeye.pipeline.db import get_all_identities, init_db
from warehouseeye.pipeline.orchestrator import Orchestrator


def main() -> None:
    """Execute orchestrator on local video and validate unique tracks."""
    parser = argparse.ArgumentParser(
        description="Run the full pipeline on a local video path.",
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        metavar="VIDEO_PATH",
        help="Local path to warehouse CCTV video (optional if --video-path is set).",
    )
    parser.add_argument(
        "--video-path",
        dest="video_path_flag",
        default=None,
        help="Same as positional VIDEO_PATH (either form works).",
    )
    parser.add_argument("--base-dir", default="data", help="Output directory for artifacts.")
    parser.add_argument("--model-id", default="PekingU/rtdetr_v2_r50vd", help="RT-DETRv2 model id.")
    parser.add_argument(
        "--min-tracks",
        type=int,
        default=1,
        metavar="N",
        help="Require at least N unique identities (default 1). Use 3+ only if the clip clearly shows that many people tracked end-to-end.",
    )
    args = parser.parse_args()

    video_path = args.video_path_flag or args.video_path
    if not video_path:
        parser.error("Provide VIDEO_PATH as the first argument or use --video-path.")

    logging.basicConfig(level=logging.INFO)
    db_path = Orchestrator(base_dir=args.base_dir, model_id=args.model_id).run(video_path)
    conn = init_db(db_path)
    identities = get_all_identities(conn)
    for row in identities:
        logging.info("identity_row", extra={"row": row})
    conn.close()

    unique_tracks = {row[0] for row in identities}
    count = len(unique_tracks)
    logging.info(
        "track_count_check",
        extra={"unique_tracks": count, "min_required": args.min_tracks},
    )
    assert count >= args.min_tracks, (
        f"Expected >= {args.min_tracks} unique tracks, got {count}. "
        "If your clip has fewer visible people or sparse keyframes, lower --min-tracks."
    )


if __name__ == "__main__":
    main()

