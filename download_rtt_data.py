from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhs_rtt_pipeline.download import DownloadConfig, download_rtt_data, download_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NHS England RTT full CSV ZIP datasets.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root. Defaults to this folder.")
    parser.add_argument("--start-financial-year", type=int, default=2015)
    parser.add_argument("--end-financial-year", type=int, default=None)
    parser.add_argument("--data-start-month", type=str, default="2015-10-01")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = DownloadConfig(
        start_financial_year=args.start_financial_year,
        end_financial_year=args.end_financial_year,
        data_start_month=args.data_start_month,
        overwrite=bool(args.overwrite),
    )
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent
    downloaded, output_paths = download_rtt_data(project_root=project_root, config=config)
    summary = download_summary(downloaded, config)

    print(json.dumps(summary, indent=2))
    print(f"Manifest: {output_paths['manifest']}")
    print(f"Download summary: {output_paths['summary']}")
    print(f"ZIP directory: {project_root / 'data' / 'raw' / 'zips'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
