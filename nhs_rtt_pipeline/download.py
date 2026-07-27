from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - fallback is exercised in lightweight local runtimes.
    BeautifulSoup = None

from .config import get_paths
from .rtt_parsing import financial_year_slug, parse_period_from_text


NHS_RTT_BASE_URL = "https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times"


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        attributes = {key.lower(): value for key, value in attrs}
        href = attributes.get("href")
        if href:
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            self.links.append({"href": self._current_href, "text": " ".join(" ".join(self._current_text).split())})
            self._current_href = None
            self._current_text = []


def iter_html_links(html: str) -> list[dict[str, str]]:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "lxml")
        return [
            {
                "href": str(anchor.get("href") or ""),
                "text": " ".join(anchor.get_text(" ", strip=True).split()),
            }
            for anchor in soup.find_all("a")
        ]
    parser = LinkExtractor()
    parser.feed(html)
    return parser.links


@dataclass(frozen=True)
class DownloadConfig:
    start_financial_year: int = 2015
    end_financial_year: Optional[int] = None
    data_start_month: str = "2015-10-01"
    overwrite: bool = False
    retries: int = 4
    timeout_seconds: int = 120
    user_agent: str = "Mozilla/5.0 nhs-rtt-msc-project/1.0"


def current_uk_financial_year_start(now: Optional[pd.Timestamp] = None) -> int:
    timestamp = pd.Timestamp.now(tz="Europe/London") if now is None else pd.Timestamp(now)
    return int(timestamp.year if timestamp.month >= 4 else timestamp.year - 1)


def http_get_bytes(url: str, config: DownloadConfig) -> bytes:
    headers = {"User-Agent": config.user_agent}
    last_error: Exception | None = None
    for attempt in range(1, int(config.retries) + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=int(config.timeout_seconds)) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt < int(config.retries):
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def discover_rtt_full_csv_manifest(config: DownloadConfig) -> pd.DataFrame:
    final_financial_year = (
        current_uk_financial_year_start()
        if config.end_financial_year is None
        else int(config.end_financial_year)
    )
    records: list[dict[str, object]] = []
    skipped_pages: list[dict[str, object]] = []

    for start_year in range(int(config.start_financial_year), final_financial_year + 1):
        slug = financial_year_slug(start_year)
        page_url = f"{NHS_RTT_BASE_URL}/rtt-data-{slug}/"
        try:
            html = http_get_bytes(page_url, config).decode("utf-8", errors="replace")
        except Exception as exc:
            skipped_pages.append({"financial_year": slug, "page_url": page_url, "error": str(exc)})
            continue

        for anchor in iter_html_links(html):
            link_text = " ".join(str(anchor.get("text", "")).split())
            href = anchor.get("href")
            if not href:
                continue
            if "full csv data file" not in link_text.lower():
                continue
            absolute_url = urllib.parse.urljoin(page_url, str(href))
            if ".zip" not in absolute_url.lower() and "zip" not in link_text.lower():
                continue
            period_date = parse_period_from_text(f"{link_text} {absolute_url}")
            records.append(
                {
                    "financial_year": slug,
                    "page_url": page_url,
                    "url": absolute_url,
                    "link_text": link_text,
                    "period_date": period_date,
                    "is_revised": "revised" in link_text.lower() or "revised" in absolute_url.lower(),
                    "source": "NHS England RTT waiting times full CSV data file",
                }
            )

    if not records:
        skipped = "; ".join(f"{row['financial_year']}: {row['error']}" for row in skipped_pages[:5])
        raise RuntimeError(f"No NHS RTT full CSV ZIP links were found. Skipped pages: {skipped}")

    manifest = pd.DataFrame(records).drop_duplicates(subset=["url"]).reset_index(drop=True)
    manifest["period_date"] = pd.to_datetime(manifest["period_date"], errors="coerce")
    data_start_month = pd.Timestamp(config.data_start_month).to_period("M").to_timestamp()
    manifest = manifest[
        manifest["period_date"].isna() | (manifest["period_date"] >= data_start_month)
    ].copy()

    with_period = manifest[manifest["period_date"].notna()].copy()
    without_period = manifest[manifest["period_date"].isna()].copy()
    if not with_period.empty:
        with_period = (
            with_period.sort_values(["period_date", "is_revised", "url"])
            .drop_duplicates(subset=["period_date"], keep="last")
            .sort_values("period_date")
        )
    manifest = pd.concat([with_period, without_period], ignore_index=True)
    manifest = manifest.sort_values(["period_date", "financial_year", "url"], na_position="last").reset_index(drop=True)
    manifest["discovery_timestamp"] = datetime.now().isoformat(timespec="seconds")
    if skipped_pages:
        manifest.attrs["skipped_pages"] = skipped_pages
    return manifest


def safe_zip_filename(url: str, period_date: object) -> str:
    parsed_name = Path(urllib.parse.urlparse(url).path).name
    parsed_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in parsed_name)
    if period_date is not None and not pd.isna(period_date):
        return f"{pd.Timestamp(period_date).strftime('%Y-%m')}_{parsed_name}"
    return parsed_name or hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ".zip"


def validate_zip_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"zip_valid": False, "zip_error": "missing", "csv_file_count": 0}
    if path.stat().st_size <= 0:
        return {"zip_valid": False, "zip_error": "empty file", "csv_file_count": 0}
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            csv_count = sum(1 for name in archive.namelist() if name.lower().endswith(".csv"))
            if bad_member is not None:
                return {"zip_valid": False, "zip_error": f"corrupt member: {bad_member}", "csv_file_count": csv_count}
            return {"zip_valid": True, "zip_error": "", "csv_file_count": csv_count}
    except Exception as exc:
        return {"zip_valid": False, "zip_error": str(exc), "csv_file_count": 0}


def download_manifest_zips(
    manifest: pd.DataFrame,
    zip_dir: Path,
    config: DownloadConfig,
) -> pd.DataFrame:
    zip_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(manifest)
    for index, record in enumerate(manifest.to_dict(orient="records"), start=1):
        url = str(record["url"])
        period_date = record.get("period_date")
        filename = safe_zip_filename(url, period_date)
        local_path = zip_dir / filename
        status = "existing"
        if config.overwrite or not local_path.exists() or local_path.stat().st_size == 0:
            status = "downloaded"
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            data = http_get_bytes(url, config)
            tmp_path.write_bytes(data)
            tmp_path.replace(local_path)
        validation = validate_zip_file(local_path)
        if not validation["zip_valid"]:
            raise RuntimeError(f"Downloaded RTT ZIP failed validation: {local_path} ({validation['zip_error']})")
        row = {
            **record,
            "local_path": str(local_path),
            "download_status": status,
            "file_size_bytes": int(local_path.stat().st_size),
            "download_index": index,
            "download_total": total,
            **validation,
        }
        rows.append(row)
        print(f"[{index}/{total}] {status}: {filename} ({row['file_size_bytes']:,} bytes)")
    return pd.DataFrame(rows)


def write_download_outputs(
    downloaded_manifest: pd.DataFrame,
    project_root: str | Path | None = None,
    skipped_pages: Optional[list[dict[str, object]]] = None,
) -> dict[str, Path]:
    paths = get_paths(project_root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.raw_dir / "rtt_full_csv_manifest.csv"
    summary_path = paths.raw_dir / "rtt_download_summary.json"
    downloaded_manifest.to_csv(manifest_path, index=False)

    period_values = pd.to_datetime(downloaded_manifest["period_date"], errors="coerce")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_page": NHS_RTT_BASE_URL,
        "manifest_path": str(manifest_path),
        "zip_directory": str(paths.raw_zip_dir),
        "files": int(len(downloaded_manifest)),
        "downloaded_files": int((downloaded_manifest["download_status"] == "downloaded").sum()),
        "existing_files": int((downloaded_manifest["download_status"] == "existing").sum()),
        "total_size_bytes": int(downloaded_manifest["file_size_bytes"].sum()),
        "first_period": period_values.min().date().isoformat() if period_values.notna().any() else None,
        "latest_period": period_values.max().date().isoformat() if period_values.notna().any() else None,
        "all_zip_files_valid": bool(downloaded_manifest["zip_valid"].all()),
        "skipped_pages": skipped_pages or [],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return {"manifest": manifest_path, "summary": summary_path}


def download_rtt_data(
    project_root: str | Path | None = None,
    config: DownloadConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    resolved_config = config or DownloadConfig()
    paths = get_paths(project_root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_zip_dir.mkdir(parents=True, exist_ok=True)
    manifest = discover_rtt_full_csv_manifest(resolved_config)
    skipped_pages = manifest.attrs.get("skipped_pages", [])
    downloaded = download_manifest_zips(manifest, paths.raw_zip_dir, resolved_config)
    output_paths = write_download_outputs(downloaded, project_root=paths.project_root, skipped_pages=skipped_pages)
    return downloaded, output_paths


def download_config_from_mapping(values: dict[str, object]) -> DownloadConfig:
    return DownloadConfig(
        start_financial_year=int(values.get("start_financial_year", 2015)),
        end_financial_year=(
            None if values.get("end_financial_year") in (None, "", "none") else int(values["end_financial_year"])
        ),
        data_start_month=str(values.get("data_start_month", "2015-10-01")),
        overwrite=bool(values.get("overwrite", False)),
        retries=int(values.get("retries", 4)),
        timeout_seconds=int(values.get("timeout_seconds", 120)),
        user_agent=str(values.get("user_agent", DownloadConfig.user_agent)),
    )


def download_summary(downloaded_manifest: pd.DataFrame, config: DownloadConfig) -> dict[str, object]:
    period_values = pd.to_datetime(downloaded_manifest["period_date"], errors="coerce")
    return {
        "config": asdict(config),
        "files": int(len(downloaded_manifest)),
        "downloaded_files": int((downloaded_manifest["download_status"] == "downloaded").sum()),
        "existing_files": int((downloaded_manifest["download_status"] == "existing").sum()),
        "first_period": period_values.min().date().isoformat() if period_values.notna().any() else None,
        "latest_period": period_values.max().date().isoformat() if period_values.notna().any() else None,
        "total_size_mb": round(float(downloaded_manifest["file_size_bytes"].sum()) / 1_000_000.0, 2),
    }
