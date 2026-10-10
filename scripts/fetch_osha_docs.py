import os
import sys
import argparse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Curated dictionary of public federal OSHA safety standards & guidance PDFs
OSHA_STANDARDS = {
    "1910.1200": {
        "title": "Hazard Communication (HazCom / GHS)",
        "cfr": "29 CFR 1910.1200",
        "filename": "29_CFR_1910.1200_hazard_communication.pdf",
        "url": "https://www.osha.gov/sites/default/files/publications/OSHA3695.pdf",
    },
    "1910.22": {
        "title": "Walking-Working Surfaces & Fall Protection",
        "cfr": "29 CFR 1910.22",
        "filename": "29_CFR_1910.22_walking_working_surfaces.pdf",
        "url": "https://www.osha.gov/sites/default/files/publications/OSHA3903.pdf",
    },
    "1910.132": {
        "title": "Personal Protective Equipment (General Requirements)",
        "cfr": "29 CFR 1910.132",
        "filename": "OSHA3151_personal_protective_equipment.pdf",
        "url": "https://www.osha.gov/sites/default/files/publications/osha3151.pdf",
    },
    "1910.134": {
        "title": "Respiratory Protection Standard",
        "cfr": "29 CFR 1910.134",
        "filename": "29_CFR_1910.134_respiratory_protection.pdf",
        "url": "https://www.osha.gov/sites/default/files/publications/OSHA3079.pdf",
    },
    "1910.146": {
        "title": "Permit-Required Confined Spaces",
        "cfr": "29 CFR 1910.146",
        "filename": "OSHA3138_confined_spaces.pdf",
        "url": "https://www.osha.gov/sites/default/files/publications/osha3138.pdf",
    },
    "1910.147": {
        "title": "Control of Hazardous Energy (Lockout/Tagout)",
        "cfr": "29 CFR 1910.147",
        "filename": "OSHA3120_lockout_tagout.pdf",
        "url": "https://www.osha.gov/sites/default/files/publications/osha3120.pdf",
    },
    "1910.119": {
        "title": "Process Safety Management of Highly Hazardous Chemicals",
        "cfr": "29 CFR 1910.119",
        "filename": "OSHA3132_process_safety_management.pdf",
        "url": "https://www.osha.gov/sites/default/files/publications/osha3132.pdf",
    },
}


def list_standards():
    print("=" * 72)
    print(" Curated Federal OSHA Standards Available for Ingestion")
    print("=" * 72)
    for key, info in OSHA_STANDARDS.items():
        print(f"  Key      : {key}")
        print(f"  Standard : {info['cfr']} — {info['title']}")
        print(f"  Filename : {info['filename']}")
        print(f"  URL      : {info['url']}")
        print("-" * 72)


def fetch_standard(standard_key: str, docs_dir: Path, force: bool = False) -> bool:
    info = OSHA_STANDARDS.get(standard_key)
    if not info:
        raise ValueError(
            f"Unknown standard key '{standard_key}'. "
            f"Valid keys: {', '.join(OSHA_STANDARDS.keys())} or 'all'."
        )

    docs_dir.mkdir(parents=True, exist_ok=True)
    target_path = docs_dir / info["filename"]

    if target_path.exists() and not force:
        print(f"[SKIP] {info['cfr']} already exists at {target_path}. Use --force to overwrite.")
        return False

    print(f"[DOWNLOADING] {info['cfr']} ({info['title']}) from {info['url']} ...")
    req = urllib.request.Request(
        info["url"],
        headers={"User-Agent": "SafetyCopilot-OSHA-Fetcher/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
        out_file.write(response.read())

    print(f"[OK] Saved {target_path} ({target_path.stat().st_size:,} bytes)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="OSHA Regulatory PDF Fetcher & Vector Store Ingestion Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_osha_docs.py --list
  python scripts/fetch_osha_docs.py --standard 1910.1200
  python scripts/fetch_osha_docs.py --standard 1910.22
  python scripts/fetch_osha_docs.py --standard all
        """
    )
    parser.add_argument(
        "--standard",
        type=str,
        help=f"Standard key to download ({', '.join(OSHA_STANDARDS.keys())}, or 'all')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all curated OSHA standards in the registry without downloading",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(ROOT_DIR / "docs"),
        help="Target directory for downloaded PDFs (default: docs/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PDF files if already present",
    )

    args = parser.parse_args()

    if args.list:
        list_standards()
        sys.exit(0)

    if not args.standard:
        parser.print_help()
        sys.exit(0)

    docs_dir = Path(args.docs_dir)
    keys_to_fetch = list(OSHA_STANDARDS.keys()) if args.standard.lower() == "all" else [args.standard]

    new_pdfs_added = 0
    for key in keys_to_fetch:
        try:
            if fetch_standard(key, docs_dir=docs_dir, force=args.force):
                new_pdfs_added += 1
        except Exception as e:
            print(f"[ERROR] Failed to fetch '{key}': {e}")
            sys.exit(1)

    if new_pdfs_added > 0:
        print(f"\n[INGEST] {new_pdfs_added} new PDF(s) added. Triggering ChromaDB ingestion...")
        from app import rag
        total_chunks = rag.ingest(str(docs_dir))
        print(f"[OK] Vector store updated. Total indexed chunks: {rag.COL.count()} (processed: {total_chunks}).")
    else:
        print("\n[INFO] No new PDFs were added; skipping vector re-ingestion.")


if __name__ == "__main__":
    main()
