#!/usr/bin/env python3
"""
Export Chroma chunks as Markdown files for browsing in the editor.

Output layout:
    chroma_export/
        <snapshot_id>/
            _index.md            every page in the snapshot with its chunk count
            <page-file-name>.md  all chunks of the page, sorted by section and chunk_index

Usage:
    # While the Chroma server is running (via HTTP)
    python mysolution/export_chroma_chunks.py --port 8050

    # Without a server, read the local folder directly
    python mysolution/export_chroma_chunks.py --path mysolution/.chroma

    # Export a single snapshot only
    python mysolution/export_chroma_chunks.py --port 8050 --snapshot 05_2026-09-28_latest
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import chromadb

COLLECTION = "member_knowledge_v1"
PAGE_SIZE = 1000


def open_collection(args: argparse.Namespace):
    # With a port, go through the HTTP server so two processes never open the same data folder
    if args.port is not None:
        client = chromadb.HttpClient(host=args.host, port=args.port)
    else:
        client = chromadb.PersistentClient(path=str(args.path))
    return client.get_collection(COLLECTION)


def fetch_all(collection, snapshot: str | None) -> list[tuple[str, str, dict]]:
    where = {"snapshot_id": snapshot} if snapshot else None
    rows: list[tuple[str, str, dict]] = []
    offset = 0
    while True:
        batch = collection.get(
            limit=PAGE_SIZE,
            offset=offset,
            where=where,
            include=["documents", "metadatas"],
        )
        if not batch["ids"]:
            break
        rows.extend(zip(batch["ids"], batch["documents"], batch["metadatas"]))
        offset += PAGE_SIZE
        print(f"  read {len(rows)} chunks", flush=True)
    return rows


def page_filename(url: str) -> str:
    # Readable file name from the URL, plus a short hash to keep it unique
    slug = re.sub(r"^https?://(www\.)?", "", url)
    slug = re.sub(r"\?.*$", "", slug)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", slug).strip("-").lower()[:80]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:6]
    return f"{slug or 'page'}-{digest}.md"


def write_export(rows: list[tuple[str, str, dict]], out_dir: Path) -> None:
    # Group by snapshot, then URL
    grouped: dict[str, dict[str, list[tuple[str, str, dict]]]] = defaultdict(lambda: defaultdict(list))
    for cid, doc, meta in rows:
        grouped[meta["snapshot_id"]][meta["url"]].append((cid, doc, meta))

    for snapshot_id in sorted(grouped):
        snap_dir = out_dir / snapshot_id
        snap_dir.mkdir(parents=True, exist_ok=True)
        index_lines = [f"# {snapshot_id}", "", "| Chunks | Title | File |", "|---|---|---|"]

        pages = grouped[snapshot_id]
        for url in sorted(pages):
            chunks = sorted(pages[url], key=lambda r: (str(r[2].get("section", "")), int(r[2].get("chunk_index", 0))))
            first = chunks[0][2]
            name = page_filename(url)

            lines = [
                f"# {first.get('title', '')}",
                "",
                f"- url: {url}",
                f"- snapshot: {snapshot_id} ({first.get('snapshot_date', '')})",
                f"- chunks: {len(chunks)}",
                "",
            ]
            for cid, doc, meta in chunks:
                lines += [
                    "---",
                    "",
                    f"## [{meta.get('section', '')}] chunk {meta.get('chunk_index', '')}",
                    "",
                    f"`id: {cid}`  length: {len(doc)} chars",
                    "",
                    doc,
                    "",
                ]
            (snap_dir / name).write_text("\n".join(lines), encoding="utf-8")

            title = str(first.get("title", "")).replace("|", "\\|")
            index_lines.append(f"| {len(chunks)} | {title} | [{name}]({name}) |")

        (snap_dir / "_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        print(f"{snapshot_id}: {len(pages)} pages -> {snap_dir}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Export Chroma chunks as Markdown")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, help="Chroma server port; if omitted, read --path directly")
    ap.add_argument("--path", type=Path, default=root / "mysolution" / ".chroma")
    ap.add_argument("--snapshot", help="Export only this snapshot_id")
    ap.add_argument("--out", type=Path, default=root / "chroma_export")
    args = ap.parse_args()

    # Clear the previous export first so no stale files remain
    if args.out.exists():
        shutil.rmtree(args.out)

    collection = open_collection(args)
    print(f"collection {COLLECTION} has {collection.count()} chunks", flush=True)
    rows = fetch_all(collection, args.snapshot)
    write_export(rows, args.out)


if __name__ == "__main__":
    main()
