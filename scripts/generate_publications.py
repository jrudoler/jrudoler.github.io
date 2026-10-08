"""Convert the CV's Better BibLaTeX export to public, Jekyll-readable JSON.

Only explicit bibliographic fields are exported; notes, abstracts, attachment
paths, library tags, and manuscripts in preparation never leave the CV repo.
"""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote, urlsplit

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bibdatabase import BibDatabase
from pylatexenc.latex2text import LatexNodes2Text

CITATION_FIELDS = {
    "ID", "ENTRYTYPE", "title", "author", "date", "year", "journaltitle",
    "booktitle", "eventtitle", "publisher", "volume", "number", "pages",
    "doi", "url", "eprint", "eprinttype", "pubstate", "type", "namea", "nameatype",
}
TYPES = {"article": "Paper", "inproceedings": "Paper", "online": "Preprint",
         "dataset": "Dataset", "poster": "Presentation", "presentation": "Presentation"}


def text(value):
    return " ".join(LatexNodes2Text().latex_to_text(value).split())


def names(value):
    result = []
    for name in re.split(r"\s+and\s+(?![^{}]*\})", value):
        if name.startswith("{"):
            result.append(text(name))
        else:
            parts = name.split(",")
            result.append(text(" ".join([parts[-1].strip(), parts[0].strip()] +
                                       ([parts[1].strip()] if len(parts) == 3 else [])))
                          if len(parts) > 1 else text(name))
    return result


def public_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        return ""
    if "proxy." in parsed.hostname or parsed.hostname in {"localhost", "127.0.0.1"}:
        return ""
    if parsed.hostname == "arxiv.org":
        return "https://arxiv.org" + parsed.path.rstrip("/")
    return value


def convert(source):
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    parser.homogenise_fields = False
    db = parser.parse(source)
    # Fail instead of silently publishing a partial bibliography.
    expected = len(re.findall(r"(?im)^\s*@(?!(?:comment|string|preamble)\b)\w+\s*[{(]", source))
    if not db.entries or len(db.entries) != expected:
        raise ValueError("Bibliography is empty or contains entries that could not be parsed")
    seen = set()
    papers = []
    for entry in db.entries:
        key = entry["ID"]
        if key in seen:
            raise ValueError(f"Duplicate citation key: {key}")
        seen.add(key)
        kind = entry["ENTRYTYPE"]
        if kind == "unpublished":
            continue
        if kind not in TYPES:
            raise ValueError(f"Unsupported publication type: {kind} ({key})")
        if not entry.get("title") or not entry.get("author"):
            raise ValueError(f"Missing title or authors: {key}")
        date = entry.get("date", entry.get("year", ""))
        if not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", date):
            raise ValueError(f"Missing or unsupported date: {key}")
        doi = entry.get("doi", "")
        doi_url = "https://doi.org/" + quote(doi, safe="/.:()") if doi else ""
        url = public_url(entry.get("url", "")) or doi_url
        preprint = ""
        if entry.get("eprinttype", "").lower() == "arxiv":
            preprint = "https://arxiv.org/abs/" + quote(entry["eprint"], safe="/.")
        url = url or preprint
        venue = next((entry[f] for f in ("journaltitle", "booktitle", "eventtitle", "publisher")
                      if entry.get(f)), "")
        status = "Accepted" if entry.get("pubstate") == "inpress" else ""
        if kind == "online" or entry.get("pubstate") == "prepublished":
            status = "Preprint"
        citation = {k: v for k, v in entry.items() if k in CITATION_FIELDS}
        if url:
            citation["url"] = url
        else:
            citation.pop("url", None)
        citation_db = BibDatabase()
        citation_db.entries = [citation]
        papers.append({
            "key": key, "title": text(entry["title"]), "authors": names(entry["author"]),
            "collaborators": names(entry["namea"]) if entry.get("namea") else [],
            "date": date, "year": int(date[:4]), "type": TYPES[kind],
            "status": status, "venue": text(venue), "url": url,
            "preprint": preprint if preprint != url else "",
            "bibtex": bibtexparser.dumps(citation_db).strip(),
        })
    return sorted(papers, key=lambda p: (p["date"], p["key"]), reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bibliography", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    papers = convert(args.bibliography.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(papers)} public records: {args.output}")


if __name__ == "__main__":
    main()
