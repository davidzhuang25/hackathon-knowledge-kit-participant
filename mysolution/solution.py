"""LangChain RAG submission for the knowledge-kit harness."""
""" 
python3 harness/run_eval.py --submission mysolution --eval eval/public_eval.jsonl 
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
import hashlib
import importlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# FlashRank turns on INFO logging globally; hide the per-request
# "INFO:httpx:HTTP Request: POST ..." lines (warnings and errors still show)
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.insert(
    0,
    str(
        next(
            parent / "harness"
            for parent in Path(__file__).resolve().parents
            if (parent / "harness" / "contract.py").is_file()
        )
    ),
)

_contract = importlib.import_module("contract")
Answer = _contract.Answer
Conflict = _contract.Conflict
Page = _contract.Page
Snapshot = _contract.Snapshot
Source = _contract.Source

INDEX_VERSION = "markdown-v4"
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 160
INGEST_BATCH_SIZE = 64
RETRIEVAL_K = 12
# Candidates fetched before near-duplicate article chunks are merged down to RETRIEVAL_K
RETRIEVAL_CANDIDATES = 50
# Packed chunks sent to FlashRank; the reranker then keeps RETRIEVAL_K
RERANK_CANDIDATES = 20
RERANK_PASSAGE_CHARS = 1500
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
# Characters of each chunk shown to the model; chunks are at most CHUNK_SIZE plus a short header
EVIDENCE_CHARS = 2000
MAX_CHUNKS_PER_URL = 2
# Distinct URLs returned in Answer.sources; the scorer gives source credit if any one matches, with no penalty for extras
MAX_SOURCES = 12
OFFICIAL_NEIGHBORS = 2
_OFFICIAL_PATH_PREFIXES = ("/about", "/personal",
                           "/business", "/contact", "/rates")
_VALUE_RE = re.compile(
    r"(?<![\w])(?:\$\s*\d[\d,]*(?:\.\d+)?|"
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|APR\b|APY\b)|"
    r"\d{1,2}:\d{2}\s*[ap]\.?m\.?)",
    re.IGNORECASE,
)
_CONFLICT_TERMS = re.compile(
    r"\b(rate|apy|apr|fee|dividend|branch|lobby|drive.?up|hours?|closing|closed|"
    r"certificate|mortgage|loan|checking|savings|credit card|minimum|maximum)\b",
    re.IGNORECASE,
)
_REFUSAL_TEXT = re.compile(
    r"\b(i don't know|i do not know|not listed|not provided|unable to confirm|"
    r"please contact|contact a)\b",
    re.IGNORECASE,
)
# Same number grammar as harness scoring.NUM_RE so extra-number refusals match the scorer.
_NUM_RE = re.compile(
    r"(\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?",
    re.IGNORECASE,
)
_RATE_QUESTION_RE = re.compile(
    r"\b(rate|rates|apr|apy|interest|fee|fees)\b",
    re.IGNORECASE,
)
_MISSING_VALUE_RE = re.compile(
    r"\b(not listed|not provided|not specified|no explicit|"
    r"isn't listed|is not listed|i don't know|i do not know)\b",
    re.IGNORECASE,
)
_STRUCTURE_ONLY_RE = re.compile(
    r"\b(adjust(?:able|ment)\s+caps?|interest[- ]rate caps?|"
    r"locked for|lock period|how an?\s+arm works|amortiz(?:ed|ation))\b",
    re.IGNORECASE,
)
_PRODUCT_RATE_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*%\s*(?:APY|APR)\b|(?:APY|APR)\s*(?:is|of|:)?\s*\d",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:%|percent)", re.IGNORECASE)
# Words just before a % value that mark it as a cap or limit, not a product rate
_CAP_CONTEXT_RE = re.compile(
    r"\b(adjust\w*|caps?|capped|within|up to|limit\w*|no more than|at most)\b",
    re.IGNORECASE,
)
_CAP_WINDOW_CHARS = 30
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RATE_REFUSAL = "That rate is not listed. Please contact a Dupaco lender."
_INFO_REFUSAL = "That information is not listed. Please contact Dupaco."
# Format A: "May 05, 2026" alone on a line, above the first "# " heading.
_PUBLISHED_A = re.compile(
    r"^\s*((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December) \d{1,2}, 20\d\d)\s*$"
)
# Format B: "Written by [Author](...) \| Aug 21, 2018 12:00:00 AM" below the title.
_PUBLISHED_B = re.compile(
    r"^\s*Written by .*?\\?\|\s*([A-Z][a-z]{2,8} \d{1,2}, 20\d\d)\b")


def extract_published_on(markdown: str) -> str:
    """Return the page publish date as YYYY-MM-DD, or "" when the page has none.

    Only the page header is searched; dates further down belong to other
    articles listed under "Handpicked for you".
    """
    lines = markdown.splitlines()
    first_heading = next((i for i, line in enumerate(
        lines) if line.startswith("# ")), None)
    header = lines[:first_heading] if first_heading is not None else lines[:10]
    for line in header:
        match = _PUBLISHED_A.match(line)
        if match:
            return datetime.strptime(match.group(1), "%B %d, %Y").date().isoformat()
    for line in lines[:15]:
        match = _PUBLISHED_B.match(line)
        if match:
            text = match.group(1)
            for fmt in ("%b %d, %Y", "%B %d, %Y"):
                try:
                    return datetime.strptime(text, fmt).date().isoformat()
                except ValueError:
                    continue
    return ""


def _chunk_key(document: Document) -> tuple:
    metadata = document.metadata
    return (
        metadata.get("url") or "",
        metadata.get("section") or "",
        int(metadata.get("chunk_index", 0)),
    )


def _is_handpicked(document: Document) -> bool:
    section = document.metadata.get("section") or ""
    return "handpicked for you" in section.lower()


def _is_thin_chunk(document: Document) -> bool:
    """True when the body is only images, links, or heading residue."""
    body = document.page_content.split("\n\n", 1)[-1]
    body = _IMAGE_RE.sub("", body)
    body = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", body)
    body = re.sub(r"[#*_`|>]+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return len(body) < 60


def _is_official_url(url: str) -> bool:
    path = urlparse(url or "").path.rstrip("/").lower()
    if not path:
        return False
    return any(
        path == prefix or path.startswith(
            prefix + "/") or path.startswith(prefix + "-")
        for prefix in _OFFICIAL_PATH_PREFIXES
    )


def _numbers_in(text: str) -> set[tuple[str, float]]:
    found: set[tuple[str, float]] = set()
    for match in _NUM_RE.finditer(text or ""):
        unit = "$" if match.group(1) else ("%" if match.group(4) else "")
        value = float(
            match.group(2).replace(",", "")
            + ("." + match.group(3) if match.group(3) else "")
        )
        found.add((unit, round(value, 4)))
    return found


def _looks_like_structure_only(text: str) -> bool:
    """True when the draft describes ARM caps/lock/amortization but not a product APY/APR."""
    if not _STRUCTURE_ONLY_RE.search(text or ""):
        return False
    return not _PRODUCT_RATE_RE.search(text or "")


def _has_rate_percent(text: str, question: str) -> bool:
    """True when the text states a % value that is a rate, not an ARM cap or limit, and is not from the question."""
    question_numbers = _numbers_in(question or "")
    for match in _PERCENT_RE.finditer(text or ""):
        if _numbers_in(match.group(0)) <= question_numbers:
            continue
        window = (text or "")[
            max(0, match.start() - _CAP_WINDOW_CHARS):match.start()]
        if not _CAP_CONTEXT_RE.search(window):
            return True
    return False


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _names_past_year(question: str, snapshot_date: Optional[str]) -> bool:
    """True when the question names a year before the snapshot year (a history question)."""
    try:
        current = int((snapshot_date or "")[:4])
    except ValueError:
        current = date.today().year
    return any(int(year) < current for year in _YEAR_RE.findall(question or ""))


def _finalize_answer_text(question: str, text: str, confident: bool) -> tuple[str, bool]:
    """Replace missing-rate drafts with a number-free refusal; keep complete rate answers intact."""
    text = (text or "").strip()
    rate_question = bool(_RATE_QUESTION_RE.search(question or ""))
    missing = bool(_MISSING_VALUE_RE.search(text))
    # A confident draft that gives rate values keeps them even if it says some other item is not listed
    missing_only = missing and not (
        confident and _has_rate_percent(text, question))
    if rate_question and (not confident or missing_only or _looks_like_structure_only(text)):
        return _RATE_REFUSAL, False
    if not confident:
        extra = _numbers_in(text) - _numbers_in(question or "")
        if extra:
            return _INFO_REFUSAL, False
        if not _REFUSAL_TEXT.search(text):
            text = f"I don't know. {text}".strip()
        return text, False
    return text, confident


class DraftAnswer(BaseModel):
    text: str = Field(
        description=(
            "Member-facing answer grounded only in supplied evidence. Be specific: name the "
            "states, places or items a general term covers, give exact URLs, phone numbers or "
            "addresses for how/where questions, and copy counts and ordinal words word for word."
        )
    )
    evidence_ids: list[int] = Field(
        default_factory=list,
        description="The [n] numbers of the evidence items the answer is based on, most important first",
    )
    confident: bool = Field(
        description="False when the evidence does not support an answer")
    conflict_note: Optional[str] = Field(
        default=None, description="Explain which dated source was preferred if evidence conflicts"
    )


class FactObservation(NamedTuple):
    snapshot_id: str
    snapshot_date: str
    url: str
    title: str
    section: str
    value: str
    quote: str


def select_snapshot(snapshots: dict[str, str], as_of: Optional[str]) -> Optional[tuple[str, str]]:
    """Return the latest full-site snapshot on or before the requested date."""
    target = date.fromisoformat(as_of) if as_of else None
    candidates = []
    for snapshot_id, taken_on in snapshots.items():
        snapshot_date = date.fromisoformat(taken_on)
        if target is None or snapshot_date <= target:
            candidates.append((snapshot_date, snapshot_id, taken_on))
    if not candidates:
        return None
    _, snapshot_id, taken_on = max(candidates)
    return snapshot_id, taken_on


_LABEL_MAX_CHARS = 60
_CATEGORY_TAG_RE = re.compile(r"\(category: ([^()]+(?:\([^()]*\))?)\)")


def _strip_category_tags(text: str) -> str:
    """Remove ingest-time '(category: ...)' tags from member-facing text."""
    return re.sub(r"[ \t]*" + _CATEGORY_TAG_RE.pattern, "", text)


def _is_label_line(line: str) -> bool:
    """A short plain-text line: not a table row, heading, image, link, list item or sentence."""
    text = line.strip()
    return (
        0 < len(text) <= _LABEL_MAX_CHARS
        and not re.match(r"^[|#!\[\-*>\d]", text)
        and not text.endswith((".", ":", "?", "!", ","))
    )


def _annotate_table_labels(lines: list[str]) -> list[str]:
    """Mark product labels above tables with the category label above them.

    A label line directly followed by a table is a product label; a label line directly
    followed by a product label is a category label. The category applies to every
    product until the next category or the end of the section.
    """
    nonblank = [i for i, line in enumerate(lines) if line.strip()]
    following = dict(zip(nonblank, nonblank[1:]))

    def is_product(i: int) -> bool:
        return (_is_label_line(lines[i]) and i in following
                and lines[following[i]].lstrip().startswith("|"))

    annotated, category = list(lines), ""
    for i in nonblank:
        if is_product(i):
            name = lines[i].strip()
            if category and category != name:
                annotated[i] = f"{name} (category: {category})"
        elif _is_label_line(lines[i]) and i in following and is_product(following[i]):
            category = lines[i].strip()
    return annotated


class MarkdownChunker:
    """Split Markdown by headings and paragraphs without separating ordinary table rows."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, page: Page, snapshot: Snapshot) -> list[Document]:
        sections: list[tuple[str, list[str]]] = []
        heading_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []

        def flush() -> None:
            if current_lines:
                section = " > ".join(
                    title for _, title in heading_stack) or page.title
                sections.append((section, current_lines.copy()))
                current_lines.clear()

        for line in page.markdown.splitlines():
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                flush()
                level, title = len(heading.group(1)), heading.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
            else:
                current_lines.append(line)
        flush()
        if not sections:
            sections.append((page.title, page.markdown.splitlines()))

        # Page-level date shared by every chunk; Chroma rejects None, so "" means unknown.
        published_on = extract_published_on(page.markdown)
        documents = []
        for section, lines in sections:
            lines = _annotate_table_labels(lines)
            blocks = [block.strip() for block in re.split(
                r"\n\s*\n", "\n".join(lines)) if block.strip()]
            chunks: list[str] = []
            pending: list[str] = []
            pending_size = 0
            for block in blocks:
                if len(block) > self.chunk_size:
                    if pending:
                        chunks.append("\n\n".join(pending))
                        pending, pending_size = [], 0
                    chunks.extend(self._split_oversized_block(block))
                elif pending and pending_size + len(block) + 2 > self.chunk_size:
                    chunks.append("\n\n".join(pending))
                    pending, pending_size = [block], len(block)
                else:
                    pending.append(block)
                    pending_size += len(block) + 2
            if pending:
                chunks.append("\n\n".join(pending))

            for chunk_index, chunk in enumerate(chunks):
                # Add the table categories in this chunk to its section path
                categories = list(dict.fromkeys(
                    _CATEGORY_TAG_RE.findall(chunk)))
                chunk_section = (
                    f"{section} > {' / '.join(categories)}" if categories else section
                )
                content = f"{page.title}\nSection: {chunk_section}\n\n{chunk}"
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "snapshot_id": snapshot.snapshot_id,
                            "snapshot_date": snapshot.taken_on,
                            "url": page.url,
                            "title": page.title,
                            "section": chunk_section,
                            "chunk_index": chunk_index,
                            "published_on": published_on,
                        },
                    )
                )
        return documents

    def _split_oversized_block(self, block: str) -> list[str]:
        lines = block.splitlines()
        if lines and all(line.lstrip().startswith("|") for line in lines):
            header = lines[:2] if len(lines) > 1 and re.match(
                r"^\s*\|?\s*:?-+", lines[1]) else lines[:1]
            rows = lines[len(header):]
            pieces, current = [], header.copy()
            current_size = sum(len(line) + 1 for line in current)
            for row in rows:
                if current_size + len(row) + 1 > self.chunk_size and len(current) > len(header):
                    pieces.append("\n".join(current))
                    current, current_size = header.copy(), sum(len(line) + 1 for line in header)
                current.append(row)
                current_size += len(row) + 1
            if current:
                pieces.append("\n".join(current))
            if pieces:
                return pieces

        pieces = []
        start = 0
        while start < len(block):
            end = min(start + self.chunk_size, len(block))
            if end < len(block):
                boundary = block.rfind(" ", start + self.chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            pieces.append(block[start:end].strip())
            if end >= len(block):
                break
            start = max(start + 1, end - self.overlap)
        return [piece for piece in pieces if piece]


class SnapshotConflictAnalyzer:
    """Report repeated, similarly worded rate/fee facts that changed over time."""

    def __init__(self):
        self._observations: dict[tuple[str, str, str],
                                 list[FactObservation]] = defaultdict(list)
        self._seen: set[tuple[str, str, str, str, str]] = set()

    def ingest(self, snapshot: Snapshot) -> None:
        for page in snapshot.pages:
            section = page.title
            for line in page.markdown.splitlines():
                heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
                if heading:
                    section = heading.group(1).strip()
                if not _CONFLICT_TERMS.search(line):
                    continue
                values = list(dict.fromkeys(match.group(0).strip()
                              for match in _VALUE_RE.finditer(line)))
                if not values:
                    continue
                normalized = _VALUE_RE.sub(" <value> ", line.lower())
                normalized = re.sub(r"[^a-z0-9<>]+", " ", normalized).strip()
                if len(normalized) < 8:
                    continue
                key = (page.url, section.lower(), normalized)
                value = ", ".join(values)
                seen_key = (*key, snapshot.snapshot_id, value)
                if seen_key in self._seen:
                    continue
                self._seen.add(seen_key)
                self._observations[key].append(
                    FactObservation(
                        snapshot_id=snapshot.snapshot_id,
                        snapshot_date=snapshot.taken_on,
                        url=page.url,
                        title=page.title,
                        section=section,
                        value=value,
                        quote=line.strip()[:500],
                    )
                )

    def conflicts(self) -> list[Conflict]:
        reports = []
        for (_, _, normalized), observations in self._observations.items():
            values = list(dict.fromkeys(
                observation.value for observation in observations))
            if len(values) < 2:
                continue
            latest_date = max(
                observation.snapshot_date for observation in observations)
            latest = [
                observation for observation in observations if observation.snapshot_date == latest_date]
            latest_values = list(dict.fromkeys(
                observation.value for observation in latest))
            resolution = latest_values[0] if len(latest_values) == 1 else None
            representative = latest[0]
            sources = []
            source_keys = set()
            for observation in sorted(observations, key=lambda item: (item.snapshot_date, item.snapshot_id)):
                source_key = (observation.snapshot_id, observation.url)
                if source_key in source_keys:
                    continue
                source_keys.add(source_key)
                sources.append(
                    Source(
                        url=observation.url, snapshot_id=observation.snapshot_id, quote=observation.quote)
                )
            reason = (
                f"Selected the value from the latest snapshot ({latest_date})."
                if resolution
                else f"The latest snapshot ({latest_date}) contains multiple values; no winner selected."
            )
            reports.append(
                Conflict(
                    topic=f"{representative.title}: {representative.section}: {normalized[:160]}",
                    values=values,
                    sources=sources,
                    resolution=resolution,
                    reason=reason,
                )
            )
        return reports


class KnowledgeAssistant:
    def __init__(
        self,
        *,
        vector_store=None,
        chat_model=None,
        embeddings=None,
        persist_directory: Optional[Path] = None,
        retrieval_k: int = RETRIEVAL_K,
        reranker=None,
    ):
        self._snapshots: dict[str, str] = {}
        self._chunker = MarkdownChunker()
        self._conflict_analyzer = SnapshotConflictAnalyzer()
        self._retrieval_k = retrieval_k

        if vector_store is None:
            embedding_model = os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            # Ingest bursts can exceed the embedding tokens-per-minute limit; the
            # OpenAI client backs off on 429 using the server's retry-after hint.
            embeddings = embeddings or OpenAIEmbeddings(
                model=embedding_model, max_retries=10)
            database_path = persist_directory or (
                Path(__file__).resolve().parent / ".chroma")
            database_path.mkdir(parents=True, exist_ok=True)
            self._state_path = database_path / "index_state.json"
            self._index_signature = f"{INDEX_VERSION}:{embedding_model}"
            self._vector_store = Chroma(
                collection_name="member_knowledge_v1",
                embedding_function=embeddings,
                persist_directory=str(database_path),
            )
            self._reranker = reranker or Ranker(model_name=RERANK_MODEL)
        else:
            self._vector_store = vector_store
            self._state_path = None
            self._index_signature = "injected"
            self._reranker = reranker

        if chat_model is None:
            chat_model = ChatOpenAI(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
                temperature=0,
                timeout=25,
                max_retries=1,
            )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You answer member questions using only the supplied dated website evidence. "
                    "Do not use outside knowledge. Preserve product, term, balance-tier, APR/APY, "
                    "and effective-date qualifiers. For a question about current information, the "
                    "evidence is from the latest applicable full-site snapshot. If it does not "
                    "support a precise answer, say 'I don't know' and set confident=false. Do not "
                    "list stale alternatives unless the question asks about a conflict. Cite no "
                    "invented URLs. Keep the answer concise. Each evidence item may show a "
                    "published date. When items give different values for the same fact (not "
                    "different tiers, terms, balances or products), prefer the item with the "
                    "latest published date. Official rate and product pages without a published "
                    "date are current as of the snapshot date and take priority over articles for "
                    "rates and fees. Never use the snapshot date as a publish date. When the "
                    "question asks for a rate, fee or hours, name the matching product and list "
                    "every tier, term or balance range given for it in the evidence, with each "
                    "value. If the evidence has no explicit rate or fee for the exact product "
                    "asked about, say it is not listed, set confident=false, and do not mention "
                    "any numbers, including related terms, caps, limits or ranges. Only give "
                    "details for a specific branch, product or location when the evidence names it "
                    "explicitly; never assume an unnamed item in the evidence is the one asked "
                    "about. If it is not named, say it is not listed and set confident=false. "
                    "When the evidence uses an ordinal word, repeat that ordinal word; do not "
                    "rewrite it as a cardinal number."
                    "\n\nScope: answer only the product or item asked about. Do not mention "
                    "other products, and do not say what is not listed for them. When the "
                    "question names a product or product category, answer from the table or "
                    "section for that exact product; values from other products (even ones "
                    "with a similar name) never answer it. For a yes/no question about a "
                    "value, answer for the named product first and give its relevant value; "
                    "if the value appears only in a different product, say no and name the "
                    "product where it does appear. When the answer to a yes/no question about "
                    "a value is no, also give the top tier and its value for every product "
                    "listed under the named category in the evidence, without skipping any "
                    "(for example 'No. X tops out at A% APY, Y at B% APY and Z at C% APY.'), "
                    "so the member sees what that category does pay. When the question names a category (for "
                    "example savings, checking or loans), use only the products listed under "
                    "that category in the evidence; a product marked with a different "
                    "'(category: ...)' is not in it, even if its name or links contain the word. "
                    "For questions that ask what products you offer or to list them (for example "
                    "'what ... do you offer', 'list all', 'what are the rates for the ...'), "
                    "include every matching product in the evidence, including business and "
                    "personal versions, each with its values. Do not copy '(category: ...)' "
                    "tags into the answer."
                    "\n\nBe specific: when you use a general term (charter area, eligibility, "
                    "service area), list the specific states, counties or names the evidence "
                    "gives for it. For how or where questions, give the exact URL, phone number "
                    "or address from the evidence. Copy counts and ordinals word for word. When "
                    "the evidence says how many times something has happened, include that "
                    "phrase from the most recently published item. For awards, recognitions and "
                    "yearly figures, report only the most recently published item, unless the "
                    "question names a specific earlier year or date (then report the item for "
                    "that year) or asks for history. For questions about how to join, apply or "
                    "sign up online, give the direct sign-up or application link from the "
                    "evidence rather than an information page. When the question asks which "
                    "link to use to see information, give the URL written in the evidence text "
                    "for it, not the page URL from an evidence item's header line."
                    "\n\nIn evidence_ids, list the [n] numbers of the evidence items you used."
                    "\n\nEvidence:\n{context}",
                ),
                ("human", "Question: {question}"),
            ]
        )
        retrieval_step = RunnableLambda(self._retrieve_and_prepare).with_config(
            {"run_name": "Retrieve evidence from selected snapshot"}
        )
        draft_chain = prompt | chat_model.with_structured_output(DraftAnswer)
        self._answer_pipeline = (
            retrieval_step | RunnablePassthrough.assign(draft=draft_chain)
        ).with_config({"run_name": "Member knowledge answer"})

    def _read_index_state(self) -> dict:
        if self._state_path is None or not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_index_state(self, state: dict) -> None:
        if self._state_path is None:
            return
        temporary_path = self._state_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(
            state, sort_keys=True), encoding="utf-8")
        temporary_path.replace(self._state_path)

    def ingest(self, snapshot: Snapshot) -> None:
        self._snapshots[snapshot.snapshot_id] = snapshot.taken_on
        self._conflict_analyzer.ingest(snapshot)
        state = self._read_index_state()
        indexed = state.get("indexed", {}) if state.get(
            "signature") == self._index_signature else {}
        if indexed.get(snapshot.snapshot_id) == snapshot.taken_on:
            return

        if self._state_path is not None:
            self._vector_store.delete(
                where={"snapshot_id": snapshot.snapshot_id})

        batch_documents: list[Document] = []
        batch_ids: list[str] = []
        chunk_ordinal = 0
        for page in snapshot.pages:
            for document in self._chunker.split(page, snapshot):
                identity = "\0".join(
                    (
                        snapshot.snapshot_id,
                        page.url,
                        str(document.metadata["section"]),
                        str(document.metadata["chunk_index"]),
                        str(chunk_ordinal),
                    )
                )
                batch_ids.append(hashlib.sha256(
                    identity.encode("utf-8")).hexdigest())
                batch_documents.append(document)
                chunk_ordinal += 1
                if len(batch_documents) >= INGEST_BATCH_SIZE:
                    self._vector_store.add_documents(
                        batch_documents, ids=batch_ids)
                    batch_documents, batch_ids = [], []
        if batch_documents:
            self._vector_store.add_documents(batch_documents, ids=batch_ids)

        indexed[snapshot.snapshot_id] = snapshot.taken_on
        self._write_index_state(
            {"signature": self._index_signature, "indexed": indexed})

    @staticmethod
    def _duplicate_key(document: Document) -> str:
        """Chunk body without the title/section header, with every number masked."""
        body = document.page_content.split("\n\n", 1)[-1]
        body = re.sub(r"\s+", " ", body.lower())
        return re.sub(r"\d[\d,.]*", "<n>", body)[:600]

    def _merge_duplicates(self, scored: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        """Collapse near-identical dated article chunks (e.g. press-release boilerplate) to the newest copy.

        Undated chunks (product and rate pages) are never merged, because different products can
        share wording while carrying different values.
        """
        merged: list[list] = []
        groups: dict[str, list] = {}
        for document, distance in scored:
            if not document.metadata.get("published_on"):
                merged.append([document, distance])
                continue
            key = self._duplicate_key(document)
            entry = groups.get(key)
            if entry is None:
                entry = [document, distance]
                groups[key] = entry
                merged.append(entry)
                continue
            # Keep the group at its best rank, but show the most recently published copy
            entry[1] = min(entry[1], distance)
            if document.metadata["published_on"] > entry[0].metadata["published_on"]:
                entry[0] = document
        merged.sort(key=lambda pair: pair[1])
        return [(document, distance) for document, distance in merged]

    @staticmethod
    def _keep_latest_article_series(scored: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        """Drop older dated articles that share a title with a newer one (e.g. a yearly press-release series).

        Undated chunks (product and rate pages) are always kept.
        """
        latest: dict[str, str] = {}
        for document, _distance in scored:
            published = document.metadata.get("published_on") or ""
            if not published:
                continue
            title = re.sub(
                r"\s+", " ", (document.metadata.get("title") or "").lower()).strip()
            if title and published > latest.get(title, ""):
                latest[title] = published
        kept = []
        for document, distance in scored:
            published = document.metadata.get("published_on") or ""
            title = re.sub(
                r"\s+", " ", (document.metadata.get("title") or "").lower()).strip()
            if published and title and published < latest.get(title, ""):
                continue
            kept.append((document, distance))
        return kept

    @staticmethod
    def _cap_chunks_per_url(
        scored: list[tuple[Document, float]],
        limit: int,
        per_url: int = MAX_CHUNKS_PER_URL,
    ) -> list[tuple[Document, float]]:
        """Keep at most per_url chunks from each URL, walking closest distance first."""
        selected: list[tuple[Document, float]] = []
        counts: dict[str, int] = defaultdict(int)
        for document, distance in scored:
            url = document.metadata.get("url") or ""
            if counts[url] >= per_url:
                continue
            selected.append((document, distance))
            counts[url] += 1
            if len(selected) >= limit:
                break
        return selected

    def _page_chunks(self, snapshot_id: str, url: str) -> list[Document]:
        """Return every indexed chunk for a URL in ingest (page) order."""
        result = self._vector_store.get(
            where={"$and": [{"snapshot_id": snapshot_id}, {"url": url}]},
            include=["metadatas", "documents"],
        )
        documents = []
        for content, metadata in zip(
            result.get("documents") or [],
            result.get("metadatas") or [],
        ):
            if content is None:
                continue
            documents.append(
                Document(page_content=content, metadata=metadata or {}))
        return documents

    def _previous_official_neighbors(
        self,
        url: str,
        snapshot_id: str,
        selected: list[tuple[Document, float]],
        selected_keys: set[tuple],
    ) -> list[tuple[Document, float]]:
        """Attach up to OFFICIAL_NEIGHBORS earlier page chunks of an official URL.

        Previous substantial sections of the same page often hold parent context
        (for example the states listed above an eligibility section).
        """
        best_document = None
        best_distance = None
        for document, distance in selected:
            if document.metadata.get("url") != url:
                continue
            if best_document is None or distance < best_distance:
                best_document = document
                best_distance = distance
        if best_document is None:
            return []
        page_docs = self._page_chunks(snapshot_id, url)
        if not page_docs:
            return []
        best_key = _chunk_key(best_document)
        position = next(
            (index for index, document in enumerate(
                page_docs) if _chunk_key(document) == best_key),
            None,
        )
        if position is None:
            return []
        neighbors: list[tuple[Document, float]] = []
        for index in range(position - 1, -1, -1):
            if len(neighbors) >= OFFICIAL_NEIGHBORS:
                break
            document = page_docs[index]
            if _is_handpicked(document) or _is_thin_chunk(document):
                continue
            if _chunk_key(document) in selected_keys:
                continue
            neighbors.append(
                (document, (best_distance or 0.0) + 0.0001 * (position - index))
            )
        return neighbors

    def _attach_official_neighbors(
        self,
        selected: list[tuple[Document, float]],
        request: dict,
    ) -> list[tuple[Document, float]]:
        """Keep k slots, swapping in 1-2 same-URL neighbors for official undated pages."""
        if not selected:
            return selected
        selected_keys = {_chunk_key(document) for document, _ in selected}
        extras: list[tuple[Document, float]] = []
        seen_urls: list[str] = []
        for document, _distance in selected:
            url = document.metadata.get("url") or ""
            if not url or url in seen_urls:
                continue
            if not _is_official_url(url) or document.metadata.get("published_on"):
                continue
            seen_urls.append(url)
            for neighbor, neighbor_distance in self._previous_official_neighbors(
                url, request["snapshot_id"], selected, selected_keys
            ):
                key = _chunk_key(neighbor)
                if key in selected_keys:
                    continue
                extras.append((neighbor, neighbor_distance))
                selected_keys.add(key)
        if not extras:
            return selected
        combined = list(selected)
        combined_keys = {_chunk_key(document) for document, _ in combined}
        for document, distance in extras:
            key = _chunk_key(document)
            if key in combined_keys:
                continue
            extra_url = document.metadata.get("url")
            if len(combined) < self._retrieval_k:
                combined.append((document, distance))
                combined_keys.add(key)
                continue
            drop_at = None
            worst = -1.0
            for index, (existing, existing_distance) in enumerate(combined):
                if existing.metadata.get("url") == extra_url:
                    continue
                if existing_distance >= worst:
                    worst = existing_distance
                    drop_at = index
            if drop_at is None:
                continue
            combined_keys.discard(_chunk_key(combined[drop_at][0]))
            combined.pop(drop_at)
            combined.append((document, distance))
            combined_keys.add(key)
        combined.sort(key=lambda pair: pair[1])
        return combined

    def _rerank(
        self,
        question: str,
        scored: list[tuple[Document, float]],
        top_n: int,
    ) -> list[tuple[Document, float]]:
        """Reorder packed chunks with FlashRank and keep the top_n most relevant."""
        if not scored:
            return scored
        passages = [
            {"id": index, "text": document.page_content[:RERANK_PASSAGE_CHARS]}
            for index, (document, _distance) in enumerate(scored)
        ]
        ranked = self._reranker.rerank(
            RerankRequest(query=question, passages=passages)
        )
        selected: list[tuple[Document, float]] = []
        seen: set[int] = set()
        for item in ranked:
            index = item["id"]
            if index in seen:
                continue
            seen.add(index)
            document, _chroma_distance = scored[index]
            selected.append((document, 1.0 - float(item["score"])))
            if len(selected) >= top_n:
                break
        return selected

    def _retrieve_and_prepare(self, request: dict) -> dict:
        scored = self._vector_store.similarity_search_with_score(
            request["question"],
            k=RETRIEVAL_CANDIDATES,
            filter={"snapshot_id": request["snapshot_id"]},
        )
        scored = [
            (document, distance)
            for document, distance in scored
            if not _is_handpicked(document)
        ]
        scored = self._merge_duplicates(scored)
        # A question about an earlier year needs the older articles in a yearly series
        if not _names_past_year(request["question"], request.get("snapshot_date")):
            scored = self._keep_latest_article_series(scored)
        packed = self._cap_chunks_per_url(scored, RERANK_CANDIDATES)
        selected = self._rerank(request["question"], packed, self._retrieval_k)
        selected = self._attach_official_neighbors(selected, request)
        documents = [document for document, _ in selected]
        distances = [distance for _, distance in selected]
        context_parts = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            context_parts.append(
                f"[{index}] {metadata.get('title', '')} | {metadata.get('section', '')} | "
                f"{metadata.get('url', '')} | published: {metadata.get('published_on') or 'unknown'} | "
                f"snapshot: {metadata.get('snapshot_date', '')}\n"
                f"{document.page_content[:EVIDENCE_CHARS]}"
            )
        return {
            **request,
            "documents": documents,
            "distances": distances,
            "context": "\n\n".join(context_parts),
        }

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        selected = select_snapshot(self._snapshots, as_of)
        if selected is None:
            return Answer(text="I don't know; no snapshot is available for that date.", confident=False)
        snapshot_id, snapshot_date = selected
        result = self._answer_pipeline.invoke(
            {
                "question": question,
                "snapshot_id": snapshot_id,
                "snapshot_date": snapshot_date,
                "as_of": as_of,
            },
            config={
                "run_name": "Answer member question",
                "tags": ["aisolution", "rag"],
                "metadata": {"snapshot_id": snapshot_id, "as_of": as_of or snapshot_date},
            },
        )
        draft = result["draft"]
        documents = result["documents"]
        if not documents:
            return Answer(text="I don't know; the selected snapshot does not contain supporting information.", confident=False)

        # Ingest-time category tags are for the model only; never show them to the member
        draft_text = _strip_category_tags(draft.text or "")
        text, confident = _finalize_answer_text(
            question, draft_text, draft.confident)
        # Cite the evidence the model says it used first, then fill by retrieval rank
        used = [
            documents[index - 1]
            for index in dict.fromkeys(draft.evidence_ids or [])
            if isinstance(index, int) and 1 <= index <= len(documents)
        ]
        sources = []
        seen_urls = set()
        for document in used + documents:
            metadata = document.metadata
            url = metadata.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                Source(
                    url=url,
                    snapshot_id=metadata.get("snapshot_id"),
                    quote=_strip_category_tags(document.page_content)[
                        :350].strip(),
                )
            )
            if len(sources) >= MAX_SOURCES:
                break
        return Answer(
            text=text or "I don't know.",
            sources=sources,
            confident=confident,
            conflict_note=draft.conflict_note,
        )

    def conflicts(self) -> list[Conflict]:
        return self._conflict_analyzer.conflicts()


def build() -> KnowledgeAssistant:
    return KnowledgeAssistant()
