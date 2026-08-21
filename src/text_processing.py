from __future__ import annotations

from pathlib import Path
import re

from .schemas import ChunkRecord, RawTextNode
from .utils import chunk_id_for, normalize_text, split_text_with_overlap
from .bangla_corrector import correct_bangla_text


BANGLA_MONTHS = [
    "জানুয়ারি",
    "ফেব্রুয়ারি",
    "মার্চ",
    "এপ্রিল",
    "মে",
    "জুন",
    "জুলাই",
    "আগস্ট",
    "সেপ্টেম্বর",
    "অক্টোবর",
    "নভেম্বর",
    "ডিসেম্বর",
]
BANGLA_MONTH_ALIASES = {"এপ্রল": "এপ্রিল", "ফেব্রয়ারি": "ফেব্রুয়ারি", "জানুয়ারী": "জানুয়ারি"}

ENGLISH_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


COMMON_BANGLA_FIXES = {
    "বাংলা দেশ": "বাংলাদেশ",
    "বাংলা=দশ": "বাংলাদেশ",
    "বাংলা=দে=শ": "বাংলাদেশ",
    "বৈাংলাদশ": "বাংলাদেশ",
    "বাংলাদশ": "বাংলাদেশ",
    "ঋতিু": "ঋতু",
    "খতু": "ঋতু",
    "বৈশষ্ট্য": "বৈশিষ্ট্য",
    "বশেষ্ট্য": "বৈশিষ্ট্য",
    "বৈশেখ": "বৈশাখ",
    "আবৈহাওয়া": "আবহাওয়া",
    "বৈৃষ্টি": "বৃষ্টি",
    "পৌষি": "পৌষ",
    "আষিাঢ়": "আষাঢ়",
    "চিত্র": "চৈত্র",
    "বৈসন্তে": "বসন্ত",
    "শীতি": "শীত",
    "হমন্তে": "হেমন্ত",
    "বৈষির্তিা": "বর্ষা",
    "গ্রীষ্মে": "গ্রীষ্ম",
    "কাশফ ুল": "কাশফুল",
    "কাশফু ল": "কাশফুল",
    "পঠাপুল": "পিঠাপুলি",
    "পঠাপুǬল": "পিঠাপুলি",
    "শমুল": "শিমুল",
    "ফু ল": "ফুল",
    "ফুটা": "ফোটা",
    "ফাটা": "ফোটা",
}


# Extra high-confidence repairs seen across generated Bangla PDFs, scanned
# government guidelines, BCS slide PDFs, and NCTB textbook scans. Keep this
# conservative: it is not a dictionary for one PDF, it only fixes recurring
# font/OCR artifacts that damage retrieval.
COMMON_BANGLA_FIXES.update({
    "বাংলো": "বাংলা",
    "বাংলা=দে=শ": "বাংলাদেশ",
    "বাংলা=দ=শ": "বাংলাদেশ",
    "বাংলো=দ=শ": "বাংলাদেশ",
    "বাংলা=দ=শর": "বাংলাদেশের",
    "বাংলা=দে=শের": "বাংলাদেশের",
    "ভিাষা": "ভাষা",
    "ভূ7মেকা": "ভূমিকা",
    "ই7তিহাস": "ইতিহাস",
    "ঐ7তিহা7সিক": "ঐতিহাসিক",
    "7বৈ": "বি",
    "7শ": "শি",
    "কৃ 7ষ": "কৃষি",
    "প্রকৃ 7তি": "প্রকৃতি",
    "মানু=ষির": "মানুষের",
    "মানু=ষর": "মানুষের",
    "বৈছ=র": "বছরে",
    "ঋতিু": "ঋতু",
    "ঋতিুর": "ঋতুর",
    "বৈছর": "বছর",
    "বৈলা": "বলা",
    "বৈলা হয়": "বলা হয়",
    "ǂদেশ": "দেশ",
    "ǂদশ": "দেশ",
    "দেশের্শ=করি": "দর্শকের",
    "সংǬক্ষিপ্ত": "সংক্ষিপ্ত",
    "আ=লাচনা": "আলোচনা",
    "ǃবৈশিষ্ট্য": "বৈশিষ্ট্য",
    "ǃবৈ7শষ্ট্য": "বৈশিষ্ট্য",
    "ǃব7শেষ্ট্য": "বৈশিষ্ট্য",
    "ǃবৈ7শেষ্ট্য": "বৈশিষ্ট্য",
    "বৈ7শষ্ট্য": "বৈশিষ্ট্য",
    "বৈ7শেষ্ট্য": "বৈশিষ্ট্য",
    "বৈশেষ্ট্য": "বৈশিষ্ট্য",
    "তথে্যু": "তথ্য",
    "ǂট7বলে": "টেবিল",
    "ǂট7বল": "টেবিল",
    "তাǬলকা": "তালিকা",
    "সার7ণ": "সারণি",
    "ǂফ্লা চাটর্য": "ফ্লো চার্ট",
    "চাটর্য": "চার্ট",
    "প্রোথমিক": "প্রাথমিক",
    "প্রেধান": "প্রধান",
    "দেু র্যাগ": "দুর্যোগ",
    "দেুর্যাগ": "দুর্যোগ",
    "দু যর্ণোগ": "দুর্যোগ",
    "দুযর্ণোগ": "দুর্যোগ",
    "দুর্যাগ": "দুর্যোগ",
    "ব্যবস্থোপনা": "ব্যবস্থাপনা",
    "ব্যবস্থোপনো": "ব্যবস্থাপনা",
    "ব্যোিস্থোপনো": "ব্যবস্থাপনা",
    "ঝু ঁ ক": "ঝুঁকি",
    "ঝুঁ ক": "ঝুঁকি",
    "মিূল্যায়ন": "মূল্যায়ন",
    "মুল্যায়ন": "মূল্যায়ন",
    "মূল্যোয়ন": "মূল্যায়ন",
    "সতকির্যতা": "সতর্কতা",
    "সতকর্ণতা": "সতর্কতা",
    "জরু²র": "জরুরি",
    "পুনরুদ্ধিার": "পুনরুদ্ধার",
    "পু নরুদ্ধোর": "পুনরুদ্ধার",
    "চক্রর": "চক্রের",
    "প্র াসবনক": "প্রশাসনিক",
    "বিষয়": "বিষয়",
    "বিিরণ": "বিবরণ",
    "মূে": "মূল",
    "উলেশ্য": "উদ্দেশ্য",
    "প্রবিষ্ঠা": "প্রতিষ্ঠা",
    "প্রবিষ্ঠার": "প্রতিষ্ঠার",
    "মনতৃত্ব": "নেতৃত্ব",
    "মদ ": "দেশ ",
    "ব্ািংক": "ব্যাংক",
    "িািংোলদল ": "বাংলাদেশ ",
    "িািংোলদলর": "বাংলাদেশের",
    "মজোয়োর": "জোয়ার",
    "িোটো": "ভাটা",
    "আিহোওয়ো": "আবহাওয়া",
    "জলিোয়ু": "জলবায়ু",
    "মিৌ সলক": "ভৌগোলিক",
    "মিৌ সলক": "ভৌগোলিক",
    "অিস্থোন": "অবস্থান",
    "পসরগিে": "পরিবেশ",
    "দুগ্োযগ": "দুর্যোগ",
    "দুগ্োয": "দুর্যোগ",
    "ব্যোিস্থোপনো": "ব্যবস্থাপনা",
    "সিসিন্ন": "বিভিন্ন",
    "সনয়ো ক": "নিয়ামক",
    "িী োনো": "সীমানা",
})

BANGLA_NOISE_CHARS = str.maketrans({
    "ǂ": "", "ǃ": "", "Ǭ": "", "ǭ": "", "ƒ": "", "ƶ": "", "": "", "￾": "",
})


def normalize_bangla_noise(text: str) -> str:
    text = (text or "").translate(BANGLA_NOISE_CHARS)
    # Custom encoded PDFs often insert '=' between Bangla letters: বাংলা=দে=শ.
    text = re.sub(r"(?<=[ঀ-৿])=(?=[ঀ-৿])", "", text)
    # Remove spaces accidentally inserted before vowel signs/hasanta.
    text = re.sub(r"([ঀ-৿])\s+([া-ৌ়্])", r"\1\2", text)
    text = re.sub(r"([়্])\s+([ঀ-৿])", r"\1\2", text)
    return text


def clean_bangla_spelling(text: str, fuzzy: bool = False) -> str:
    text = normalize_bangla_noise(normalize_text(text or ""))
    if not text:
        return text
    for bad, good in COMMON_BANGLA_FIXES.items():
        text = text.replace(bad, good)
    text = normalize_bangla_noise(text)
    text = correct_bangla_text(text, fuzzy=fuzzy)
    text = re.sub(r"\s+([।,;:?!])", r"\1", text)
    text = re.sub(r"([\u0980-\u09FF])\s+([া-ৗ])", r"\1\2", text)
    return normalize_text(text)


def _bn_digit_to_en(text: str) -> str:
    return (text or "").translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))


def _en_digit_to_bn(text: str) -> str:
    return (text or "").translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))


def _is_response_time_header(header: str) -> bool:
    h = normalize_text(header).lower()
    return (
        "প্রতিক্রিয়া" in h
        or "প্রতিক্রিয়া" in h
        or "response time" in h
        or "average response" in h
        or "গড় প্রতিক্রিয়া" in h
        or "গড় প্রতিক্রিয়া" in h
    )


def repair_lost_decimal_response_time(value: str) -> str:
    """Repair OCR/table extraction loss only for response-time cells.

    Example: ২২ সেকেন্ড -> ২.২ সেকেন্ড, ২০ সেকেন্ড -> ২.০ সেকেন্ড.
    This is intentionally not applied to ordinary numbers.
    """
    original = normalize_text(value)
    if not original or "." in original or "।" in original:
        return original
    en = _bn_digit_to_en(original)
    match = re.search(r"(?<!\d)(\d{2})(?!\d)\s*(সেকেন্ড|second|seconds|sec)?", en, re.IGNORECASE)
    if not match:
        return original
    num = match.group(1)
    repaired = _en_digit_to_bn(f"{num[0]}.{num[1]}")
    suffix = " সেকেন্ড" if "সেকেন্ড" in original or re.search(r"second|sec", en, re.I) else ""
    return repaired + suffix


def group_text_nodes_by_page(nodes: list[RawTextNode]) -> list[tuple[int, str, str]]:
    grouped: list[tuple[int, str, str]] = []
    bucket: list[str] = []
    current_page: int | None = None
    current_heading = ""

    for node in nodes:
        if current_page is None:
            current_page = node.page
            current_heading = node.heading
        same_page = node.page == current_page
        if not same_page and bucket:
            grouped.append((current_page or -1, current_heading, " ".join(bucket)))
            bucket = []
            current_page = node.page
            current_heading = node.heading
        if node.heading and not current_heading:
            current_heading = node.heading
        if node.text:
            bucket.append(node.text)
    if bucket:
        grouped.append((current_page or -1, current_heading, " ".join(bucket)))
    return grouped



def make_text_chunks(
    *,
    document_id: str,
    source_path: str,
    pdf_type: str,
    nodes: list[RawTextNode],
    chunk_size: int,
    overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    ordinal = 0
    for page_no, heading, page_text in group_text_nodes_by_page(nodes):
        page_text = clean_bangla_spelling(page_text)
        if not page_text:
            continue
        for piece in split_text_with_overlap(page_text, chunk_size=chunk_size, overlap=overlap):
            ordinal += 1
            piece = clean_bangla_spelling(piece)
            retrieval_text = piece if not heading else f"শিরোনাম: {clean_bangla_spelling(heading)}\n{piece}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id_for(source_path, page_no, "text", ordinal),
                    document_id=document_id,
                    source_path=source_path,
                    page=page_no,
                    modality="text",
                    heading=heading,
                    pdf_type=pdf_type,
                    text_bn=piece,
                    text_en="",
                    retrieval_text_bn=retrieval_text,
                    retrieval_text_en="",
                    metadata={
                        "document_id": document_id,
                        "source_file": Path(source_path).name,
                    },
                )
            )
    return chunks



def _looks_like_separator(line: str) -> bool:
    return set(line.strip()) <= {"|", "-", ":", " "}


def _parse_compact_pipe_table(markdown_text: str) -> tuple[list[str], list[list[str]]]:
    """Parse Docling-style single-line markdown tables with repeated | | row starts."""
    text = normalize_text(markdown_text)
    if "|" not in text:
        return [], []

    header_match = re.match(
        r"^\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|",
        text,
    )
    if not header_match:
        return [], []
    headers = [clean_bangla_spelling(header_match.group(i)) for i in range(1, 5)]

    row_re = re.compile(
        r"\|\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|",
    )
    rows: list[list[str]] = []
    for match in row_re.finditer(text):
        # Leading "| |" is an empty serial-number cell, then 3 captured fields.
        captured = [clean_bangla_spelling(match.group(i)) for i in range(1, 5)]
        while captured and not captured[-1]:
            captured.pop()
        if not any(captured):
            continue
        # A joined markdown separator row (e.g. ":---|:---|:---") also matches this
        # pattern when the header has no leading index column, since Docling joins
        # each original line with "| |" regardless of whether a row is real data or
        # the separator itself. Reject it here rather than returning it as a fake
        # data row, so the caller falls through to the more general fallback parser.
        if all(set(cell) <= {"-", ":", " "} for cell in captured if cell):
            continue
        row = [""] + captured
        row = row[: len(headers)] + [""] * max(0, len(headers) - len(row))
        rows.append(row[: len(headers)])
    if rows and all(not any(c.strip() for c in row) or all(set(c) <= {"-", ":", " "} for c in row if c) for row in rows):
        return [], []
    return headers, rows


def _expand_compact_markdown_table(markdown_text: str) -> str:
    """Turn Docling one-line markdown tables into multi-line tables for parsing."""
    headers, rows = _parse_compact_pipe_table(markdown_text)
    if headers and rows:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    text = normalize_text(markdown_text)
    if not text or "|" not in text:
        return text
    line_count = len([ln for ln in text.splitlines() if ln.strip()])
    if line_count >= 2:
        return text
    return re.sub(r"\|\s*\|", "|\n|", text)


def parse_markdown_table(markdown_text: str) -> tuple[list[str], list[list[str]]]:
    markdown_text = _expand_compact_markdown_table(markdown_text)
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return [], []
    headers = [normalize_text(cell) for cell in lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    for line in lines[1:]:
        if _looks_like_separator(line):
            continue
        rows.append([normalize_text(cell) for cell in line.strip("|").split("|")])
    width = max([len(headers)] + [len(row) for row in rows], default=0)
    if width == 0:
        return [], []
    headers = headers + [""] * (width - len(headers))
    rows = [row + [""] * (width - len(row)) for row in rows]
    return headers, rows



def _infer_missing_header(headers: list[str], rows: list[list[str]]) -> list[str]:
    fixed = headers[:]
    if len(fixed) >= 4 and not normalize_text(fixed[3]):
        tail_values = [normalize_text(row[3]) for row in rows if len(row) > 3 and normalize_text(row[3])]
        if any("সেকেন্ড" in value for value in tail_values):
            fixed[3] = "গড় প্রতিক্রিয়া সময়"
        elif any("second" in value.lower() for value in tail_values):
            fixed[3] = "average response time"
    return fixed



def _is_month_header(text: str) -> bool:
    text = normalize_text(text).lower()
    return text in {"মাস", "month", "months"}



def _month_index(value: str, month_source: list[str]) -> int | None:
    value = normalize_text(value)
    if month_source is BANGLA_MONTHS:
        value = BANGLA_MONTH_ALIASES.get(value, value)
    lowered = value.lower()
    lowered_source = [m.lower() for m in month_source]
    return lowered_source.index(lowered) if lowered in lowered_source else None


def _infer_missing_months(headers: list[str], rows: list[list[str]]) -> list[list[str]]:
    if not rows or not headers or not _is_month_header(headers[0]):
        return rows
    fixed = [row[:] for row in rows]
    observed = [normalize_text(row[0]) for row in fixed]
    month_source = BANGLA_MONTHS if any(any(month in cell or cell in BANGLA_MONTH_ALIASES for month in BANGLA_MONTHS) for cell in observed) else ENGLISH_MONTHS
    next_idx = 0
    for row in fixed:
        value = normalize_text(row[0])
        if value:
            match_idx = _month_index(value, month_source)
            if match_idx is not None:
                if month_source is BANGLA_MONTHS:
                    row[0] = BANGLA_MONTHS[match_idx]
                next_idx = match_idx + 1
            continue
        if next_idx < len(month_source):
            row[0] = month_source[next_idx]
            next_idx += 1
    return fixed



def repair_table(headers: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    headers = _infer_missing_header(headers, rows)
    rows = _infer_missing_months(headers, rows)
    return headers, rows



def verbalize_table_markdown(markdown_text: str, heading: str = "") -> str:
    headers, rows = parse_markdown_table(markdown_text)
    headers, rows = repair_table(headers, rows)
    if not headers and not rows:
        return normalize_text(markdown_text)

    segments: list[str] = []
    if heading:
        segments.append(f"টেবিলের শিরোনাম: {heading}।")
    visible_headers = [h for h in headers if h]
    if visible_headers:
        segments.append("কলামসমূহ: " + ", ".join(visible_headers) + "।")

    for row_index, row in enumerate(rows[:20], start=1):
        facts: list[str] = []
        for col_name, value in zip(headers, row):
            col_name = clean_bangla_spelling(col_name)
            value = clean_bangla_spelling(value)
            if _is_response_time_header(col_name):
                value = repair_lost_decimal_response_time(value)
            if not value:
                continue
            if col_name:
                facts.append(f"{col_name}: {value}")
            else:
                facts.append(value)
        if facts:
            segments.append(f"সারি {row_index}: " + "; ".join(facts) + "।")
    return " ".join(segments)



def _row_to_bangla_fact(headers: list[str], row: list[str], heading: str = "") -> str:
    pairs: list[str] = []
    month_value = clean_bangla_spelling(row[0]) if row else ""
    for idx, (header, value) in enumerate(zip(headers, row)):
        header = clean_bangla_spelling(header)
        value = clean_bangla_spelling(value)
        if _is_response_time_header(header):
            value = repair_lost_decimal_response_time(value)
        if not value:
            continue
        if idx == 0 and _is_month_header(header):
            continue
        if header:
            pairs.append(f"{header} {value}")
        else:
            pairs.append(value)
    prefix = f"{month_value} মাসে " if month_value else ""
    text = prefix + ", ".join(pairs)
    if heading:
        text = f"{heading}: {text}" if text else heading
    return clean_bangla_spelling(text).rstrip("।") + "।" if text else ""



def make_table_chunks(
    *,
    source_path: str,
    document_id: str,
    page: int,
    pdf_type: str,
    heading: str,
    table_markdown: str,
    image_path: str,
    ordinal_start: int,
) -> list[ChunkRecord]:
    table_markdown = (table_markdown or "").strip()
    if not table_markdown:
        return []

    headers, rows = parse_markdown_table(table_markdown)
    headers, rows = repair_table(headers, rows)

    chunks: list[ChunkRecord] = []
    ordinal = ordinal_start

    heading = clean_bangla_spelling(heading)
    table_summary = verbalize_table_markdown(table_markdown, heading=heading)
    chunks.append(
        ChunkRecord(
            chunk_id=chunk_id_for(source_path, page, "table", ordinal),
            document_id=document_id,
            source_path=source_path,
            page=page,
            modality="table",
            heading=heading,
            pdf_type=pdf_type,
            text_bn=table_summary or table_markdown,
            text_en="",
            retrieval_text_bn=table_summary or table_markdown,
            retrieval_text_en="",
            raw_table_md=table_markdown,
            image_path=image_path,
            metadata={
                "document_id": document_id,
                "source_file": Path(source_path).name,
                "table_variant": "summary",
            },
        )
    )
    ordinal += 1

    for row_idx, row in enumerate(rows, start=1):
        row_fact = _row_to_bangla_fact(headers, row, heading=heading)
        if not row_fact:
            continue
        raw_fields = []
        for h, v in zip(headers, row):
            h_clean = clean_bangla_spelling(h)
            v_clean = clean_bangla_spelling(v)
            if _is_response_time_header(h_clean):
                v_clean = repair_lost_decimal_response_time(v_clean)
            if v_clean:
                raw_fields.append(f"{h_clean}={v_clean}")
        raw_row = " | ".join(raw_fields)
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id_for(source_path, page, "table-row", ordinal),
                document_id=document_id,
                source_path=source_path,
                page=page,
                modality="table",
                heading=heading,
                pdf_type=pdf_type,
                text_bn=row_fact,
                text_en="",
                retrieval_text_bn=row_fact,
                retrieval_text_en="",
                raw_table_md=raw_row,
                image_path=image_path,
                metadata={
                    "document_id": document_id,
                    "source_file": Path(source_path).name,
                    "table_variant": "row",
                    "row_index": row_idx,
                },
            )
        )
        ordinal += 1

    return chunks



ARROW_SPLIT_RE = re.compile(r"\s*(?:→|->|=>|⇒|➜|➔|>|›|→)\s*")
ENUM_SPLIT_RE = re.compile(r"(?:^|\s)(?:[০-৯]+|[0-9]+)[.)।:-]\s*")


def _clean_sequence_item(value: str) -> str:
    value = clean_bangla_spelling(value)
    value = re.sub(r"^[•*\-–—]+\s*", "", value)
    value = re.sub(r"^(ধাপ|step)\s*[০-৯0-9]*\s*[:.-]?\s*", "", value, flags=re.I)
    return value.strip(" ।,;:-")


def extract_generic_sequences(text: str) -> list[list[str]]:
    """Extract ordered list/flow/chart-like sequences from any Bangla PDF text.

    Works for flow charts, timeline arrows, process steps, and numbered lists.
    It is intentionally generic and does not contain PDF-specific topics.
    """
    text = clean_bangla_spelling(text)
    sequences: list[list[str]] = []
    for line in re.split(r"[\n।]+", text):
        line = normalize_text(line)
        if not line:
            continue
        if any(sym in line for sym in ["→", "->", "=>", "⇒", "➜", "➔"]):
            items = [_clean_sequence_item(x) for x in ARROW_SPLIT_RE.split(line)]
            items = [x for x in items if len(x) >= 2]
            if len(items) >= 2:
                sequences.append(items[:12])
        elif len(ENUM_SPLIT_RE.findall(line)) >= 2:
            parts = [_clean_sequence_item(x) for x in ENUM_SPLIT_RE.split(line)]
            parts = [x for x in parts if len(x) >= 2]
            if len(parts) >= 2:
                sequences.append(parts[:12])
    return sequences


def make_structure_chunks(
    *,
    document_id: str,
    source_path: str,
    page: int,
    pdf_type: str,
    heading: str,
    text: str,
    ordinal_start: int,
) -> list[ChunkRecord]:
    """Create extra chunks for ordered structures: flow charts, steps, timelines.

    This is generic for many Bangla PDFs: disaster cycles, science processes,
    historical timelines, geography sequences, guideline workflows, etc.
    """
    sequences = extract_generic_sequences(text)
    chunks: list[ChunkRecord] = []
    ordinal = ordinal_start
    heading = clean_bangla_spelling(heading)
    for seq in sequences:
        if len(seq) < 2:
            continue
        numbered = ", ".join(f"{_en_digit_to_bn(str(i))}. {item}" for i, item in enumerate(seq, start=1))
        first = seq[0]
        last = seq[-1]
        fact = f"{heading + ': ' if heading else ''}ক্রম/ধাপসমূহ হলো: {numbered}। প্রথম ধাপ হলো {first}। শেষ ধাপ হলো {last}।"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id_for(source_path, page, "structure", ordinal),
                document_id=document_id,
                source_path=source_path,
                page=page,
                modality="structure",
                heading=heading,
                pdf_type=pdf_type,
                text_bn=fact,
                text_en="",
                retrieval_text_bn=fact,
                retrieval_text_en="",
                metadata={
                    "document_id": document_id,
                    "source_file": Path(source_path).name,
                    "structure_variant": "ordered_sequence",
                    "first_item": first,
                    "last_item": last,
                    "item_count": len(seq),
                },
            )
        )
        ordinal += 1
    return chunks


def make_image_chunk(
    *,
    source_path: str,
    document_id: str,
    page: int,
    pdf_type: str,
    heading: str,
    image_path: str,
    caption: str,
    ordinal: int,
    ocr_text: str = "",
) -> ChunkRecord:
    caption = clean_bangla_spelling(caption)
    heading = clean_bangla_spelling(heading)
    generic_caption = normalize_text(caption).lower() in {
        "image extracted from the document.",
        "নথিতে থেকে ছবি।",
    } or "image extracted from the document" in normalize_text(caption).lower()
    parts: list[str] = []
    if heading:
        parts.append(f"শিরোনাম: {heading}")
    if ocr_text:
        parts.append(ocr_text)
    if caption and not generic_caption:
        parts.append(caption)
    elif not ocr_text:
        parts.append(caption or "Image extracted from the document.")
    retrieval_text = "\n".join(parts)
    return ChunkRecord(
        chunk_id=chunk_id_for(source_path, page, "image", ordinal),
        document_id=document_id,
        source_path=source_path,
        page=page,
        modality="image",
        heading=heading,
        pdf_type=pdf_type,
        text_bn=retrieval_text,
        text_en="",
        retrieval_text_bn=retrieval_text,
        retrieval_text_en="",
        image_path=image_path,
        metadata={
            "document_id": document_id,
            "source_file": Path(source_path).name,
            "ocr_text": ocr_text,
        },
    )
