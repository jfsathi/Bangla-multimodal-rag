from __future__ import annotations

import re
import requests
from requests.exceptions import ConnectionError, ReadTimeout, RequestException

from .image_ocr import chart_fact_hints
from .schemas import RetrievedChunk
from .text_processing import clean_bangla_spelling, parse_markdown_table, verbalize_table_markdown
from .utils import format_source_tag, normalize_text


BANGLA_SYSTEM_PROMPT = """
তুমি একটি বাংলা RAG সহকারী।

নিয়ম:
১. শুধুমাত্র দেওয়া Retrieved context ব্যবহার করে উত্তর দাও। বাইরের জ্ঞান ব্যবহার করবে না।
২. Context-এ OCR/font/table noise থাকতে পারে: শব্দ জোড়া লাগা, বানান ভুল, বাংলা/ইংরেজি মিশ্রণ, টেবিলের সারি এক লাইনে থাকা—এসব দেখে অর্থ বুঝে উত্তর দেবে।
৩. প্রশ্নে সামান্য বানান ভুল থাকলে কাছাকাছি শব্দ মিলাবে; যেমন "বিশ্ববিদ্দালয়" = "বিশ্ববিদ্যালয়", "প্রতিষ্ঠা সাল" = "শুরু/প্রতিষ্ঠা সাল"।
৪. টেবিল/সারণি থাকলে matching row এবং matching column দেখে নির্দিষ্ট মান বের করবে। Raw table কপি করবে না।
৫. চিত্র/চার্ট/গ্রাফ থাকলে OCR-এ দেওয়া গ্রহ/বস্তুর নাম ও দূরত্ব/সংখ্যা matching row হিসেবে ব্যবহার করবে।
৬. যদি প্রশ্নে “সাল/তারিখ/কত/কয়টি/কোন/কে/কোথায়/কাকে বলে/কী” থাকে, তাহলে সরাসরি ছোট উত্তর দেবে।
৭. Rank 1 chunk-এ উত্তর না থাকলেও পরের chunk-এ উত্তর থাকলে সেটি ব্যবহার করবে।
৮. "Primary matched table fact" বা "প্রতিষ্ঠা সাল" দেখলে সেই মানই উত্তরে ব্যবহার করবে; আইন অনুমোদনের তারিখকে প্রতিষ্ঠা সাল বানাবে না।
৯. "Chart/image matched fact" বা চিত্র OCR-এ গ্রহের পাশে দূরত্ব থাকলে সেটাই উত্তরে ব্যবহার করবে।
১০. উত্তর শুদ্ধ, প্রমিত বাংলায় লিখবে এবং শেষে উৎস পৃষ্ঠা দেবে: [file.pdf p.1]
১১. Context-এ কোনোভাবেই উত্তর না থাকলেই শুধু বলবে: "প্রদত্ত নথিতে এর উত্তর পাওয়া যায়নি।"
১২. উত্তর সর্বোচ্চ ১–২ বাক্য হবে, যদি ব্যবহারকারী বিস্তারিত না চায়।
"""

ENGLISH_SYSTEM_PROMPT = """
You are a grounded RAG assistant for Bangla document question answering.
Use only the Retrieved context. OCR/table noise may exist: joined words, misspellings, mixed Bangla/English, or table rows in one line. For table questions, find the matching row and matching column, then answer naturally with the source tag. Say not found only when the answer is genuinely absent from all retrieved sources.
"""

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _normalize_for_matching(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("বিশ্ববিদ্দালয়", "বিশ্ববিদ্যালয়").replace("বিশ্ববিদ্যালয়", "বিশ্ববিদ্যালয়")
    text = text.replace("ঢাকাবিশ্ববিদ্যালয়", "ঢাকা বিশ্ববিদ্যালয়")
    text = text.replace("বিশ্ববিদ্যালয়মঞ্জুরী", "বিশ্ববিদ্যালয় মঞ্জুরী")
    text = text.replace("শুরুপ্রতিষ্ঠাসাল", "শুরু প্রতিষ্ঠা সাল")
    text = text.replace("প্রতিষ্ঠাসাল", "প্রতিষ্ঠা সাল")
    text = re.sub(r"(?<=[\u0980-\u09FF])বিশ্ববিদ্যালয়", " বিশ্ববিদ্যালয়", text)
    text = re.sub(r"বিশ্ববিদ্যালয়(?=[\u0980-\u09FF])", "বিশ্ববিদ্যালয় ", text)
    text = re.sub(r"(?<=[\u0980-\u09FF])সাল", " সাল", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _question_terms(question: str) -> list[str]:
    q = _normalize_for_matching(question)
    q = re.sub(r"[?।,;:()\[\]{}|/\\\-]+", " ", q)
    stop = {"এর", "কি", "কী", "কত", "কোন", "কে", "কাকে", "বলে", "দাও", "লিখ", "হয়", "হয়", "টি", "গুলো"}
    terms = []
    for tok in q.split():
        if len(tok) >= 2 and tok not in stop:
            terms.append(tok)
    return terms[:12]


def _make_table_more_readable(markdown: str, heading: str = "") -> str:
    """Format retrieved table evidence for the LLM (rows/columns, not a final answer)."""
    md = normalize_text(markdown)
    if not md:
        return ""
    verbalized = verbalize_table_markdown(md, heading=heading)
    if verbalized and verbalized != md:
        return _normalize_for_matching(verbalized)
    md = _normalize_for_matching(md)
    md = re.sub(r"\|\s*\|", "|\n|", md)
    md = re.sub(r"\n\s*\|\s*[-: ]{3,}.*", "", md)
    md = re.sub(r"\s*\|\s*", " | ", md)
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not re.fullmatch(r"[|\-: ]+", ln)]
    return "\n".join(lines[:80])


def _table_evidence_hints(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Build short row-level hints from parsed tables to ground the LLM."""
    terms = _question_terms(question)
    if not terms:
        return ""
    q_norm = _normalize_for_matching(question)
    wants_year = any(k in q_norm for k in ("প্রতিষ্ঠা", "স্থাপন", "সাল", "তারিখ", "কত"))
    hints: list[str] = []
    for item in retrieved_chunks:
        chunk = item.chunk
        table_md = chunk.raw_table_md or (chunk.text_bn if chunk.modality == "table" else "")
        if not table_md or "|" not in table_md:
            continue
        headers, rows = parse_markdown_table(table_md)
        if not headers or not rows:
            continue
        name_idx = next(
            (i for i, h in enumerate(headers) if "নাম" in _normalize_for_matching(h)),
            1 if len(headers) > 1 else 0,
        )
        year_idx = next(
            (
                i
                for i, h in enumerate(headers)
                if any(k in _normalize_for_matching(h) for k in ("প্রতিষ্ঠা", "শুরু"))
            ),
            len(headers) - 1,
        )
        for row in rows:
            row_text = _normalize_for_matching(" ".join(row))
            if not row_text:
                continue
            name_cell = row[name_idx] if name_idx < len(row) else ""
            hits = sum(1 for term in terms if term in row_text or (name_cell and term in _normalize_for_matching(name_cell)))
            if hits == 0:
                continue
            pairs = []
            if name_cell:
                pairs.append(f"বিশ্ববিদ্যালয়: {clean_bangla_spelling(name_cell)}")
            if wants_year and year_idx < len(row) and row[year_idx]:
                pairs.append(f"প্রতিষ্ঠা সাল: {clean_bangla_spelling(row[year_idx])}")
            for header, value in zip(headers, row):
                header = clean_bangla_spelling(header)
                value = clean_bangla_spelling(value)
                if not value or header in {"", "ক্রনং"}:
                    continue
                label = _normalize_for_matching(header)
                if name_cell and value == name_cell:
                    continue
                if year_idx < len(row) and value == row[year_idx] and "প্রতিষ্ঠা সাল" in " ".join(pairs):
                    continue
                pairs.append(f"{header}: {value}")
            if pairs:
                hints.append("; ".join(pairs))
        if len(hints) >= 3:
            break
    return "\n".join(hints[:3])


def _primary_table_fact(question: str, table_hints: str) -> str:
    """Pick the best matching table hint line for the question."""
    if not table_hints:
        return ""
    terms = _question_terms(question)
    best_line = ""
    best_score = 0
    for line in table_hints.splitlines():
        norm = _normalize_for_matching(line)
        score = sum(2 for term in terms if term in norm)
        if "প্রতিষ্ঠা সাল:" in line:
            score += 3
        if score > best_score:
            best_score = score
            best_line = line
    return best_line if best_score >= 3 else ""


def _chart_evidence_hints(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    hints: list[str] = []
    for item in retrieved_chunks:
        chunk = item.chunk
        if chunk.modality != "image":
            continue
        ocr_text = normalize_text(chunk.metadata.get("ocr_text", "")) or chunk.text_bn
        hint = chart_fact_hints(question, ocr_text)
        if hint:
            hints.append(hint)
    return "\n".join(hints[:3])


def _primary_chart_fact(question: str, chart_hints: str) -> str:
    if not chart_hints:
        return ""
    terms = _question_terms(question)
    best_line = ""
    best_score = 0
    for line in chart_hints.splitlines():
        for segment in re.split(r"[;\n]", line):
            segment = normalize_text(segment)
            if not segment or ":" not in segment:
                continue
            score = sum(2 for term in terms if term in _normalize_for_matching(segment))
            if score > best_score:
                best_score = score
                best_line = segment
    return best_line if best_score >= 2 else ""


def _make_focused_excerpt(
    question: str,
    text: str,
    raw_table_md: str = "",
    heading: str = "",
    limit: int = 2200,
) -> str:
    terms = _question_terms(question)
    parts: list[str] = []

    table_text = _make_table_more_readable(raw_table_md, heading=heading) if raw_table_md else ""
    body = _normalize_for_matching(text)

    # If table exists, put readable table first; it is often where exact values live.
    combined = "\n".join(x for x in [table_text, body] if x)
    if not combined:
        return ""

    # Focus lines containing question terms, but also include nearby table header lines.
    lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
    selected: list[str] = []
    for i, line in enumerate(lines):
        normalized_line = _normalize_for_matching(line)
        hit_count = sum(1 for t in terms if t in normalized_line)
        if hit_count > 0:
            for j in range(max(0, i - 1), min(len(lines), i + 2)):
                if lines[j] not in selected:
                    selected.append(lines[j])

    if selected:
        focused = "\n".join(selected)
        # Add full readable table/body only if focused rows are too short.
        if len(focused) < 500:
            focused = focused + "\n\nFull retrieved content:\n" + combined
    else:
        focused = combined

    if len(focused) > limit:
        return focused[:limit].rstrip() + " ... [truncated]"
    return focused


class OllamaGenerator:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        timeout_seconds: int = 420,
        max_context_chars: int = 12000,
        max_chunk_chars: int = 2400,
        num_predict: int = 180,
        num_ctx: int = 8192,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_context_chars = max_context_chars
        self.max_chunk_chars = max_chunk_chars
        self.num_predict = num_predict
        self.num_ctx = num_ctx

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        text = text or ""
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " ... [truncated]"

    def _build_prompt(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        answer_language: str,
        *,
        strict_retry: bool = False,
    ) -> str:
        terms = ", ".join(_question_terms(question))
        context_blocks: list[str] = []
        for item in retrieved_chunks:
            chunk = item.chunk
            source_tag = format_source_tag(chunk.source_path, chunk.page)
            if answer_language == "bn":
                base_context = clean_bangla_spelling(chunk.text_bn or chunk.text_en)
            else:
                base_context = chunk.text_en or chunk.text_bn
            focused_context = _make_focused_excerpt(
                question=question,
                text=base_context,
                raw_table_md=chunk.raw_table_md or (chunk.text_bn if chunk.modality == "table" else ""),
                heading=chunk.heading or "",
                limit=self.max_chunk_chars,
            )
            ocr_evidence = ""
            if chunk.modality == "image":
                ocr_text = chunk.metadata.get("ocr_text", "") or chunk.text_bn or chunk.text_en
                if ocr_text:
                    ocr_evidence = f"OCR evidence:\n{normalize_text(ocr_text)}\n\n"

            context_blocks.append(
                f"SOURCE {item.rank} {source_tag}\n"
                f"Rank score: {item.score:.3f}\n"
                f"Modality: {chunk.modality}\n"
                f"Heading: {_normalize_for_matching(chunk.heading)}\n"
                f"Readable evidence:\n{ocr_evidence}{focused_context}"
            )
        joined_context = self._truncate_text("\n\n".join(context_blocks), self.max_context_chars)
        table_hints = _table_evidence_hints(question, retrieved_chunks)
        primary_fact = _primary_table_fact(question, table_hints)
        if table_hints:
            joined_context = (
                "Parsed table row hints (from retrieved evidence only):\n"
                f"{table_hints}\n\n"
                + joined_context
            )
            if primary_fact:
                joined_context = (
                    "Primary matched table fact (you must use this value in the answer):\n"
                    f"{primary_fact}\n\n"
                    + joined_context
                )
            joined_context = self._truncate_text(joined_context, self.max_context_chars)
        chart_hints = _chart_evidence_hints(question, retrieved_chunks)
        primary_chart = _primary_chart_fact(question, chart_hints)
        if chart_hints:
            joined_context = (
                "Parsed chart/image facts (from retrieved figure OCR only):\n"
                f"{chart_hints}\n\n"
                + joined_context
            )
            if primary_chart:
                joined_context = (
                    "Primary matched chart/image fact (you must use this value in the answer):\n"
                    f"{primary_chart}\n\n"
                    + joined_context
                )
            joined_context = self._truncate_text(joined_context, self.max_context_chars)
        if answer_language == "bn":
            system_prompt = BANGLA_SYSTEM_PROMPT
            answer_prefix = "উত্তর:"
        else:
            system_prompt = ENGLISH_SYSTEM_PROMPT
            answer_prefix = "Answer:"

        retry_instruction = ""
        if strict_retry:
            retry_instruction = (
                "\nবিশেষ নির্দেশ: আগের উত্তরে তুমি 'পাওয়া যায়নি' বলেছিলে। আবার context ভালোভাবে দেখো। "
                "বিশেষ করে টেবিলের matching row ও column মিলাও। যদি কোনো সারিতে প্রশ্নের বিষয় থাকে এবং পাশে সাল/মান থাকে, সেটি দিয়ে উত্তর দাও।\n"
            )

        return (
            f"{system_prompt}\n"
            f"{retry_instruction}\n\n"
            f"প্রশ্নের গুরুত্বপূর্ণ শব্দ / Question terms: {terms}\n\n"
            "Retrieved context:\n"
            f"{joined_context}\n\n"
            f"প্রশ্ন / Question:\n{question}\n\n"
            f"{answer_prefix}"
        )

    @staticmethod
    def _looks_like_not_found(answer: str) -> bool:
        a = normalize_text(answer)
        markers = [
            "উত্তর পাওয়া যায়নি",
            "উত্তর পাওয়া যায়নি",
            "পাওয়া যায়নি",
            "পাওয়া যায়নি",
            "not found",
            "not available",
        ]
        return any(m in a for m in markers)

    @staticmethod
    def _strip_model_thinking(text: str) -> str:
        text = normalize_text(text)
        if not text:
            return ""
        think_block = re.compile(
            r"<" + r"think" + r">.*?</" + r"think" + r">",
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = think_block.sub("", text)
        answer_marker = "<" + r"/think" + r">"
        if answer_marker in text:
            text = text.split(answer_marker, 1)[-1]
        return normalize_text(text)

    def _call_ollama(self, prompt: str) -> str:
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
            "keep_alive": "5m",
        }
        # Qwen3.x may spend tokens on internal reasoning unless disabled.
        if "qwen3" in self.model_name.lower():
            payload["think"] = False
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(10, self.timeout_seconds),
            )
            response.raise_for_status()
        except ReadTimeout as exc:
            raise RuntimeError(
                f"Ollama model '{self.model_name}' timed out after {self.timeout_seconds} seconds. "
                "Try Top-k 3, use qwen2.5:3b-instruct/qwen3.5:latest, or reduce context."
            ) from exc
        except ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Start the local Ollama server and ensure the base URL is correct. "
                "Example: `ollama serve --listen http://localhost:11434` or update `ollama_base_url`."
            ) from exc
        except RequestException as exc:
            raise RuntimeError(f"Ollama generation failed for model '{self.model_name}': {exc}") from exc
        body = response.json()
        return self._strip_model_thinking(body.get("response", ""))

    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk], answer_language: str) -> str:
        prompt = self._build_prompt(question, retrieved_chunks=retrieved_chunks, answer_language=answer_language)
        answer = self._call_ollama(prompt)

        # If relevant chunks were retrieved but the model falsely says not found,
        # retry once with a stricter prompt. This is still LLM generation from retrieved chunks.
        if self._looks_like_not_found(answer) and retrieved_chunks:
            retry_prompt = self._build_prompt(
                question,
                retrieved_chunks=retrieved_chunks,
                answer_language=answer_language,
                strict_retry=True,
            )
            retry_answer = self._call_ollama(retry_prompt)
            if retry_answer and not (self._looks_like_not_found(retry_answer) and len(retry_answer) <= len(answer) + 20):
                answer = retry_answer

        return clean_bangla_spelling(answer) if answer_language == "bn" else answer
