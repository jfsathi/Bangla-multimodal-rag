#!/usr/bin/env python
"""Build data/eval_ground_truth_full.csv: the full 10-document ground-truth set
for the Q1-journal result-analysis evaluation.

Migrates the original 28 Bangla questions (Science9/Geography/piechart) to the
new language-neutral schema and adds newly drafted questions for the other 7
documents, including the one full-English document (DUETBooklet), to prove the
system generalizes beyond Bangla.

gold_chunk_id is left blank here; scripts/backfill_gold_chunk_ids.py fills it
in after the projects are built, by matching each question's answer text
against the real corpus chunks.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD_GT = ROOT / "data" / "eval_ground_truth.csv"
NEW_GT = ROOT / "data" / "eval_ground_truth_full.csv"

FIELDNAMES = [
    "question_id", "pdf_id", "pdf_name", "pdf_type", "page_no", "gold_chunk_id",
    "modality", "question_type", "answer_language", "question_text",
    "gold_answer_text", "expected_keywords_text", "difficulty",
]


def migrate_old_rows() -> list[dict]:
    rows = []
    with OLD_GT.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "question_id": r["question_id"],
                "pdf_id": r["pdf_id"],
                "pdf_name": r["pdf_name"],
                "pdf_type": r["pdf_type"],
                "page_no": r["page_no"],
                "gold_chunk_id": r["gold_chunk_id"],
                "modality": r["modality"],
                "question_type": r["question_type"],
                "answer_language": "bn",
                "question_text": r["question_bn"],
                "gold_answer_text": r["gold_answer_bn"],
                "expected_keywords_text": r["expected_keywords_bn"],
                "difficulty": r["difficulty"],
            })
    return rows


def row(qid, pdf_id, pdf_name, pdf_type, page, modality, qtype, lang, q, a, kw, diff):
    return {
        "question_id": qid, "pdf_id": pdf_id, "pdf_name": pdf_name, "pdf_type": pdf_type,
        "page_no": page, "gold_chunk_id": "", "modality": modality, "question_type": qtype,
        "answer_language": lang, "question_text": q, "gold_answer_text": a,
        "expected_keywords_text": kw, "difficulty": diff,
    }


def agriculture_rows() -> list[dict]:
    pid, pname, ptype = "Agriculture", "Bangla_Agriculture_RAG_Test_Document.pdf", "native_text"
    r = lambda *a: row("", pid, pname, ptype, *a)
    return [
        r(1, "text", "short_fact", "bn", "ধানের মাঝারি মেয়াদি জাতের গড় জীবনকাল কত দিন?", "১২০ দিন", "১২০ দিন", "easy"),
        r(1, "text", "short_fact", "bn", "ধান চাষের জন্য মাটির সাধারণ উপযোগী pH সীমা কত?", "৫.৫-৬.৫", "৫.৫;৬.৫;pH", "easy"),
        r(1, "text", "short_fact", "bn", "চারা লাগানোর পর নিয়ন্ত্রিত পানি ধরে রাখার সাধারণ গভীরতা কত?", "৪-৫ সেমি", "৪-৫ সেমি", "easy"),
        r(1, "structure", "count_list", "bn", "পুষ্টি ব্যবস্থাপনার মূল ধাপ কয়টি এবং কী কী?", "৩ ধাপ: মাটি পরীক্ষা, সুষম সার, পর্যবেক্ষণ", "মাটি পরীক্ষা;সুষম সার;পর্যবেক্ষণ", "medium"),
        r(1, "structure", "flow_order", "bn", "ধান চাষের ধাপগুলোর ক্রম কী (বীজ নির্বাচন থেকে ফসল সংগ্রহ পর্যন্ত)?", "বীজ নির্বাচন → বীজতলা → রোপণ → ব্যবস্থাপনা → ফসল সংগ্রহ", "বীজ নির্বাচন;বীজতলা;রোপণ;ব্যবস্থাপনা;ফসল সংগ্রহ", "medium"),
        r(1, "structure", "short_fact", "bn", "চারা রোপণের সাধারণ দূরত্ব কত?", "২০×১৫ সেমি", "২০×১৫ সেমি", "medium"),
        r(2, "table", "short_fact", "bn", "নাইট্রোজেন (N) ঘাটতির প্রধান লক্ষণ কী?", "পাতা হলুদ, গাছ খাটো, কুশি কম", "পাতা হলুদ;গাছ খাটো;কুশি কম", "medium"),
        r(2, "table", "short_fact", "bn", "পটাশিয়াম (K) এর প্রধান ভূমিকা কী?", "কাণ্ড শক্ত করা, রোগ সহনশীলতা বাড়ানো এবং দানা পূরণে সহায়তা করা", "কাণ্ড শক্ত;রোগ সহনশীলতা;দানা পূরণ", "medium"),
        r(2, "table", "short_fact", "bn", "জিংক (Zn) ঘাটতির লক্ষণ কী?", "পাতায় বাদামি দাগ ও চারা খর্বাকৃতি", "বাদামি দাগ;খর্বাকৃতি", "medium"),
        r(2, "image", "chart_value", "bn", "বার চার্টে কুশি গঠন পর্যায়ে পুষ্টি চাহিদা সূচক কত শতাংশ দেখানো হয়েছে?", "৮৮%", "৮৮%;কুশি গঠন", "medium"),
        r(2, "image", "chart_value", "bn", "বার চার্টে চারা পর্যায়ে পুষ্টি চাহিদা সূচক কত?", "৪৫%", "৪৫%;চারা পর্যায়", "medium"),
        r(3, "image", "chart_value", "bn", "পাই চার্ট অনুযায়ী ফলন ক্ষতির সবচেয়ে বড় কারণ কী এবং তার অংশ কত?", "পুষ্টি ঘাটতি, ৪৫%", "পুষ্টি ঘাটতি;৪৫%", "medium"),
        r(3, "image", "chart_value", "bn", "পাই চার্টে পানি চাপের কারণে ফলন ক্ষতির অংশ কত শতাংশ?", "২৫%", "২৫%;পানি চাপ", "medium"),
        r(3, "text", "short_fact", "bn", "কোন পর্যায়ে পানি ঘাটতি সবচেয়ে বেশি ক্ষতিকর?", "শীষ বের হওয়া ও ফুল আসার সময়", "শীষ বের হওয়া;ফুল আসার সময়", "hard"),
        r(4, "table", "short_fact", "bn", "কালিগঞ্জ গ্রামের কেস স্টাডিতে প্রথম বছরে ফলন কত ছিল?", "৪.১ টন/হেক্টর", "৪.১ টন/হেক্টর", "easy"),
        r(4, "table", "short_fact", "bn", "কালিগঞ্জ গ্রামের কেস স্টাডিতে দ্বিতীয় বছরে ফলন কত হয়েছিল?", "৪.৮ টন/হেক্টর", "৪.৮ টন/হেক্টর", "easy"),
        r(4, "table", "short_fact", "bn", "কালিগঞ্জ গ্রামের জমির মাটির pH কত ছিল?", "৬.২", "৬.২;pH", "medium"),
        r(4, "table", "short_fact", "bn", "দ্বিতীয় বছরে প্রথম বছরের তুলনায় ফলন কতটা বৃদ্ধি পেয়েছিল?", "০.৭ টন/হেক্টর", "০.৭ টন/হেক্টর", "hard"),
        r(4, "text", "definition", "bn", "মালচিং কী?", "মাটির উপর খড়, পাতা বা জৈব আবরণ দেওয়া, যা মাটির আর্দ্রতা ধরে রাখে", "মালচিং;খড়;পাতা;আর্দ্রতা", "medium"),
        r(4, "text", "definition", "bn", "সমন্বিত বালাই ব্যবস্থাপনা কী?", "রাসায়নিক, জৈব ও সাংস্কৃতিক পদ্ধতি একসঙ্গে ব্যবহার করে ক্ষতি কমানো", "সমন্বিত বালাই ব্যবস্থাপনা;রাসায়নিক;জৈব;সাংস্কৃতিক", "medium"),
    ]


def bd_history_rows() -> list[dict]:
    pid, pname, ptype = "BDHistory", "bangladesh_history.pdf", "native_text"
    r = lambda *a: row("", pid, pname, ptype, *a)
    return [
        r(1, "text", "short_fact", "bn", "বাংলাদেশের জাতীয় ফুল কী?", "শাপলা", "শাপলা", "easy"),
        r(1, "text", "short_fact", "bn", "বাংলাদেশের রাজধানীর নাম কী?", "ঢাকা", "ঢাকা", "easy"),
        r(1, "text", "short_fact", "bn", "বাংলাদেশ কত সালে স্বাধীনতা অর্জন করে?", "১৯৭১", "১৯৭১", "easy"),
        r(2, "table", "short_fact", "bn", "বাংলাদেশের রাষ্ট্রের পূর্ণ নাম কী?", "গণপ্রজাতন্ত্রী বাংলাদেশ", "গণপ্রজাতন্ত্রী বাংলাদেশ", "easy"),
        r(2, "table", "short_fact", "bn", "বাংলাদেশের জাতীয় পাখি ও জাতীয় মাছ কী কী?", "জাতীয় পাখি দোয়েল, জাতীয় মাছ ইলিশ", "দোয়েল;ইলিশ", "medium"),
        r(2, "table", "short_fact", "bn", "বাংলাদেশের জাতীয় ফল কী?", "কাঁঠাল", "কাঁঠাল", "easy"),
        r(2, "table", "short_fact", "bn", "বাংলাদেশের ভৌগোলিক অবস্থান কেমন?", "দক্ষিণ এশিয়া; ভারত, মিয়ানমার ও বঙ্গোপসাগর দ্বারা পরিবেষ্টিত", "দক্ষিণ এশিয়া;ভারত;মিয়ানমার;বঙ্গোপসাগর", "medium"),
        r(2, "structure", "flow_order", "bn", "বাংলাদেশের ভৌগোলিক গঠনের প্রবাহচিত্র অনুযায়ী ক্রম কী?", "হিমালয় ও উজানের নদী → পলি বহন → সমভূমি ও ডেল্টা → কৃষি ও বসতি", "হিমালয়;পলি বহন;সমভূমি;ডেল্টা;কৃষি", "medium"),
        r(3, "structure", "short_fact", "bn", "১৯৫২ সালের ভাষা আন্দোলনের মূল শিক্ষা কী?", "মাতৃভাষার মর্যাদা রক্ষা", "মাতৃভাষার মর্যাদা", "medium"),
        r(3, "structure", "short_fact", "bn", "মুক্তিযুদ্ধের মূল শিক্ষা কী হিসেবে উল্লেখ করা হয়েছে?", "স্বাধীনতা ও আত্মনিয়ন্ত্রণের অধিকার", "স্বাধীনতা;আত্মনিয়ন্ত্রণ", "medium"),
        r(3, "table", "date_lookup", "bn", "ছয় দফা আন্দোলন কোন সালে হয়?", "১৯৬৬", "১৯৬৬;ছয় দফা", "medium"),
        r(3, "table", "date_lookup", "bn", "গণঅভ্যুত্থান কোন সালে হয়েছিল?", "১৯৬৯", "১৯৬৯;গণঅভ্যুত্থান", "medium"),
        r(4, "text", "short_fact", "bn", "বাংলাদেশ কয়টি প্রশাসনিক বিভাগে বিভক্ত?", "আটটি বিভাগ", "আটটি;৮", "easy"),
        r(4, "table", "short_fact", "bn", "সিলেট বিভাগের উল্লেখযোগ্য দিক কী কী?", "চা, পর্যটন, হাওর", "চা;পর্যটন;হাওর", "medium"),
        r(4, "table", "short_fact", "bn", "চট্টগ্রাম বিভাগের পরিচিতি/বৈশিষ্ট্য কী?", "বন্দর, পাহাড় ও সমুদ্রের অঞ্চল", "বন্দর;পাহাড়;সমুদ্র", "medium"),
        r(5, "image", "chart_value", "bn", "অর্থনৈতিক ক্ষেত্রের পাই চার্টে সেবা খাতের অংশ কত শতাংশ?", "৩৮%", "৩৮%;সেবা খাত", "medium"),
        r(5, "image", "chart_value", "bn", "অর্থনৈতিক ক্ষেত্রের পাই চার্টে সবচেয়ে ছোট অংশ কোনটি এবং কত শতাংশ?", "প্রবাসী আয়/অন্যান্য, ১০%", "প্রবাসী আয়;১০%", "hard"),
        r(5, "table", "short_fact", "bn", "কৃষি খাতের প্রধান চ্যালেঞ্জ কী কী?", "বন্যা, জলবায়ু পরিবর্তন, জমির চাপ", "বন্যা;জলবায়ু পরিবর্তন;জমির চাপ", "medium"),
    ]


def polashir_juddho_rows() -> list[dict]:
    pid, pname, ptype = "PolashirJuddho", "polashir_juddho.pdf", "native_text"
    r = lambda *a: row("", pid, pname, ptype, *a)
    return [
        r(1, "text", "date_lookup", "bn", "পলাশীর যুদ্ধ কবে সংঘটিত হয়?", "২৩ জুন ১৭৫৭", "২৩ জুন;১৭৫৭", "easy"),
        r(1, "text", "short_fact", "bn", "পলাশীর যুদ্ধ কোথায় হয়েছিল?", "পলাশী, ভাগীরথী নদীর তীরে, বর্তমান নদিয়া জেলা, পশ্চিমবঙ্গ", "পলাশী;ভাগীরথী নদী;নদিয়া", "easy"),
        r(1, "text", "short_fact", "bn", "পলাশীর যুদ্ধের মূল প্রতিপক্ষ কারা ছিল?", "নবাব সিরাজউদ্দৌলা ও ইংরেজ ইস্ট ইন্ডিয়া কোম্পানি", "সিরাজউদ্দৌলা;ইস্ট ইন্ডিয়া কোম্পানি", "easy"),
        r(1, "text", "short_fact", "bn", "পলাশীর যুদ্ধের মূল সংঘর্ষের সময়কাল কত ছিল?", "প্রায় ৮ ঘণ্টা", "৮ ঘণ্টা", "medium"),
        r(1, "table", "short_fact", "bn", "নবাবের সেনাপতি হিসেবে যুদ্ধক্ষেত্রে নিষ্ক্রিয় থেকে ইংরেজদের জয়ে সহায়তা করেন কে?", "মীর জাফর", "মীর জাফর", "easy"),
        r(1, "table", "short_fact", "bn", "ইস্ট ইন্ডিয়া কোম্পানির সেনানায়ক কে ছিলেন?", "রবার্ট ক্লাইভ", "রবার্ট ক্লাইভ", "easy"),
        r(1, "table", "short_fact", "bn", "রাজনৈতিক পরিবর্তনে অর্থনৈতিক সহায়তার অভিযোগ কার বিরুদ্ধে ছিল?", "জগৎশেঠ", "জগৎশেঠ", "medium"),
        r(2, "image", "chart_value", "bn", "পাই চার্ট অনুযায়ী পলাশীর যুদ্ধের কারণগুলোর মধ্যে সবচেয়ে বেশি গুরুত্বপূর্ণ কারণ কী এবং কত শতাংশ?", "রাজনৈতিক ষড়যন্ত্র, ৩৫%", "রাজনৈতিক ষড়যন্ত্র;৩৫%", "medium"),
        r(2, "image", "chart_value", "bn", "পাই চার্টে বিশ্বাসঘাতকতার অংশ কত শতাংশ?", "২৫%", "২৫%;বিশ্বাসঘাতকতা", "medium"),
        r(2, "image", "chart_value", "bn", "পাই চার্টে প্রস্তুতির দুর্বলতার অংশ কত শতাংশ?", "৫%", "৫%;প্রস্তুতির দুর্বলতা", "hard"),
        r(3, "structure", "flow_order", "bn", "ঘটনার ধারাবাহিকতা ফ্লো চার্ট অনুযায়ী পলাশীর যুদ্ধের আগে সর্বশেষ ধাপ কী ছিল?", "মীর জাফরসহ দরবারি গোষ্ঠীর গোপন চুক্তি", "মীর জাফর;গোপন চুক্তি", "hard"),
        r(3, "table", "date_lookup", "bn", "কলকাতা কোন সালে নবাবের দখলে যায়?", "১৭৫৬", "১৭৫৬", "medium"),
        r(3, "table", "date_lookup", "bn", "ইংরেজরা কলকাতা কবে পুনর্দখল করে?", "১৭৫৭", "১৭৫৭", "medium"),
        r(4, "table", "short_fact", "bn", "যুদ্ধের পর কে বাংলার নবাব হন?", "মীর জাফর", "মীর জাফর;নবাব", "easy"),
        r(4, "text", "short_fact", "bn", "নবাব সিরাজউদ্দৌলা কেন পরাজিত হলেন তার একটি প্রধান কারণ কী?", "মীর জাফর ও তার অনুগত অংশ যুদ্ধক্ষেত্রে কার্যকরভাবে নবাবকে সহায়তা করেননি", "মীর জাফর;সহায়তা করেননি", "medium"),
        r(5, "text", "short_fact", "bn", "পলাশীর যুদ্ধের প্রধান ফলাফল কী?", "বাংলায় ইংরেজ ইস্ট ইন্ডিয়া কোম্পানির রাজনৈতিক প্রভাব প্রতিষ্ঠিত হয়", "ইংরেজ ইস্ট ইন্ডিয়া কোম্পানি;রাজনৈতিক প্রভাব", "medium"),
        r(5, "text", "short_fact", "bn", "পলাশীর যুদ্ধের রাজনৈতিক ফল কী ছিল?", "মীর জাফরকে নবাব করা হয়; বাংলার নবাবি ক্ষমতা ইংরেজ কোম্পানির প্রভাবাধীন হয়ে পড়ে", "মীর জাফর;নবাবি ক্ষমতা;প্রভাবাধীন", "hard"),
        r(5, "text", "short_fact", "bn", "পলাশীর যুদ্ধের ঐতিহাসিক গুরুত্ব কী?", "ভারত উপমহাদেশে ব্রিটিশ শাসনের সূচনালগ্নের অন্যতম প্রধান ঘটনা হিসেবে দেখা হয়", "ব্রিটিশ শাসন;সূচনা", "medium"),
    ]


def ugc1_rows() -> list[dict]:
    pid, pname, ptype = "UGC1", "UGC1(Public University law).pdf", "native_text"
    r = lambda *a: row("", pid, pname, ptype, *a)
    return [
        r(1, "table", "date_lookup", "bn", "ঢাকা বিশ্ববিদ্যালয়ের আইন পাস/কার্যক্রম শুরুর সাল কত?", "১৯২১", "১৯২১;ঢাকা বিশ্ববিদ্যালয়", "easy"),
        r(1, "table", "date_lookup", "bn", "রাজশাহী বিশ্ববিদ্যালয়ের প্রতিষ্ঠা সাল কত?", "১৯৫৩", "১৯৫৩;রাজশাহী বিশ্ববিদ্যালয়", "easy"),
        r(1, "table", "date_lookup", "bn", "বাংলাদেশ কৃষি বিশ্ববিদ্যালয়ের আইন অনুমোদনের সাল কত?", "১৯৬১", "১৯৬১;কৃষি বিশ্ববিদ্যালয়", "easy"),
        r(1, "table", "date_lookup", "bn", "বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়ের (বুয়েট) প্রতিষ্ঠা/কার্যক্রম শুরুর সাল কত?", "১৯৬২", "১৯৬২;প্রকৌশল বিশ্ববিদ্যালয়", "easy"),
        r(1, "table", "date_lookup", "bn", "চট্টগ্রাম বিশ্ববিদ্যালয়ের আইন কোন সালে অনুমোদিত হয়?", "১৯৬৬", "১৯৬৬;চট্টগ্রাম বিশ্ববিদ্যালয়", "easy"),
        r(1, "table", "date_lookup", "bn", "জাহাঙ্গীরনগর বিশ্ববিদ্যালয়ের প্রতিষ্ঠা সাল কত?", "১৯৭০", "১৯৭০;জাহাঙ্গীরনগর বিশ্ববিদ্যালয়", "easy"),
        r(1, "table", "date_lookup", "bn", "শাহজালাল বিজ্ঞান ও প্রযুক্তি বিশ্ববিদ্যালয়ের প্রতিষ্ঠা সাল কত?", "১৯৮৭", "১৯৮৭;শাহজালাল", "medium"),
        r(1, "table", "date_lookup", "bn", "খুলনা বিশ্ববিদ্যালয়ের প্রতিষ্ঠা সাল কত?", "১৯৯০", "১৯৯০;খুলনা বিশ্ববিদ্যালয়", "medium"),
        r(1, "table", "date_lookup", "bn", "জাতীয় বিশ্ববিদ্যালয়ের প্রতিষ্ঠা সাল কত?", "১৯৯২", "১৯৯২;জাতীয় বিশ্ববিদ্যালয়", "medium"),
        r(2, "table", "date_lookup", "bn", "বেগম রোকেয়া বিশ্ববিদ্যালয়, রংপুরের প্রতিষ্ঠা সাল কত?", "২০০৯", "২০০৯;বেগম রোকেয়া বিশ্ববিদ্যালয়", "medium"),
        r(2, "table", "date_lookup", "bn", "শেখ হাসিনা বিশ্ববিদ্যালয়ের প্রতিষ্ঠা সাল কত?", "২০১৮", "২০১৮;শেখ হাসিনা বিশ্ববিদ্যালয়", "medium"),
        r(2, "table", "date_lookup", "bn", "রবীন্দ্র বিশ্ববিদ্যালয়, বাংলাদেশ এর প্রতিষ্ঠা সাল কত?", "২০১৭", "২০১৭;রবীন্দ্র বিশ্ববিদ্যালয়", "medium"),
        r(2, "table", "date_lookup", "bn", "খুলনা কৃষি বিশ্ববিদ্যালয়ের প্রতিষ্ঠা সাল কত?", "২০১৮", "২০১৮;খুলনা কৃষি বিশ্ববিদ্যালয়", "medium"),
        r(3, "table", "date_lookup", "bn", "কুড়িগ্রাম কৃষি বিশ্ববিদ্যালয়ের প্রতিষ্ঠা সাল কত?", "২০২২", "২০২২;কুড়িগ্রাম কৃষি বিশ্ববিদ্যালয়", "hard"),
        r(3, "table", "date_lookup", "bn", "মুজিবনগর বিশ্ববিদ্যালয়, মেহেরপুরের প্রতিষ্ঠা সাল কত?", "২০২৩", "২০২৩;মুজিবনগর বিশ্ববিদ্যালয়", "hard"),
        r(3, "table", "date_lookup", "bn", "সাতক্ষীরা বিজ্ঞান ও প্রযুক্তি বিশ্ববিদ্যালয়ের আইন কোন সালে কার্যকর হয়?", "২০২৩", "২০২৩;সাতক্ষীরা বিজ্ঞান ও প্রযুক্তি বিশ্ববিদ্যালয়", "hard"),
        r(1, "table", "count_list", "bn", "এই তালিকায় মোট কতটি পাবলিক বিশ্ববিদ্যালয়ের নাম আছে?", "৬১টি", "৬১", "hard"),
        r(1, "table", "short_fact", "bn", "তালিকা অনুযায়ী সবচেয়ে পুরনো প্রতিষ্ঠা সালের বিশ্ববিদ্যালয় কোনটি?", "ঢাকা বিশ্ববিদ্যালয় (১৯২১)", "ঢাকা বিশ্ববিদ্যালয়;১৯২১", "hard"),
    ]


def ugc2_rows() -> list[dict]:
    pid, pname, ptype = "UGC2", "UGC2( Scholarship).pdf", "scanned"
    r = lambda *a: row("", pid, pname, ptype, *a)
    return [
        r(1, "text", "short_fact", "bn", "এই নোটিশটি কোন বৃত্তির জন্য আবেদন আহ্বান করছে?", "ইউজিসি বৈদেশিক মাস্টার্স/এম.ফিল/পিএইচডি বৃত্তি (UGC Overseas Masters/M.Phil/PhD Scholarship for University Teachers)", "ইউজিসি;মাস্টার্স;পিএইচডি;বৃত্তি", "easy"),
        r(1, "text", "short_fact", "bn", "এই বৃত্তি কোন নীতিমালার আলোকে ঘোষণা করা হয়েছে?", "ইউজিসি বৈদেশিক মাস্টার্স/এম.ফিল/পিএইচডি বৃত্তি নীতিমালা ২০২৫", "নীতিমালা ২০২৫", "medium"),
        r(1, "text", "date_lookup", "bn", "আবেদন জমা দেওয়ার শেষ তারিখ কত?", "০৫/০৪/২০২৬", "০৫/০৪/২০২৬", "easy"),
        r(1, "text", "short_fact", "bn", "আবেদনপত্র কোন ই-মেইলে পাঠাতে হবে?", "icdivision.ugc@gmail.com", "icdivision.ugc@gmail.com", "medium"),
        r(1, "text", "short_fact", "bn", "এই বৃত্তির জন্য কারা আবেদন করতে পারবেন?", "বিশ্ববিদ্যালয়ের শিক্ষকগণ", "বিশ্ববিদ্যালয়ের শিক্ষক", "medium"),
        r(1, "text", "short_fact", "bn", "নোটিশটি কে অনুমোদন করেছেন (স্বাক্ষরকারী পরিচালক কে)?", "মোছা: জেসমিন পারভীন", "জেসমিন পারভীন;পরিচালক", "medium"),
        r(1, "text", "short_fact", "bn", "নোটিশে উল্লেখিত স্মারক নম্বর কী?", "ইউজিসি/আইসিসিসি/৫.৭৫/২১/২০১৮-১৮/৫১৮", "স্মারক নং", "hard"),
        r(1, "text", "short_fact", "bn", "নোটিশটি কোন বিভাগ থেকে জারি করা হয়েছে?", "ইন্টারন্যাশনাল কোলাবোরেশন বিভাগ", "ইন্টারন্যাশনাল কোলাবোরেশন বিভাগ", "medium"),
    ]


def ugc3_rows() -> list[dict]:
    pid, pname, ptype = "UGC3", "UGC3 (Rokeya Chair).pdf", "scanned"
    r = lambda *a: row("", pid, pname, ptype, *a)
    return [
        r(1, "text", "short_fact", "bn", "রোকেয়া চেয়ার নীতিমালা কোন সালের?", "২০২৬", "রোকেয়া চেয়ার নীতিমালা ২০২৬", "easy"),
        r(1, "text", "date_lookup", "bn", "রোকেয়া চেয়ার নীতিমালা ২০২৬ কমিশনের কততম সভায় অনুমোদিত হয়?", "১৭৬তম সভা", "১৭৬তম", "medium"),
        r(1, "text", "date_lookup", "bn", "রোকেয়া চেয়ার কোন সাল থেকে প্রবর্তন করা হয়েছে?", "২০০৭ সাল", "২০০৭", "easy"),
        r(1, "text", "short_fact", "bn", "রোকেয়া চেয়ার কাদের সম্মানার্থে প্রদান করা হয়?", "নারী শিক্ষা, নারী ক্ষমতায়ন, নারী উন্নয়ন ও নারী সমাজের অগ্রগতির জন্য কাজ করা শিক্ষাবিদ, গবেষক ও বিশেষজ্ঞগণ", "নারী শিক্ষা;নারী ক্ষমতায়ন;নারী উন্নয়ন", "medium"),
        r(1, "text", "short_fact", "bn", "একজন অধ্যাপক গবেষককে রোকেয়া চেয়ারের জন্য কত বছরের জন্য নিয়োগ দেওয়া হয়?", "২ (দুই) বছর", "২ বছর", "easy"),
        r(1, "text", "short_fact", "bn", "রোকেয়া চেয়ারের জন্য যোগ্য অধ্যাপক গবেষকের সর্বনিম্ন বয়স কত হতে হবে?", "৫৫ (পঞ্চান্ন) বৎসর", "৫৫ বৎসর", "medium"),
        r(2, "text", "short_fact", "bn", "নিয়োগপ্রাপ্ত অধ্যাপক গবেষককে কী পরিমাণ মাসিক সম্মানী দেওয়া হবে?", "বাংলাদেশের পাবলিক বিশ্ববিদ্যালয়ের একজন সিলেকশন গ্রেডের অধ্যাপকের সর্বোচ্চ মাসিক বেতন সমতুল্য", "সিলেকশন গ্রেড;অধ্যাপক;বেতন সমতুল্য", "medium"),
        r(2, "text", "short_fact", "bn", "বই, জার্নাল, গবেষণা সামগ্রী বাবদ এককালীন সর্বোচ্চ কত টাকা দেওয়া হবে?", "৫০,০০০ (পঞ্চাশ হাজার) টাকা", "৫০,০০০ টাকা", "medium"),
        r(2, "text", "short_fact", "bn", "সেমিনার/ওয়ার্কশপ আয়োজনের জন্য সর্বোচ্চ কত টাকা প্রদান করা হবে?", "১,০০,০০০ (এক লক্ষ) টাকা", "১,০০,০০০ টাকা;এক লক্ষ", "medium"),
        r(2, "text", "short_fact", "bn", "গবেষণা সহকারী/সহযোগীর মাসিক সম্মানী কত?", "১২,০০০ (বারো হাজার) টাকা", "১২,০০০ টাকা", "medium"),
        r(2, "text", "short_fact", "bn", "অফিস সহকারী কাম কম্পিউটার টাইপিস্টের মাসিক সম্মানী কত?", "১২,০০০ (বারো হাজার) টাকা", "১২,০০০ টাকা", "hard"),
        r(2, "text", "short_fact", "bn", "মেয়াদকালে অধ্যাপক গবেষককে কতটি গবেষণা প্রবন্ধ প্রকাশ করতে হবে?", "২ (দুই)টি গবেষণা প্রবন্ধ", "২টি গবেষণা প্রবন্ধ", "medium"),
        r(3, "text", "short_fact", "bn", "একবার রোকেয়া চেয়ার হিসেবে নির্বাচিত অধ্যাপক গবেষক কি পরবর্তীতে আবার মনোনয়ন পাবেন?", "না, পরবর্তীতে এই চেয়ারের জন্য মনোনয়ন দেওয়া যাবে না", "মনোনয়ন দেওয়া যাবে না", "hard"),
        r(3, "text", "short_fact", "bn", "নীতিমালা সংশোধনের ক্ষমতা কার উপর ন্যস্ত থাকবে?", "কমিশনের উপর", "কমিশন", "medium"),
        r(3, "text", "short_fact", "bn", "রোকেয়া চেয়ার নীতিমালা ২০২৬ কে সচিব হিসেবে স্বাক্ষর করেছেন?", "ড. মোঃ ফখরুল ইসলাম", "ফখরুল ইসলাম;সচিব", "medium"),
        r(3, "text", "short_fact", "bn", "রোকেয়া চেয়ার নীতিমালা ২০২৬ এ চেয়ারম্যান হিসেবে কে স্বাক্ষর করেছেন?", "প্রফেসর ড. এস এম এ ফায়েজ", "এস এম এ ফায়েজ;চেয়ারম্যান", "medium"),
    ]


def duet_booklet_rows() -> list[dict]:
    pid, pname, ptype = "DUETBooklet", "information booklet duet.pdf", "native_text"
    r = lambda *a: row("", pid, pname, ptype, *a)
    return [
        r(1, "text", "short_fact", "en", "What is this booklet for?", "Information Booklet for Postgraduate Studies", "Information Booklet;Postgraduate Studies", "easy"),
        r(1, "text", "short_fact", "en", "Which institute offers the M.Sc/M.Engg degrees described in this booklet?", "Institute of Information & Communication Technology (IICT)", "Institute of Information & Communication Technology;IICT", "easy"),
        r(2, "text", "short_fact", "en", "What is the minimum duration of the M Sc. Engg. / M Engg. program?", "Three semesters, and generally not more than 5 academic years", "three semesters;5 academic years", "medium"),
        r(2, "text", "short_fact", "en", "What is the minimum duration of a semester?", "13 (thirteen) weeks", "13 weeks", "easy"),
        r(2, "table", "short_fact", "en", "How many total credit hours are required for the M Sc. Engg. degree?", "36 credit hours, of which 18 credit hours are for a Thesis", "36 credit hours;18;Thesis", "medium"),
        r(2, "table", "short_fact", "en", "How many total credit hours are required for the M Engg. degree?", "36 credit hours, of which 6 credit hours are for a Project", "36 credit hours;6;Project", "medium"),
        r(2, "table", "short_fact", "en", "How many theory credit hours are required for M Sc Engg vs M Engg?", "3 x 6 = 18 for M.Sc Engg, and 3 x 10 = 30 for M Engg", "18;30;theory courses", "hard"),
        r(3, "table", "code_lookup", "en", "What course does course code ICT 6000 correspond to, and how many credits?", "Thesis, 18 credits", "ICT 6000;Thesis;18", "easy"),
        r(3, "table", "code_lookup", "en", "What course does course code ICT 6001 correspond to, and how many credits?", "Project, 6 credits", "ICT 6001;Project;6", "easy"),
        r(3, "table", "code_lookup", "en", "What is the course title of ICT 6101?", "Research Methodology", "ICT 6101;Research Methodology", "medium"),
        r(4, "table", "code_lookup", "en", "What is the course code for Machine Learning, and how many credits does it carry?", "ICT 6504, 3 credits", "ICT 6504;Machine Learning;3", "medium"),
        r(4, "table", "code_lookup", "en", "What is the course code for Advanced Artificial Intelligence?", "ICT 6503", "ICT 6503;Advanced Artificial Intelligence", "medium"),
        r(4, "table", "code_lookup", "en", "What is the course code for Automated Planning?", "ICT 6505", "ICT 6505;Automated Planning", "medium"),
        r(3, "table", "code_lookup", "en", "What is the course title of ICT 6202?", "Big Data Analysis and Design", "ICT 6202;Big Data Analysis and Design", "medium"),
        r(3, "table", "code_lookup", "en", "What is the course title of ICT 6203?", "Advanced Information Theory and Coding", "ICT 6203;Advanced Information Theory and Coding", "medium"),
        r(3, "table", "code_lookup", "en", "What is the course code for Information Retrieval?", "ICT 6204", "ICT 6204;Information Retrieval", "medium"),
        r(14, "table", "code_lookup", "en", "What is the course code for Biomedical Image Processing?", "ICT 6307", "ICT 6307;Biomedical Image Processing", "medium"),
        r(15, "table", "code_lookup", "en", "What is the course code for Intrusion Management and Ethical Hacking?", "ICT 6401", "ICT 6401;Intrusion Management and Ethical Hacking", "medium"),
        r(17, "table", "code_lookup", "en", "What is the course code for Computational Linguistics?", "ICT 6501", "ICT 6501;Computational Linguistics", "medium"),
        r(18, "table", "code_lookup", "en", "What is the course title of ICT 6502?", "Statistical Machine Translation", "ICT 6502;Statistical Machine Translation", "medium"),
        r(21, "table", "code_lookup", "en", "What is the course title of ICT 6603?", "Radar Engineering", "ICT 6603;Radar Engineering", "medium"),
        r(29, "table", "code_lookup", "en", "What is the course code for Satellite and Navigation?", "ICT 6710", "ICT 6710;Satellite and Navigation", "medium"),
        r(33, "table", "code_lookup", "en", "What is the course title and credit of ICT 6900?", "Selected Topics in ICT, 3 credits", "ICT 6900;Selected Topics in ICT;3", "hard"),
        r(24, "table", "code_lookup", "en", "What is the course code for Advanced Communication Engineering?", "ICT 6702", "ICT 6702;Advanced Communication Engineering", "hard"),
    ]


def main() -> None:
    all_rows = migrate_old_rows()
    for builder in (
        agriculture_rows, bd_history_rows, polashir_juddho_rows,
        ugc1_rows, ugc2_rows, ugc3_rows, duet_booklet_rows,
    ):
        all_rows.extend(builder())

    # Renumber question_id sequentially as Q001, Q002, ...
    for i, r in enumerate(all_rows, start=1):
        r["question_id"] = f"Q{i:03d}"

    with NEW_GT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    by_pdf: dict[str, int] = {}
    for r in all_rows:
        by_pdf[r["pdf_id"]] = by_pdf.get(r["pdf_id"], 0) + 1
    print(f"Wrote {NEW_GT} with {len(all_rows)} questions total")
    for pdf_id, n in by_pdf.items():
        print(f"  {pdf_id}: {n}")


if __name__ == "__main__":
    main()
