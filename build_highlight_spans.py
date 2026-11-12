#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_highlight_spans.py (ULTIMATE HYBRID EDITION)
--------------------------------------------------
- WEICHE: Ollama (Tabloid Style) / Gemini (Batch Mode)
- PROMPTS: Vollständig isolierte Prompts für Ollama und Gemini!
"""

import argparse, pathlib, re, json, yaml, difflib, math
from datetime import datetime
from typing import Optional
import srt
from core.time_utils import parse_time_to_seconds, seconds_to_hms, seconds_to_hms_ms, hms_to_seconds
from core.llm_client import UnifiedLLM, clean_thinking
from core.span_contract import ClipSpan, SpanValidationError, clamp_end_to_max, validate_span
from domains.transcription.processor import SubtitleProcessor

# ====================== CONFIG ======================

def _set_model_for_provider(config, provider, model):
    """Write an explicit model into the active provider's field."""
    provider = str(provider).lower()
    if provider == "gemini-api":
        provider = "gemini"
    if provider == "openrouter":
        nested = config.get("openrouter")
        if isinstance(nested, dict):
            nested["model"] = model
        else:
            config["openrouter_model"] = model
    elif provider == "gemini":
        config["gemini_model"] = model
    elif provider == "ollama":
        config["ollama_model"] = model
    elif provider == "openai":
        config["openai_model"] = model
    elif provider == "llama-server":
        config["llama_server_model"] = model
    # Keep the legacy generic field populated for compatibility; provider-specific
    # fields still win in UnifiedLLM.
    config["llm_model"] = model


def apply_cli_model_overrides(config, args):
    """Apply only explicitly supplied CLI provider/model/key overrides."""
    llm_provider = getattr(args, "llm_provider", None)
    if llm_provider:
        config["llm_provider"] = llm_provider
        config["llm_provider_explicit"] = True

    provider = str(llm_provider or config.get("llm_provider") or "ollama").lower()
    if provider == "gemini-api":
        provider = "gemini"

    if getattr(args, "gemini_api_key", None):
        config["gemini_api_key"] = args.gemini_api_key
        config["gemini_api_key_explicit"] = True
    if getattr(args, "openrouter_api_key", None):
        config["openrouter_api_key"] = args.openrouter_api_key
        config["openrouter_api_key_explicit"] = True
    if getattr(args, "llm_model", None):
        _set_model_for_provider(config, provider, args.llm_model)
        config["model_explicit"] = True
    if getattr(args, "gemini_model", None):
        config["gemini_model"] = args.gemini_model
        config["gemini_model_explicit"] = True
        if provider == "gemini":
            config["model_explicit"] = True
    return config

# ====================== 1. OLLAMA PROMPTS (Streng & Strukturiert) ======================

OLLAMA_PROMPT_TABLOID_TRANSFORM = """
AUFGABE: Du bist ein brillanter Social Media Redakteur. Finde für diesen Video-Clip den perfekten, aufmerksamkeitsstarken Titel (Visual Hook).

REGELN:
1. KLARE AUSSAGEN: Nutze direkte, starke Statements. Verboten sind Formulierungen wie "Was wäre, wenn..." oder "Wie ein...". 
2. KEINE KIRCHENSPRACHE: Vermeide fromme Begriffe ("Gott", "Jesus", "Segen").
3. SINNHAFTIGKEIT: Keine erzwungenen Reime oder unsinnigen Alliterationen! Der Satz muss logisch Sinn ergeben.
4. GANZ NATÜRLICHES DEUTSCH: Nutze fließendes Deutsch.
5. Maximal 3-8 Wörter.
6. Setze am Ende IMMER ein Ausrufezeichen (!) oder einen Punkt (.). Setze NIEMALS ein Fragezeichen.

BEISPIELE FÜR DEINEN VIBE:
- "Vierte Klasse ist zu spät!"
- "Körper schwindet, Geist bleibt stark!"
- "Hauptsache gesund stimmt absolut nicht!"
- "Warum manche Menschen unzerbrechlich sind."

JETZT DU!
Input Beschreibung: "{description}"
Dein Titel (NUR DER TEXT, ZWINGEND AUF DEUTSCH):
"""

OLLAMA_PROMPT_RETRY_LAZY = """
FEHLER: Der Titel war zu langweilig, eine Frage oder enthielt Kirchensprache.
AUFGABE: Schreibe eine knackige AUSSAGE. Nutze alltagsnahes, modernes Deutsch. Keine Fragen!

Alter Text: "{description}"
Deine NEUE Version (3-8 Wörter, mit Punkt oder Ausrufezeichen am Ende):
"""

OLLAMA_PROMPT_PICK_END = """Finde das Satzende. Länge {min_s}-{max_s}s.
Transkript: {context}
Antworte NUR mit Zeitstempel (HH:MM:SS)."""

# Provider-neutral aliases: these prompts are not Ollama-specific.
PROMPT_TABLOID_TRANSFORM = OLLAMA_PROMPT_TABLOID_TRANSFORM
PROMPT_RETRY_LAZY = OLLAMA_PROMPT_RETRY_LAZY
PROMPT_PICK_END = OLLAMA_PROMPT_PICK_END

OLLAMA_PROMPT_CAPTION_MONDAY = """
Du bist Social Media Manager. Zielgruppe: Menschen, die im Alltag gestresst sind und viel zu tun haben.
Schreibe einen authentischen Instagram-Post für MONTAGMORGEN.

INPUT:
- Transkript: "{context}"
- Titel: "{title}"

AUFGABE:
1. **HEADLINE:** Emotionale, kreative Überschrift + Emoji. (WICHTIG: Nutze nicht immer Standard-Sätze wie "Ein neuer Anfang", sondern sei spezifisch zum Text!)
2. **BODY:** Greife einen Gedanken aus dem Video auf und übertrage ihn auf den Start in die Arbeitswoche. 
   -> WICHTIGSTE REGELN: 
   - Achte auf PERFEKTE deutsche Grammatik. Keine erfundenen Wörter!
   - DU-FORM: Sprich den Leser IMMER mit "Du" oder "Ihr" an. Nutze NIEMALS das förmliche "Sie".
   - Starte NIEMALS mit Begrüßungen wie "Hey" oder "Hallo". Steig direkt mit dem ersten Satz ins Thema ein.
3. **CHALLENGE:** Eine konkrete Mini-Aufgabe für heute.
4. **FRAGE:** Eine offene Frage für die Kommentare.

FORMAT (Strikt - Drucke KEINE eckigen Klammern aus!):
HEADLINE: Deine Überschrift
BODY: Dein Text + Challenge + Frage + Hashtags
"""

OLLAMA_PROMPT_CAPTION_FRIDAY = """
Du bist Social Media Manager einer Kirchengemeinde. Schreibe einen Post für FREITAG, der Vorfreude aufs Wochenende macht und herzlich zum Gottesdienst einlädt.

INPUT VIDEO:
- Transkript: "{context}"

INPUT SONNTAG:
- Datum: {next_date}
- Prediger: {next_preacher}
- THEMA: "{next_topic}" (Kann leer sein!)
- Specials: {next_specials}

LOGIK & REGELN:
1. SPRACHE: Perfektes, natürliches Deutsch! Immer in der "Du" oder "Ihr" Form.
2. ROLLENKLÄRUNG: Du sprichst als das "Wir" der Gemeinde. Tu NIEMALS so, als wärst du selbst der Pastor!
3. VERBOTEN: Nutze KEINE Floskeln wie "Hey liebe Gemeinde" oder "Hallo". Steig direkt mit dem Video-Gedanken ein!
4. Nutze das Video als Brücke zur Gottesdienst-Einladung.
5. ZEITLICHE LOGIK: Der Post erscheint Freitag, der Gottesdienst ist am Sonntag.

FORMAT (Strikt - Halte dich EXAKT an diese Struktur, drucke KEINE eckigen Klammern aus!):
HEADLINE: Deine Vorfreude-Überschrift
BODY: Dein Text + Einladung

📅 Wann: Das Datum und die Uhrzeit
🗣️ Predigt: Der Prediger
✨ Specials: Die Specials (Nur wenn welche vorhanden sind, sonst weglassen)

[Hier 4-5 passende Hashtags]
"""

# ====================== 2. GEMINI PROMPTS (Kreativ & Fließend) ======================

GEMINI_BATCH_SYSTEM_PROMPT = """Du bist ein Experte für virale Video-Titel auf TikTok/Reels.
Generiere für jeden Clip einen genialen "Visual Hook".

REGELN:
1. Maximal 4-7 Wörter in perfektem, modernem Deutsch.
2. STARTE NIE MIT "UND", "ABER", "DENN".
3. Setze IMMER ein Satzzeichen ans Ende (! oder ? oder .).
4. Provokant, emotional oder neugierig machend ("Open Loop").

Antworte strikt im JSON-Format:
[{"id": 1, "title": "...!"}, {"id": 2, "title": "...?"}, ...]
"""

GEMINI_BATCH_USER_TEMPLATE = """Hier sind die Clips.
--- CLIPS ---
{clips_text}
"""

GEMINI_PROMPT_CAPTION_MONDAY = """
Erstelle als erfahrener Copywriter einen inspirierenden, hochwertigen Instagram-Post für Montagmorgen. 
Zielgruppe: Menschen im Berufsleben, die einen Impuls für die neue Woche brauchen.

Basis ist dieses Video-Transkript: "{context}" (Titel des Clips: "{title}").

VORGABEN FÜR DEN STIL:
- Schreibe extrem flüssig und elegant. 
- Steig sofort tief ins Thema ein (absolutes Verbot von Begrüßungen wie "Hey Leute", "Guten Morgen" etc. – das ist unprofessionell).
- Schlage den Bogen vom Video-Thema zum Start in die Arbeitswoche.
- Schließe mit einem praktischen Impuls (Mini-Challenge) und einer Frage für die Kommentare ab.

GIB GENAU DIESES FORMAT AUS:
HEADLINE: [Deine starke, kurze Headline mit einem Emoji]
BODY: [Dein fließender Text, Challenge, Frage und 3-4 passende Hashtags]
"""

GEMINI_PROMPT_CAPTION_FRIDAY = """
Verfasse einen herzlichen und stilvollen Instagram-Post für Freitagabend. Der Post soll die Follower zum Nachdenken anregen und sie fließend in eine Einladung zum kommenden Sonntags-Gottesdienst überleiten.

Videogedanke als Aufhänger: "{context}"

Fakten für Sonntag:
- Datum & Zeit: {next_date}
- Prediger: {next_preacher}
- Thema: "{next_topic}" (Falls leer, fokussiere dich komplett auf den Videogedanken)
- Specials: {next_specials}

VORGABEN FÜR DEN STIL:
- Keine abgedroschenen Begrüßungen ("Hallo Gemeinde"). Beginne direkt mit einer starken Aussage oder Frage, die sich auf das Video bezieht.
- Sei warm, echt und einladend. 
- Präsentiere die Fakten für den Sonntag am Ende sehr übersichtlich.

GIB GENAU DIESES FORMAT AUS:
HEADLINE: [Stimmungsvolle Headline]
BODY: [Dein Text, die Einladung, die Fakten und Hashtags]
"""

SCORE_SYSTEM_PROMPT = """Du bist ein Social-Media-Redakteur für virale Reels.
Bewerte Predigt-Highlight-Kandidaten mit drei Kriterien auf einer Skala von 1 bis 10:
1. hook_strength: Sofortige Aufmerksamkeit und Neugier-Faktor in den ersten 3 Sekunden.
2. standalone_clarity: Verständlichkeit ohne Vorwissen aus dem Rest der Predigt.
3. emotional_or_practical_value: Konkreter Alltagsbezug, Tiefgang oder Humor.

Antworte ausschließlich mit gültigem JSON. Keine Erklärungen, kein Markdown."""

SCORE_USER_PROMPT = """Bewerte ALLE folgenden Kandidaten. Jeder Kandidat hat eine ID,
eine Zeitposition (Anfang, Hauptteil, Schluss) und einen Beschreibungstext.

--- KANDIDATEN ---
{clips_json}

Antworte exakt in diesem JSON-Schema:
{{"scores": [{{"id": 1, "hook_strength": 8, "standalone_clarity": 7, "emotional_or_practical_value": 9}}, ...]}}
"""


# ====================== LLM Wrapper ======================

# UnifiedLLM and clean_thinking are imported from core.llm_client.


# ====================== Helper Functions ======================


def load_srt(path):
    return list(srt.parse(pathlib.Path(path).read_text(encoding="utf-8")))


def collect_context_timed(subs, start_s, duration=60):
    buf = []
    for s in subs:
        if s.end.total_seconds() < start_s:
            continue
        if s.start.total_seconds() > start_s + duration:
            break
        buf.append(f"[{seconds_to_hms(s.end.total_seconds())}] {s.content.strip()}")
    return "\n".join(buf)


def get_next_sunday_info(yaml_path):
    defaults = {
        "date": "diesen Sonntag, 10:00 Uhr",
        "prediger": "unserem Team",
        "thema": "",
        "specials": "Gute Musik & Kaffee",
    }
    if not yaml_path:
        return defaults
    if not yaml_path.exists():
        return defaults

    try:
        content = yaml_path.read_text(encoding="utf-8")
        parts = content.split("---")
        yaml_content = parts[1] if len(parts) >= 3 else content

        data = yaml.safe_load(yaml_content)
        now = datetime.now()

        for t in data.get("termine", []):
            raw_date = str(t.get("date", "")).strip().strip(".")
            if not raw_date:
                continue
            try:
                d_day, d_month = map(int, raw_date.split("."))
                y = now.year if not (now.month == 12 and d_month == 1) else now.year + 1
                srv_date = datetime(y, d_month, d_day, 10, 0)

                if srv_date.date() >= now.date():
                    info = defaults.copy()
                    info["date"] = f"Sonntag, {raw_date}., {t.get('time', '10:00')} Uhr"
                    if t.get("prediger"):
                        info["prediger"] = t["prediger"]
                    if t.get("thema"):
                        info["thema"] = str(t["thema"]).strip()

                    specs = []
                    if t.get("icon1"):
                        specs.append("Kindergottesdienst 🦊")
                    if t.get("icon2"):
                        specs.append("Abendmahl 🍞")
                    if specs:
                        info["specials"] = " + ".join(specs)
                    return info
            except:
                continue
    except Exception as e:
        print(f"⚠️ Fehler beim Lesen der besuchen.md: {e}")
    return defaults


# --- LONGLIST PARSER ---
def parse_longlist(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    entries = []
    pattern = re.compile(
        r"\[(\d+):(\d+):(\d+(?:\.\d+)?)\]\s*(?:\*\*(.*?)\*\*:?)?\s*\"?(.*?)\"?\s*$",
        re.MULTILINE,
    )
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            sec = hms_to_seconds(m.group(1), m.group(2), m.group(3))
            desc = m.group(5).strip()
            category = m.group(4)
            if category and len(desc.split()) < 3:
                desc = f"{category}: {desc}"

            entries.append({"sec": sec, "text": desc})
    return entries


def parse_highlights_json(path):
    """Parse the strict highlights.json payload into longlist-style entries."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("highlights", data) if isinstance(data, dict) else data
    entries = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = parse_time_to_seconds(item.get("start", 0.0))
        title = str(item.get("title", "")).strip()
        reason = str(item.get("reason", "")).strip()
        text = f"{title}: {reason}" if reason else title
        entries.append({"sec": start, "text": text})
    return entries


def finalize_clip_span(
    start_s: float,
    end_s: float,
    title: str,
    media_duration: Optional[float],
    min_duration: float,
    max_duration: float,
) -> ClipSpan:
    """Clamp an over-long end and validate the final clip span."""
    end_s = clamp_end_to_max(start_s, end_s, max_duration, media_duration)
    return validate_span(
        ClipSpan(start=start_s, end=end_s, title=title, finalized=True),
        media_duration=media_duration,
        min_duration=min_duration,
        max_duration=max_duration,
    )


def _position_label(sec, total_duration):
    if total_duration <= 0:
        return "unknown"
    ratio = sec / total_duration
    if ratio < 1 / 3:
        return "Anfang"
    if ratio < 2 / 3:
        return "Hauptteil"
    return "Schluss"


def score_candidates_with_llm(entries, total_duration, llm):
    """Score all candidates in one structured JSON batch request."""
    if not llm or not entries:
        return None

    clips = []
    for i, entry in enumerate(entries, 1):
        clips.append({
            "id": i,
            "start": seconds_to_hms_ms(entry["sec"]),
            "position": _position_label(entry["sec"], total_duration),
            "text": entry["text"],
        })

    prompt = SCORE_USER_PROMPT.format(
        clips_json=json.dumps(clips, ensure_ascii=False, indent=2)
    )

    try:
        raw = llm.chat_json(SCORE_SYSTEM_PROMPT, prompt)
        if not raw:
            return None

        data = json.loads(raw)
        if isinstance(data, list):
            score_items = data
        elif isinstance(data, dict):
            score_items = data.get("scores", [])
        else:
            return None

        scores = {}
        for item in score_items:
            if not isinstance(item, dict):
                continue
            try:
                cid = int(item["id"])
                hook = max(1, min(10, int(item["hook_strength"])))
                clarity = max(1, min(10, int(item["standalone_clarity"])))
                value = max(1, min(10, int(item["emotional_or_practical_value"])))
            except (KeyError, TypeError, ValueError):
                continue
            scores[cid] = hook + clarity + value

        return scores or None
    except Exception as e:
        print(f"⚠️ LLM-Scoring fehlgeschlagen: {e}")
        return None


def _fallback_balance(entries, total_duration, target_count):
    if not entries:
        return []
    entries.sort(key=lambda x: x["sec"])
    selected = []
    sector_size = total_duration / target_count
    used_indices = set()

    for i in range(target_count):
        sector_start = i * sector_size
        sector_end = (i + 1) * sector_size
        candidates = []
        for idx, e in enumerate(entries):
            if idx in used_indices:
                continue
            if sector_start <= e["sec"] < sector_end:
                candidates.append((idx, e))

        if candidates:
            idx, best = candidates[0]
            selected.append(best)
            used_indices.add(idx)
        else:
            remaining = []
            for idx, e in enumerate(entries):
                if idx not in used_indices:
                    remaining.append((idx, e))
            if remaining:
                idx, best = min(
                    remaining, key=lambda x: abs(x[1]["sec"] - sector_start)
                )
                selected.append(best)
                used_indices.add(idx)

    return sorted(selected, key=lambda x: x["sec"])


def balance_highlights_from_data(
    entries,
    total_duration,
    target_count,
    llm=None,
    min_gap_sec=30,
):
    if not entries or target_count <= 0:
        return []

    entries.sort(key=lambda x: x["sec"])
    scores = score_candidates_with_llm(entries, total_duration, llm)
    if not scores:
        return _fallback_balance(entries, total_duration, target_count)

    selected = []
    selected_secs = []
    used_indices = set()
    sector_size = total_duration / target_count

    for i in range(target_count):
        sector_start = i * sector_size
        sector_end = (i + 1) * sector_size

        candidates = []
        for idx, e in enumerate(entries):
            if idx in used_indices:
                continue
            if sector_start <= e["sec"] < sector_end:
                candidates.append((idx, e))

        # Highest LLM score first, then earlier position as tie-breaker.
        candidates.sort(key=lambda item: (-scores.get(item[0] + 1, 0), item[1]["sec"]))

        chosen = None
        for idx, e in candidates:
            if all(abs(e["sec"] - s) >= min_gap_sec for s in selected_secs):
                chosen = (idx, e)
                break

        if chosen is None:
            remaining = [
                (idx, e)
                for idx, e in enumerate(entries)
                if idx not in used_indices
            ]
            remaining.sort(
                key=lambda item: (
                    -scores.get(item[0] + 1, 0),
                    abs(item[1]["sec"] - sector_start),
                )
            )
            for idx, e in remaining:
                if all(abs(e["sec"] - s) >= min_gap_sec for s in selected_secs):
                    chosen = (idx, e)
                    break

        if chosen is not None:
            idx, e = chosen
            selected.append(e)
            used_indices.add(idx)
            selected_secs.append(e["sec"])

    return sorted(selected, key=lambda x: x["sec"])


# --- TITLE CLEANING & SOCIAL MEDIA LOGIC ---


def clean_title_final(text):
    if not text:
        return ""
    text = re.sub(
        r"^(Headline|Titel|Schlagzeile|Antwort)[:\s]*", "", text, flags=re.IGNORECASE
    )
    text = text.replace("...", "").replace("..", "").replace("…", "")
    text = text.replace('"', "").replace("'", "").replace("*", "").replace("_", "")

    lower = text.lower()
    bad_starts = [
        "und ",
        "aber ",
        "wo ",
        "denn ",
        "weil ",
        "als ",
        "dass ",
        "ein ",
        "eine ",
        "der ",
        "die ",
        "das ",
    ]
    for bad in bad_starts:
        if lower.startswith(bad):
            text = text[len(bad) :]
            lower = text.lower()

    text = text.strip(" ,;:-")
    if len(text) > 0:
        text = text[0].upper() + text[1:]

    if ":" in text:
        parts = text.split(":")
        if len(parts[0].split()) < 2:
            text = parts[1].strip()

    return text


def llm_transform_title(llm, description):
    try:
        # Wir geben Qwen den Text und lassen ihn den besten Titel daraus formen
        prompt = PROMPT_TABLOID_TRANSFORM.format(description=description)
        raw = llm.chat("Du bist Redakteur.", prompt, temp=0.7, max_tokens=60)
        title = clean_title_final(raw)

        # Nur noch ein Sicherheitsnetz, falls Qwen komplett versagt oder zu viel faselt
        if len(title.split()) < 2 or len(title.split()) > 10:
            retry_prompt = PROMPT_RETRY_LAZY.format(description=description)
            raw = llm.chat("Du bist Redakteur.", retry_prompt, temp=0.9, max_tokens=60)
            title = clean_title_final(raw)

        if len(title.split()) < 2:
            return "Highlight ansehen."

        # Sicherstellen, dass das geforderte Satzzeichen da ist
        if not title[-1] in ".!?":
            title += "!"

        return title
    except Exception as e:
        return "Highlight."


def parse_llm_response(raw_text, default_headline):
    headline = ""
    body = ""
    curr = None
    for line in raw_text.splitlines():
        if re.match(r"^\**HEADLINE:?\**", line, re.IGNORECASE):
            headline = (
                re.sub(r"^\**HEADLINE:?\**\s*", "", line, flags=re.IGNORECASE)
                .strip()
                .strip('"')
            )
            curr = "h"
        elif re.match(r"^\**BODY:?\**", line, re.IGNORECASE):
            b_start = re.sub(r"^\**BODY:?\**\s*", "", line, flags=re.IGNORECASE).strip()
            if b_start:
                body = b_start
            curr = "b"
        else:
            if curr == "b":
                body += "\n" + line
            elif curr == "h" and not body and line.strip():
                headline += " " + line

    if not headline:
        headline = default_headline
    if not body:
        body = raw_text
    return headline, body.strip()


def llm_generate_posts(llm, title, context, next_sunday_info):
    res = {
        "monday": {"headline": title, "body": "Link in Bio."},
        "friday": {"headline": title, "body": "Wir sehen uns Sonntag!"},
    }

    # 1. WEICHE: Den richtigen Prompt anhand des Modells wählen
    prompt_monday = (
        GEMINI_PROMPT_CAPTION_MONDAY if llm.is_cloud_llm else OLLAMA_PROMPT_CAPTION_MONDAY
    )
    prompt_friday = (
        GEMINI_PROMPT_CAPTION_FRIDAY if llm.is_cloud_llm else OLLAMA_PROMPT_CAPTION_FRIDAY
    )

    sys_role_m = (
        "Du bist ein eleganter Copywriter."
        if llm.is_cloud_llm
        else "Du bist Social Media Manager."
    )
    sys_role_f = (
        "Du bist ein eleganter Copywriter."
        if llm.is_cloud_llm
        else "Du bist Community Manager."
    )

    # Monday
    try:
        m_prompt = prompt_monday.format(title=title, context=context)
        m_raw = llm.chat(sys_role_m, m_prompt, temp=0.4)
        h, b = parse_llm_response(m_raw, title)
        res["monday"] = {"headline": h, "body": b}
    except Exception as e:
        print(f"Fehler Monday Gen: {e}")

    # Friday
    try:
        f_prompt = prompt_friday.format(
            context=context,
            next_date=next_sunday_info["date"],
            next_preacher=next_sunday_info["prediger"],
            next_topic=next_sunday_info["thema"],
            next_specials=next_sunday_info["specials"],
        )
        f_raw = llm.chat(sys_role_f, f_prompt, temp=0.3)
        h, b = parse_llm_response(f_raw, title)
        res["friday"] = {"headline": h, "body": b}
    except Exception as e:
        print(f"Fehler Friday Gen: {e}")

    return res


# ====================== MAIN ======================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlights", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json")
    ap.add_argument("--transcription-json")
    ap.add_argument("--target-count", type=int, default=6)
    ap.add_argument("--target-min", type=int, default=15)
    ap.add_argument("--target-max", type=int, default=60)
    ap.add_argument("--min-gap-sec", type=int, default=30)
    ap.add_argument("--llm-titles", action="store_true")
    ap.add_argument("--no-llm", action="store_true")

    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-provider", default=None)
    ap.add_argument("--gemini-api-key", default=None)
    ap.add_argument("--openrouter-api-key", default=None)
    ap.add_argument("--gemini-model", default=None)

    ap.add_argument("--title-min")
    ap.add_argument("--title-max")
    ap.add_argument("--church-yaml")
    ap.add_argument("--end-boundary")
    ap.add_argument("--end-forward-tol")
    ap.add_argument("--title-model")

    args = ap.parse_args()

    # 1. Config aus der Datei laden
    config = {}
    if pathlib.Path("config.yaml").exists():
        config = (
            yaml.safe_load(pathlib.Path("config.yaml").read_text(encoding="utf-8"))
            or {}
        )

    # 2. Nur explizit gesetzte Argumente als Override zulassen
    config = apply_cli_model_overrides(config, args)

    if "llm_provider" not in config:
        config["llm_provider"] = "ollama"

    use_llm = not args.no_llm

    llm = None
    if use_llm:
        print("\n" + "=" * 40)
        llm = UnifiedLLM(config)
        print(f"LLM: {llm.provider} / {llm.model}")
        print("=" * 40 + "\n")
    else:
        print("\n⚠️  WARNUNG: LLM Modus ist DEAKTIVIERT. Erstelle Placeholder-Titel.\n")

    subs = load_srt(args.srt)
    total_duration = 0.0
    media_duration: Optional[float] = None
    if subs:
        raw_duration = subs[-1].end.total_seconds()
        if math.isfinite(raw_duration) and raw_duration > 0:
            total_duration = raw_duration
            media_duration = raw_duration
            print(f"ℹ️ Mediendauer-Proxy aus SRT-Ende: {raw_duration:.3f}s")
        else:
            print(
                f"⚠️ SRT-Ende als Mediendauer-Proxy ungültig: {raw_duration!r}; "
                "Media-Boundary unbekannt"
            )
    else:
        print("⚠️ Keine SRT-Daten; Mediendauer unbekannt")

    processor = SubtitleProcessor()
    transcription_json = (
        pathlib.Path(args.transcription_json)
        if args.transcription_json
        else None
    )
    words = (
        processor.load_words_from_json(transcription_json)
        if transcription_json and transcription_json.exists()
        else []
    )

    hl_path = pathlib.Path(args.highlights)
    longlist_path = hl_path.parent / "highlights_longlist.md"

    if hl_path.suffix.lower() == ".json":
        print(f"✅ Nutze JSON Highlights: {hl_path.name}")
        entries = parse_highlights_json(hl_path)
    elif longlist_path.exists():
        print(f"✅ Nutze Longlist-Beschreibungen: {longlist_path.name}")
        entries = parse_longlist(longlist_path)
    else:
        print(f"⚠️ Keine Longlist gefunden. Nutze Fallback.")
        entries = parse_longlist(hl_path)

    if not entries:
        print("❌ Keine Highlights gefunden. Abbruch.")
        return

    print(f"Wähle {args.target_count} beste Clips aus {len(entries)} Kandidaten...")
    selected_entries = balance_highlights_from_data(
        entries,
        total_duration,
        args.target_count,
        llm=llm,
        min_gap_sec=args.min_gap_sec,
    )

    # Sunday Info laden
    next_sunday = get_next_sunday_info(
        pathlib.Path(args.church_yaml) if args.church_yaml else None
    )
    if args.church_yaml:
        print(
            f"ℹ️  Infos für Sonntag geladen: {next_sunday['date']} (Thema: '{next_sunday['thema']}')"
        )

    lines, spans, captions_output = [], [], []
    generated_gemini_titles = {}

    # --- GEMINI BATCH VERARBEITUNG FÜR TITEL ---
    if llm and llm.is_gemini and args.llm_titles:
        print("🚀 Sende Batch-Request für alle Titel an Gemini...")
        clips_text = "".join(
            [
                f"\nCLIP {i}:\n{item['text']}\n---"
                for i, item in enumerate(selected_entries, 1)
            ]
        )
        full_prompt = GEMINI_BATCH_USER_TEMPLATE.format(clips_text=clips_text)
        json_str = llm.chat_json(GEMINI_BATCH_SYSTEM_PROMPT, full_prompt)

        if json_str:
            try:
                data_list = json.loads(
                    json_str.replace("```json", "").replace("```", "").strip()
                )
                for item in data_list:
                    generated_gemini_titles[item["id"]] = clean_title_final(
                        item.get("title", "")
                    )
            except Exception as e:
                print(f"❌ JSON Parse Error: {e}")

    for i, item in enumerate(selected_entries, 1):
        start_s = item["sec"]
        description = item["text"]

        # A.0 ACOUSTIC START SNAP (only when WhisperX words are available)
        if words:
            start_s = processor.snap_start_to_sentence_begin(words, start_s)

        # A. SMART CUT
        end_s = start_s + 20
        if llm and not llm.is_gemini:
            ctx_timed = collect_context_timed(subs, start_s, args.target_max + 10)
            end_resp = llm.chat(
                "Cutter",
                PROMPT_PICK_END.format(
                    min_s=args.target_min, max_s=args.target_max, context=ctx_timed
                ),
                temp=0.1,
                max_tokens=15,
            )
            try:
                parsed_end = parse_time_to_seconds(end_resp.strip())
                if parsed_end > start_s:
                    end_s = parsed_end
            except:
                pass
        else:
            for s in subs:
                if s.end.total_seconds() > start_s + args.target_min:
                    if re.search(r"[.!?]", s.content):
                        end_s = s.end.total_seconds()
                        break

        # A.2 ACOUSTIC PAUSE SNAP (only when WhisperX words are available)
        if words:
            end_s = processor.snap_end_to_speech_pause(
                words,
                chosen_end=end_s,
                start_sec=start_s,
                min_duration=args.target_min,
                max_duration=args.target_max,
            )

        # A.3 FINAL VALIDATION: clamp over-long ends, reject impossible spans.
        try:
            final_span = finalize_clip_span(
                start_s,
                end_s,
                "",
                media_duration,
                args.target_min,
                args.target_max,
            )
        except SpanValidationError as e:
            print(f"⚠️ Clip {i} verworfen: {e}")
            continue
        start_s = final_span.start
        end_s = final_span.end

        # B. TITLE GENERATION
        title = "Clip Highlight"
        if llm and llm.is_gemini:
            title = generated_gemini_titles.get(i, clean_title_final(description))
            if not title:
                title = " ".join(description.split()[:5]) + "."
        elif llm:
            title = llm_transform_title(llm, description)
        else:
            title = " ".join(description.split()[:5]) + "."

        # C. SOCIAL MEDIA POSTS
        posts = {
            "monday": {"headline": title, "body": ""},
            "friday": {"headline": title, "body": ""},
        }
        if llm:
            print(f"   -> Generiere Social Media Posts für Clip {i}...")
            ctx_for_posts = collect_context_timed(subs, start_s, end_s - start_s + 10)
            posts = llm_generate_posts(llm, title, ctx_for_posts, next_sunday)

        lines.append(f"- [{seconds_to_hms_ms(start_s)} -> {seconds_to_hms_ms(end_s)}] {title}")
        spans.append(
            {
                "start": seconds_to_hms_ms(start_s),
                "end": seconds_to_hms_ms(end_s),
                "title": title,
                "monday": posts["monday"],
                "friday": posts["friday"],
                "finalized": True,
            }
        )

        md_block = (
            f"### Reel {i}: {title}\n"
            f"**Clip:** {seconds_to_hms_ms(start_s)} - {seconds_to_hms_ms(end_s)}\n\n"
            f"#### 🗓️ MONDAY\n"
            f"**Headline:** {posts['monday']['headline']}\n\n"
            f"{posts['monday']['body']}\n\n"
            f"#### ⛪ FRIDAY\n"
            f"**Headline:** {posts['friday']['headline']}\n\n"
            f"{posts['friday']['body']}\n\n"
            f"---\n"
        )
        captions_output.append(md_block)
        print(f"   Clip {i} ({seconds_to_hms(start_s)}): {title}")

    # Output
    out_md_path = pathlib.Path(args.out_md)
    out_md_path.write_text("\n".join(lines), encoding="utf-8")

    captions_file = out_md_path.parent / "reels_content.md"
    captions_file.write_text("\n".join(captions_output), encoding="utf-8")
    print(f"📄 Captions gespeichert in: {captions_file.name}")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(
            json.dumps(spans, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print("Fertig.")


if __name__ == "__main__":
    main()
