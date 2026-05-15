#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_highlight_spans.py (ULTIMATE HYBRID EDITION)
--------------------------------------------------
- WEICHE: Ollama (Tabloid Style) / Gemini (Batch Mode)
- PROMPTS: Vollständig isolierte Prompts für Ollama und Gemini!
"""

import argparse, pathlib, re, json, yaml, time, difflib
from datetime import datetime
import srt
from core.time_utils import parse_time_to_seconds, seconds_to_hms, hms_to_seconds


try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# ====================== CONFIG ======================

DEFAULT_MODEL = "qwen2.5:14b-instruct-q6_K"

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


# ====================== LLM Wrapper ======================


def clean_thinking(text):
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


class UnifiedLLM:
    def __init__(self, config):
        self.provider = config.get("llm_provider", "ollama").lower()
        self.model = config.get("llm_model", DEFAULT_MODEL)

        self.gemini_key = config.get("gemini_api_key", "")

        if self.gemini_key and self.provider == "gemini":
            self.client = genai.Client(api_key=self.gemini_key)
            self.model = config.get("gemini_model", "gemini-1.5-flash")
            self.is_gemini = True
            print(f"🚀 GEMINI MODE AKTIV: {self.model}")
        else:
            self.client = OllamaClient(host="http://127.0.0.1:11434")
            self.is_gemini = False
            print(f"🦙 OLLAMA MODE AKTIV: {self.model}")

    def chat_json(self, sys_p, user_p):
        if self.is_gemini:
            try:
                time.sleep(1)
                res = self.client.models.generate_content(
                    model=self.model,
                    contents=user_p,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_p,
                        temperature=0.7,
                        response_mime_type="application/json",
                    ),
                )
                return res.text
            except Exception as e:
                print(f"❌ Gemini JSON Error: {e}")
                return None
        return None

    def chat(self, sys, user, temp=0.8, max_tokens=None):
        if self.is_gemini:
            try:
                time.sleep(1)
                res = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=sys, temperature=temp
                    ),
                )
                return clean_thinking(res.text)
            except Exception as e:
                print(f"LLM Error: {e}")
                return ""
        else:
            try:
                opts = {"temperature": temp, "num_ctx": 4096}
                if max_tokens:
                    opts["num_predict"] = max_tokens

                res = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                    options=opts,
                )
                return clean_thinking(res["message"]["content"])
            except Exception as e:
                print(f"LLM Error: {e}")
                return ""


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
        r"\[(\d+):(\d+):(\d+)\]\s*(?:\*\*(.*?)\*\*:?)?\s*\"?(.*?)\"?\s*$", re.MULTILINE
    )
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            sec = h * 3600 + mn * 60 + s
            desc = m.group(5).strip()
            category = m.group(4)
            if category and len(desc.split()) < 3:
                desc = f"{category}: {desc}"

            entries.append({"sec": sec, "text": desc})
    return entries


def balance_highlights_from_data(entries, total_duration, target_count):
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
        prompt = OLLAMA_PROMPT_TABLOID_TRANSFORM.format(description=description)
        raw = llm.chat("Du bist Redakteur.", prompt, temp=0.7, max_tokens=60)
        title = clean_title_final(raw)

        # Nur noch ein Sicherheitsnetz, falls Qwen komplett versagt oder zu viel faselt
        if len(title.split()) < 2 or len(title.split()) > 10:
            retry_prompt = OLLAMA_PROMPT_RETRY_LAZY.format(description=description)
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
        GEMINI_PROMPT_CAPTION_MONDAY if llm.is_gemini else OLLAMA_PROMPT_CAPTION_MONDAY
    )
    prompt_friday = (
        GEMINI_PROMPT_CAPTION_FRIDAY if llm.is_gemini else OLLAMA_PROMPT_CAPTION_FRIDAY
    )

    sys_role_m = (
        "Du bist ein eleganter Copywriter."
        if llm.is_gemini
        else "Du bist Social Media Manager."
    )
    sys_role_f = (
        "Du bist ein eleganter Copywriter."
        if llm.is_gemini
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
    ap.add_argument("--target-count", type=int, default=6)
    ap.add_argument("--target-min", type=int, default=15)
    ap.add_argument("--target-max", type=int, default=60)
    ap.add_argument("--llm-titles", action="store_true")
    ap.add_argument("--no-llm", action="store_true")

    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--gemini-api-key", default="")
    ap.add_argument("--gemini-model", default="gemini-1.5-flash")

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

    # 2. Argumente als Fallback/Override zulassen
    if args.gemini_api_key:
        config["gemini_api_key"] = args.gemini_api_key
    if args.llm_model:
        config["llm_model"] = args.llm_model

    # DIESE ZEILE FEHLT BEI DIR:
    if args.gemini_model:
        config["gemini_model"] = args.gemini_model

    if "llm_provider" not in config:
        config["llm_provider"] = "ollama"

    use_llm = not args.no_llm

    llm = None
    if use_llm:
        print("\n" + "=" * 40)
        llm = UnifiedLLM(config)
        print("=" * 40 + "\n")
    else:
        print("\n⚠️  WARNUNG: LLM Modus ist DEAKTIVIERT. Erstelle Placeholder-Titel.\n")

    subs = load_srt(args.srt)
    total_duration = subs[-1].end.total_seconds() if subs else 0

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
        entries, total_duration, args.target_count
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

        # A. SMART CUT
        end_s = start_s + 20
        if llm and not llm.is_gemini:
            ctx_timed = collect_context_timed(subs, start_s, args.target_max + 10)
            end_resp = llm.chat(
                "Cutter",
                OLLAMA_PROMPT_PICK_END.format(
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
                        end_s = int(s.end.total_seconds())
                        break

        # B. TITLE GENERATION
        title = "Clip Highlight"
        if llm and llm.is_gemini:
            title = generated_gemini_titles.get(i, clean_title_final(description))
            if not title:
                title = " ".join(description.split()[:5]) + "."
        elif llm and not llm.is_gemini:
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

        lines.append(f"- [{seconds_to_hms(start_s)} -> {seconds_to_hms(end_s)}] {title}")
        spans.append(
            {
                "start": seconds_to_hms(start_s),
                "end": seconds_to_hms(end_s),
                "title": title,
                "monday": posts["monday"],
                "friday": posts["friday"],
            }
        )

        md_block = (
            f"### Reel {i}: {title}\n"
            f"**Clip:** {seconds_to_hms(start_s)} - {seconds_to_hms(end_s)}\n\n"
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
path = pathlib.Path(args.out_md)
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
