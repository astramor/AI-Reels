#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
core/prompts.py
---------------
Zentrale Sammlung aller LLM-Prompts für die Reel-Pipeline.
Konsolidiert aus summarizer.py, srt_summarizer_local.py und build_highlight_spans.py.
"""

# ==============================================================================
# 1. SUMMARY & HIGHLIGHT SCOUTING (SermonSummarizer)
# ==============================================================================

OLLAMA_SUMMARIZER_PROMPTS = {
    "map": """Aufgabe: Du bist Content-Scout. Finde ALLES in diesem Text, was viral gehen könnte.
Regeln:
1. Format: `- [HH:MM:SS] Der Text`
2. WICHTIG: Sei nicht zu streng. Suche nach spannenden Sätzen, Humor oder steilen Thesen.
3. Zeitstempel immer exakt kopieren.
TEXT: {block}""",
    "reduce": """Wähle die absoluten Top {n} Highlights aus dieser Liste.
Achte auf Abwechslung (Humor vs. Tiefe).
Gib NUR eine Markdown-Liste zurück.
Format: `- [HH:MM:SS] Text`.
INPUT: {longlist}""",
    "chunk": """Fasse diesen Textabschnitt sachlich zusammen. 
Nutze STRIKTE Bulletpoints. Keine langen Schachtelsätze.
Abschnitt:\n{chunk}""",
    "merge": """Erstelle aus diesen Teil-Zusammenfassungen eine strukturierte Gesamt-Übersicht.
Nutze exakt diese Struktur:
1. Kernaussage (1 Satz)
2. Hauptpunkte (Bulletpoints)
3. Wichtigstes Takeaway

Textteile:\n{parts}"""
}

GEMINI_SUMMARIZER_PROMPTS = {
    # --- PHASE 1: SCOUTING (MAP) ---
    "scout_system": """Du bist ein analytischer Content-Scout für Social Media Reels. 
Deine Aufgabe ist es, ein Predigt-Transkript zu analysieren und virale Hooks zu extrahieren (tiefgründige, provokante oder humorvolle Sätze).

REGELN:
1. Die Sätze müssen komplett für sich alleine stehen können.
2. Entferne Füllwörter am Satzanfang (Und, Aber, Denn).
3. Behalte den exakten Zeitstempel des Satzanfangs bei.
4. Wähle nur Sätze, die eine starke emotionale oder intellektuelle Reaktion auslösen.""",
    
    "scout_user": "Hier ist das Transkript:\n{chunk}",

    # --- PHASE 2: EDITING (REDUCE) ---
    "editor_system": """Du bist der finale Redakteur für virale Video-Highlights.
Deine Aufgabe ist es, aus einer großen Liste von Kandidaten die stärksten Zitate auszuwählen.

REGELN:
1. Achte auf eine exzellente Mischung aus Inspiration, Provokation und Alltagsrelevanz.
2. Verändere den Wortlaut der Zitate nicht.
3. Wähle EXAKT die angeforderte Anzahl an Highlights aus.""",

    "editor_user": "Wähle exakt {n} Highlights aus dieser Liste aus:\n{longlist}",

    # --- SUMMARY ---
    "chunk": """Fasse diesen Textabschnitt präzise und elegant zusammen.
Schreibe einen fließenden, gut lesbaren Text (keine stumpfen Bulletpoints), der den gedanklichen Bogen des Sprechers gut einfängt.
Abschnitt:\n{chunk}""",
    "merge": """Führe diese Teil-Zusammenfassungen zu einer hochwertigen, zusammenhängenden Predigt-Zusammenfassung zusammen.
Gliedere den Text in sinnvolle, thematische Absätze und schreibe auf hohem sprachlichen Niveau. Keine Aufzählungszeichen, sondern schöner Fließtext.
Textteile:\n{parts}"""
}

# ==============================================================================
# 2. TITLE GENERATION (build_highlight_spans)
# ==============================================================================

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

# ==============================================================================
# 3. SOCIAL MEDIA CAPTIONS (Instagram)
# ==============================================================================

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

VORGABEN FOR DEN STIL:
- Keine abgedroschenen Begrüßungen ("Hallo Gemeinde"). Beginne direkt mit einer starken Aussage oder Frage, die sich auf das Video bezieht.
- Sei warm, echt und einladend. 
- Präsentiere die Fakten für den Sonntag am Ende sehr übersichtlich.

GIB GENAU DIESES FORMAT AUS:
HEADLINE: [Stimmungsvolle Headline]
BODY: [Dein Text, die Einladung, die Fakten und Hashtags]
"""
