from google import genai
import yaml

# 1. API Key laden
api_key = ""
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
        api_key = conf.get("gemini_api_key", "")
except:
    pass

if not api_key:
    print("❌ Kein API Key gefunden. Bitte trage ihn direkt im Skript ein.")
    exit()

print(f"--- Prüfe Modelle für Key: {api_key[:10]}... ---")

try:
    client = genai.Client(api_key=api_key)

    print("\nGefundene Modelle:")
    print("-" * 60)

    # Wir iterieren einfach über alle und geben den Namen aus
    # Das neue SDK liefert oft Objekte zurück, die man direkt printen kann
    for model in client.models.list():
        # Wir versuchen, den Namen sicher zu extrahieren
        name = getattr(model, "name", str(model))
        display_name = getattr(model, "display_name", "")

        # Nur relevante Modelle anzeigen (Flash/Pro)
        if "gemini" in str(name).lower():
            print(f"✅ ID: {name}")
            if display_name:
                print(f"   Name: {display_name}")
            print("-" * 20)

except Exception as e:
    print(f"\n❌ Fehler: {e}")
