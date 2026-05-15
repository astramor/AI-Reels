class LLMApiError(Exception):
    """Geworfen bei Fehlern in der Kommunikation mit der LLM-API (Ollama/Gemini)."""
    pass

class LLMInvalidJsonError(Exception):
    """Geworfen, wenn die LLM-Ausgabe kein valides JSON enthält."""
    pass
