class HighlightValidationError(Exception):
    """Geworfen, wenn das validierte Highlight-JSON nicht dem Schema entspricht."""
    pass

class EmptyTranscriptError(Exception):
    """Geworfen, wenn das übergebene Transkript leer ist."""
    pass

class NoValidHighlightsError(Exception):
    """Geworfen, wenn keine gültigen Highlights extrahiert werden konnten."""
    pass
