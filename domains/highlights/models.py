from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from loguru import logger

class HighlightItem(BaseModel):
    title: str = Field(..., min_length=1, description="Kurzer Titel")
    start: float = Field(..., ge=0.0, description="Startzeit in Sekunden")
    end: float = Field(..., description="Endzeit in Sekunden")
    reason: str = Field(..., description="Warum dieser Clip stark ist")
    hook: Optional[str] = Field(None, description="Optionaler Einstiegssatz oder Hook")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_duration(self):
        if self.end <= self.start:
            raise ValueError(f"Ungültige Zeiten für '{self.title}': end ({self.end}) muss größer als start ({self.start}) sein.")
        
        duration = self.end - self.start
        # Ideal 30-60s, Toleranz 20-75s
        if duration < 20.0 or duration > 75.0:
            logger.warning(f"Clip '{self.title}' hat eine extreme Dauer von {duration:.1f}s (Toleranz: 20-75s). Wird geduldet.")
        return self

class HighlightPayload(BaseModel):
    highlights: List[HighlightItem] = Field(..., min_length=1)

    @model_validator(mode='after')
    def remove_overlaps_and_sort(self):
        # Nach Startzeit sortieren
        sorted_hl = sorted(self.highlights, key=lambda x: x.start)
        filtered = []
        for hl in sorted_hl:
            # Überschneidungs-Logik: Falls Startzeit des neuen Clips vor (Endzeit - 10s) des alten liegt
            if filtered and hl.start < (filtered[-1].end - 10.0):
                logger.warning(f"Überschneidung erkannt zwischen '{filtered[-1].title}' und '{hl.title}'.")
                # Wir behalten den Clip mit der höheren Confidence
                conf_new = hl.confidence or 0.0
                conf_old = filtered[-1].confidence or 0.0
                if conf_new > conf_old:
                    logger.info(f"Ersetze durch '{hl.title}' (höhere Confidence).")
                    filtered[-1] = hl
                else:
                    logger.info(f"Verwerfe '{hl.title}'.")
            else:
                filtered.append(hl)
        self.highlights = filtered
        return self
