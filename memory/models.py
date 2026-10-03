from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class Unit:
    id: str
    record_id: str
    source: str
    time: datetime
    text: str
    title: Optional[str] = None
    speaker: Optional[str] = None
    speaker_label: Optional[str] = None
    speaker_confidence: Optional[float] = None
    speaker_known: bool = False
    thread_key: Optional[str] = None
    seq: Optional[int] = None
    edited: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
