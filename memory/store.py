import bisect
from datetime import datetime
from typing import List, Dict

from memory.ingest import load_all
from memory.models import Unit

class MemoryStore:
    def __init__(self, data_dir: str):
        self.units, self.deletions, self.edits = load_all(data_dir)
        # Sort units by time for fast slice
        self.units.sort(key=lambda u: u.time)
        self.id_to_unit: Dict[str, Unit] = {u.id: u for u in self.units}

    def get(self, id: str) -> Unit:
        return self.id_to_unit.get(id)

    def visible(self, as_of: str | datetime) -> List[Unit]:
        if isinstance(as_of, str):
            as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))

        # Find the split point for time <= as_of
        # Since we just want all units <= as_of, we can use bisect
        # Create a dummy object to bisect on time
        class Dummy:
            def __init__(self, time): self.time = time
            def __lt__(self, other): return self.time < other.time

        # Extract times
        times = [u.time for u in self.units]
        idx = bisect.bisect_right(times, as_of)
        
        valid_units = self.units[:idx]

        out = []
        for u in valid_units:
            # Check deletion
            # 1. Never return message_deleted events. They have meta.deleted_target
            if "deleted_target" in u.meta:
                continue
            
            # 2. Remove Slack targets deleted at or before as_of
            if u.id in self.deletions and self.deletions[u.id] <= as_of:
                continue

            # 3. Remove edit-event units whose target is deleted by as_of
            if "edit_target" in u.meta:
                target_id = u.meta["edit_target"]
                if target_id in self.deletions and self.deletions[target_id] <= as_of:
                    continue

            # 4. Apply latest edit to the original message
            # If this is the original message, see if it has edits
            if u.id in self.edits:
                valid_edits = [text for t, text in self.edits[u.id] if t <= as_of]
                if valid_edits:
                    # Creating a shallow copy to return a mutated state
                    u_edited = Unit(
                        id=u.id, record_id=u.record_id, source=u.source, time=u.time,
                        text=valid_edits[-1], title=u.title, speaker=u.speaker,
                        speaker_label=u.speaker_label, speaker_confidence=u.speaker_confidence,
                        speaker_known=u.speaker_known, thread_key=u.thread_key,
                        seq=u.seq, edited=True, meta=u.meta.copy()
                    )
                    out.append(u_edited)
                    continue

            out.append(u)

        return out
