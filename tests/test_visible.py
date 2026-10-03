import sys
from memory.store import MemoryStore
from eval_harness.records import visible as harness_visible

def test_crosscheck_eval_harness():
    data_dir = "./data"
    store = MemoryStore(data_dir)
    
    times = [
        "2026-09-08T12:00:00-07:00",
        "2026-09-10T09:00:00-07:00",
        "2026-09-14T00:00:00-07:00",
        "2026-09-16T18:00:00-07:00",
        "2026-09-18T23:59:59-07:00"
    ]

    for t in times:
        from datetime import datetime
        as_of = datetime.fromisoformat(t)
        
        my_units = store.visible(as_of)
        my_ids = {u.id for u in my_units}
        
        harness_units = harness_visible(data_dir, as_of)
        harness_ids = {u.id for u in harness_units}
        
        diff = harness_ids - my_ids
        diff2 = my_ids - harness_ids
        
        print(f"\nTime {t}:")
        print(f"  My units: {len(my_ids)}")
        print(f"  Harness units: {len(harness_ids)}")
        
        # We expect my_ids to be slightly smaller because we don't return message_deleted events
        # Let's verify that the ONLY things in harness_ids not in my_ids are message_deleted events.
        if diff:
            print(f"  In harness but not mine: {len(diff)}")
            for d_id in diff:
                # Find it in harness units
                h_u = next(u for u in harness_units if u.id == d_id)
                print(f"    {d_id}: {h_u.text[:80]}")
        
        if diff2:
            print(f"  In mine but not harness: {len(diff2)}")
            for d_id in diff2:
                m_u = next(u for u in my_units if u.id == d_id)
                print(f"    {d_id}: {m_u.text[:80]}")

if __name__ == "__main__":
    test_crosscheck_eval_harness()
