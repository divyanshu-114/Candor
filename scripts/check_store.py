#!/usr/bin/env python3
import sys
from collections import Counter
from memory.store import MemoryStore

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    print("Loading MemoryStore...")
    store = MemoryStore(data_dir)
    
    # Total units per source
    total = Counter(u.source for u in store.units)
    print("\nTotal units per source:")
    for k, v in total.items():
        print(f"  {k}: {v}")

    # Check visible at 2026-09-09T09:00:00-07:00
    t1 = "2026-09-09T09:00:00-07:00"
    vis1 = store.visible(t1)
    c1 = Counter(u.source for u in vis1)
    print(f"\nVisible units at {t1}: {len(vis1)}")
    for k, v in c1.items():
        print(f"  {k}: {v}")

    # Check visible at 2026-09-18T18:00:00-07:00
    t2 = "2026-09-18T18:00:00-07:00"
    vis2 = store.visible(t2)
    c2 = Counter(u.source for u in vis2)
    print(f"\nVisible units at {t2}: {len(vis2)}")
    for k, v in c2.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
