#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path


def clean(value: object, limit: int) -> bytes:
    text = str(value or "").replace('"', "'").replace("\\", "/")
    text = " ".join(text.split())
    raw = text.encode("ascii", "replace")[: limit - 1]
    return raw + b"\0" * (limit - len(raw))


def event_description(event: dict, classes: str) -> str:
    """Build an ASCII-safe semantic description for the bare-metal mailbox.

    The Fudan report description is UTF-8 Chinese. The constrained decoder
    mailbox is currently ASCII, so passing that text through ``replace`` used
    to turn every Chinese byte into question marks. Preserve the event meaning
    using the already validated event type and participant classes instead.
    """
    if not bool(event.get("has_event")):
        return "No abnormal event detected"
    event_type = str(event.get("event_type") or "traffic event").strip()
    participants = [str(item).strip() for item in event.get("participant_classes", []) if str(item).strip()]
    if event_type == "collision" and len(participants) >= 2:
        first, second = participants[0].lower(), participants[1].lower()
        if first == second:
            plural = {"person": "people", "bus": "buses"}.get(first, first + "s")
            return f"Two {plural} collision detected."
        return f"{first.capitalize()} and {second} collision detected."
    return f"{event_type.capitalize()} involving {classes} detected."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    event = report["event"]
    summary = str(report["vehicle_summary"])
    vehicle_count = int(summary.split()[0])
    classes = ", ".join(map(str, event.get("participant_classes", []))) or "participants"
    description = event_description(event, classes)
    payload = struct.pack(
        "<4I32s16s32s64s128s",
        0x31534342,
        1,
        vehicle_count,
        int(bool(event["has_event"])),
        clean(report["timestamp"], 32),
        clean("scene", 16),
        clean(event.get("event_type", "event"), 32),
        clean(classes, 64),
        clean(description, 128),
    )
    args.output.write_bytes(payload)
    print(f"SCHEMA_BYTES={len(payload)}")


if __name__ == "__main__":
    main()
