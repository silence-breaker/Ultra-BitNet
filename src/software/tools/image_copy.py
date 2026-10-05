"""Bounded binary scatter operations for the five-bank image builder."""
from pathlib import Path
from typing import Any, BinaryIO, Mapping
import os
COPY_CHUNK = 4 * 1024 * 1024
class ImageBuildError(RuntimeError):
    pass

def _checked_source(package_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path.replace("/", os.sep))
    if relative.is_absolute() or ".." in relative.parts:
        raise ImageBuildError("unsafe source path: %s" % relative_path)
    root = package_root.resolve()
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ImageBuildError(
            "source escapes package root: %s" % relative_path) from exc
    if not source.is_file():
        raise ImageBuildError("source payload is missing: %s" % source)
    return source


def _copy_range(source: BinaryIO, destination: BinaryIO, source_offset: int,
                destination_offset: int, byte_count: int) -> None:
    if min(source_offset, destination_offset) < 0 or byte_count <= 0:
        raise ImageBuildError("invalid contiguous copy range")
    source.seek(source_offset)
    destination.seek(destination_offset)
    remaining = byte_count
    while remaining:
        data = source.read(min(remaining, COPY_CHUNK))
        if not data:
            raise ImageBuildError("source ended inside a planned copy")
        destination.write(data)
        remaining -= len(data)


def _copy_strided(source: BinaryIO, destination: BinaryIO,
                  source_offset: int, destination_offset: int,
                  records: int, source_stride: int,
                  bytes_per_record: int, destination_stride: int) -> None:
    if records <= 0 or bytes_per_record <= 0:
        raise ImageBuildError("strided copy must contain data")
    if source_stride < bytes_per_record or destination_stride < bytes_per_record:
        raise ImageBuildError("strided copy records overlap")
    for record in range(records):
        _copy_range(
            source, destination,
            source_offset + record * source_stride,
            destination_offset + record * destination_stride,
            bytes_per_record)


def materialize_bank(package_root: Path, bank: Mapping[str, Any],
                     output: Path) -> None:
    used_bytes = int(bank["used_bytes"])
    if used_bytes <= 0 or used_bytes > int(bank["capacity_bytes"]):
        raise ImageBuildError("invalid used byte count for bank %s" % bank["bank"])
    with output.open("w+b") as destination:
        destination.truncate(used_bytes)
        for obj in bank["objects"]:
            base = int(obj["local_offset"])
            end = int(obj["end_offset_exclusive"])
            if base < 0 or end > used_bytes or end - base != int(obj["bytes"]):
                raise ImageBuildError("invalid object range: %s" % obj["name"])

            if obj.get("zero_initialize"):
                # A newly created, truncated file reads as zero.  Leaving this
                # range sparse avoids writing the initial KV cache pointlessly.
                continue
            if "source_strided" in obj:
                record = obj["source_strided"]
                with _checked_source(package_root, record["relative_path"]).open("rb") as source:
                    _copy_strided(
                        source, destination,
                        int(record["source_offset"]),
                        base + int(record["destination_offset"]),
                        int(record["records"]), int(record["source_stride"]),
                        int(record["bytes_per_record"]),
                        int(record["destination_stride"]))
                continue
            if "source_interleaved" in obj:
                record = obj["source_interleaved"]
                common = {
                    "source_offset": int(record["source_offset"]),
                    "records": int(record["records_per_source"]),
                    "source_stride": int(record["source_stride"]),
                    "bytes_per_record": int(record["bytes_per_record"]),
                    "destination_stride": int(record["destination_stride"]),
                }
                if (common["destination_stride"] !=
                        2 * common["bytes_per_record"] or
                        int(record["up_destination_offset"]) !=
                        common["bytes_per_record"]):
                    raise ImageBuildError(
                        "Gate/Up interleave is not a dense alternating layout")
                for role, destination_key in (
                        ("gate", "destination_offset"),
                        ("up", "up_destination_offset")):
                    relative_key = role + "_relative_path"
                    with _checked_source(
                            package_root, record[relative_key]).open("rb") as source:
                        _copy_strided(
                            source, destination,
                            common["source_offset"],
                            base + int(record[destination_key]),
                            common["records"], common["source_stride"],
                            common["bytes_per_record"],
                            common["destination_stride"])
                continue
            if "source" in obj:
                record = obj["source"]
                with _checked_source(package_root, record["relative_path"]).open("rb") as source:
                    _copy_range(
                        source, destination, int(record["source_offset"]),
                        base + int(record["destination_offset"]),
                        int(record["bytes"]))
                continue
            if "source_stripes" in obj:
                for record in obj["source_stripes"]:
                    with _checked_source(package_root, record["relative_path"]).open("rb") as source:
                        _copy_strided(
                            source, destination,
                            int(record["source_offset"]),
                            base + int(record["destination_offset"]),
                            int(record["rows"]), int(record["source_row_stride"]),
                            int(record["bytes_per_row"]),
                            int(record["destination_row_stride"]))
                continue
            if "source_pieces" in obj:
                for record in obj["source_pieces"]:
                    with _checked_source(package_root, record["relative_path"]).open("rb") as source:
                        _copy_range(
                            source, destination, int(record["source_offset"]),
                            base + int(record["destination_offset"]),
                            int(record["bytes"]))
                continue
            raise ImageBuildError("object has no materialisation rule: %s" % obj["name"])

        destination.flush()
        os.fsync(destination.fileno())
