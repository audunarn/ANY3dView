"""Byte-exact patch helper for CRLF sources (worker tool, not product code)."""
import io, sys

def patch(path, replacements):
    with io.open(path, "r", newline="") as handle:
        text = handle.read()
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit(f"MATCH FAILURE ({text.count(old)} matches) for: {old[:120]!r}")
        text = text.replace(old, new)
    with io.open(path, "w", newline="") as handle:
        handle.write(text)
    print("patched", path)

if __name__ == "__main__":
    pass
