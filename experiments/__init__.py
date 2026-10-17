import sys as _sys

# Insights contain symbols such as "±" and "Δ"; a Windows console defaults to cp1252 and would crash on print.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
