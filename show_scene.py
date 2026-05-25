#!/usr/bin/env python3
"""Show the composition metadata embedded in a render_layers PNG.

The renderer (render_layers._save_png_with_metadata) writes the full
composition JSON into a PNG tEXt chunk under the key "composition".
This tool reads it back and prints a human-readable summary -- or the
raw JSON, or just the list of chunks present.

Examples:
  python3 show_scene.py images/redhead_layers_20260524_170547.png
  python3 show_scene.py images/foo.png --raw         # bare JSON
  python3 show_scene.py images/foo.png --keys        # list tEXt chunks
"""
import argparse
import json
import sys

from PIL import Image


def _print_summary(data, png_path):
    print(f"file:    {png_path}")
    print(f"scene:   {data.get('scene_name', '?')}")
    comp = data.get("composition", {})
    if "out" in comp:
        print(f"out:     {comp['out']}")
    bg = comp.get("background")
    if bg is not None:
        print(f"bg:      {[round(c, 3) for c in bg]}")

    layers = comp.get("layers", [])
    print(f"\nlayers ({len(layers)}):")
    skip = {"name", "type", "enabled", "alpha", "params"}
    for layer in layers:
        on = "on " if layer.get("enabled", True) else "off"
        name = layer.get("name", "?")
        type_ = layer.get("type", "?")
        alpha = layer.get("alpha", 1.0)
        print(f"  [{on}] {name}  type={type_}  alpha={alpha:.2f}")
        extras = {k: v for k, v in layer.items()
                  if k not in skip and not k.startswith("_ui_")}
        for k, v in extras.items():
            print(f"        {k:>22s} = {_fmt(v)}")
        params = layer.get("params") or {}
        if params:
            print(f"        params:")
            for k, v in params.items():
                print(f"        {k:>22s} = {_fmt(v)}")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return v


def main():
    p = argparse.ArgumentParser(
        description="Show scene/composition metadata embedded in a render PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no flags: human-readable summary. --raw: bare JSON. --keys: list tEXt chunks.")
    p.add_argument("png", help="path to a render_layers PNG")
    p.add_argument("--raw",  action="store_true", help="print bare JSON only")
    p.add_argument("--keys", action="store_true", help="list all tEXt chunk keys and bail")
    args = p.parse_args()

    try:
        img = Image.open(args.png)
    except FileNotFoundError:
        sys.exit(f"file not found: {args.png}")
    except Exception as exc:
        sys.exit(f"could not open {args.png}: {exc}")

    info = img.info or {}
    if args.keys:
        if not info:
            print("(no tEXt chunks)")
            return
        for k, v in info.items():
            n = len(str(v)) if isinstance(v, str) else "?"
            print(f"{k}  ({n} chars)")
        return

    raw = info.get("composition")
    if raw is None:
        present = ", ".join(info.keys()) if info else "<none>"
        sys.exit(f"no 'composition' chunk in {args.png}\n"
                 f"available chunks: {present}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"malformed JSON in 'composition' chunk: {exc}")

    if args.raw:
        print(json.dumps(data, indent=2))
    else:
        _print_summary(data, args.png)


if __name__ == "__main__":
    main()
