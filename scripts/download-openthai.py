"""Provision the pinned OpenThai snapshot without executing downloaded code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "inference" / "openthai" / "model-manifest.json"


def provision(output, manifest):
    output = output.resolve()
    contents = {}
    for name, expected in manifest["sha256"].items():
        target = (output / name).resolve()
        if not target.is_relative_to(output):
            raise ValueError("Invalid manifest path: " + name)
        if target.exists():
            content = target.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Existing file does not match pinned snapshot: " + name)
        else:
            url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Downloaded checksum mismatch: " + name)
        contents[name] = content
    # Verify the whole snapshot before writing any model/runtime files.
    for name, content in contents.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (output / "provisioned.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(contents)} files for OpenThai {manifest['version']} at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "weights" / "openthai2.1")
    args = parser.parse_args()
    provision(args.output, json.loads(MANIFEST.read_text(encoding="utf-8")))
