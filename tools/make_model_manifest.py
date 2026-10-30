"""Write <model_dir>/e0_model_manifest.json (the format verify_manifest checks at every worker start) and print
its sha256, to be exported as RESTATE_MODEL_MANIFEST_SHA256. Usage: make_model_manifest.py MODEL_DIR REPO_ID REVISION"""
import hashlib, json, sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

model, repo, rev = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
files = [dict(name=p.name, size_bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(model.iterdir())
         if p.is_file() and p.name != "e0_model_manifest.json" and not p.name.startswith(".")]
out = model / "e0_model_manifest.json"
out.write_text(json.dumps(dict(repo_id=repo, revision=rev, files=files, complete=True), indent=1, sort_keys=True) + "
")
print(hashlib.sha256(out.read_bytes()).hexdigest())
