"""Site configuration for the ReState artifact. Every value can be overridden with an environment variable."""
import os

ROOT = os.environ.get("RESTATE_ROOT", "/opt/restate_tmc")
PYTHON = os.environ.get("RESTATE_PYTHON", ROOT + "/env/bin/python")
MODEL = os.environ.get("RESTATE_MODEL", ROOT + "/models/Qwen2.5-14B/97e1e76335b7017d8f67c08a19d103c0504298c9")
TOOL_ROOT = os.environ.get("RESTATE_TC_ROOT", ROOT + "/tools/iproute2-5.5")   # tc/ip used for the RTT (netem) experiment
RESULTS = os.environ.get("RESTATE_RESULTS", ROOT + "/results")
PROBE_SRC = os.environ.get("RESTATE_PROBE_SRC", os.path.dirname(os.path.abspath(__file__)))
TARGET_IP = os.environ.get("RESTATE_TARGET_IP", "")                           # internal IP of the target (destination) host
HOLDER_IPS = [x for x in os.environ.get("RESTATE_HOLDER_IPS", "").split(",") if x]   # internal IPs of holder 0 and holder 1
ROUTE_PROBE_IP = os.environ.get("RESTATE_ROUTE_PROBE_IP", "10.0.0.1")         # any address routed via the experiment NIC
MODEL_REVISION = os.environ.get("RESTATE_MODEL_REVISION", "97e1e76335b7017d8f67c08a19d103c0504298c9")
MODEL_MANIFEST_SHA256 = os.environ.get("RESTATE_MODEL_MANIFEST_SHA256", "")   # sha256 of <model>/e0_model_manifest.json
MODEL_ID = os.environ.get("RESTATE_MODEL_ID", "Qwen/Qwen2.5-14B")
