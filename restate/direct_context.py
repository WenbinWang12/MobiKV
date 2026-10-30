"""Explicit direct-execution identity; never synthesizes scheduler/host identities.

Only E0--E4 are executable. Same-host and two-host observations remain separate
topology cohorts. Application pacing is not physical NIC/propagation emulation.
"""
from __future__ import annotations

import ctypes
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket


STAGES = {"E4"}
TOPOLOGIES = {"two_physical_hosts"}
PHYSICAL_KEYS = ("hostname", "boot_id", "dmi_uuid", "internal_ip")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing/symlinked direct artifact: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(directory):
    directory = Path(directory)
    paths = sorted({*directory.glob("*.py"), *directory.glob("*_protocol.json")})
    require(paths, "Empty direct runtime source closure")
    return {path.name: sha(path) for path in paths}


def gpu_uuid(value):
    require(isinstance(value, str), "GPU UUID must be a string")
    require(re.fullmatch(r"(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value),
            "Expected a complete physical CUDA GPU UUID")
    return "GPU-" + value.removeprefix("GPU-").lower()


def validate_context(context, plan, environment, hostname):
    """Pure validation used by workers and CPU tests; no scheduler is queried."""
    require(context.get("schema_version") == 1 and context.get("scheduler") == "direct", "Expected direct context v1")
    require(context.get("topology") in TOPOLOGIES, "Unsupported direct topology")
    require(plan.get("stage") in STAGES and context.get("stage") == plan["stage"], "This frozen runner supports E4 only")
    require(plan.get("stage") == "E4" and plan.get("mode") in ("e4_smoke", "e4_full", "e4_profile")
            and context.get("mode") == plan["mode"], "Full E4 direct mode differs")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", str(context.get("run_id", ""))), "Invalid direct run ID")
    require(isinstance(context.get("cluster_id"), str) and context["cluster_id"], "Missing direct cluster identity")
    require(type(context.get("world_size")) is int and context["world_size"] == 2
            and environment.get("WORLD_SIZE") == "2", "Exactly two direct ranks are required")
    require(environment.get("RANK") in ("0", "1"), "RANK must be zero or one")
    rank = int(environment["RANK"])
    ranks = context.get("ranks")
    require(isinstance(ranks, list) and len(ranks) == 2, "Two explicit rank identities are required")
    for index, row in enumerate(ranks):
        require(type(row.get("rank")) is int and row["rank"] == index, "Rank list must be ordered zero, one")
        require(all(isinstance(row.get(key), str) and row[key] for key in PHYSICAL_KEYS), "Incomplete physical host identity")
        for key in ("boot_id", "dmi_uuid"):
            require(re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", row[key]), "Malformed " + key)
        require(isinstance(ipaddress.ip_address(row["internal_ip"]), ipaddress.IPv4Address), "Direct sockets require an IPv4 endpoint")
        gpu_uuid(row.get("gpu_uuid"))
    require(context.get("hosts") == [row["hostname"] for row in ranks], "Host order must match rank identities")
    require(hostname == ranks[rank]["hostname"], "Worker hostname differs from its direct rank identity")
    require(gpu_uuid(ranks[0]["gpu_uuid"]) != gpu_uuid(ranks[1]["gpu_uuid"]), "Direct ranks cannot share a physical GPU")
    for key in PHYSICAL_KEYS:
        values = [row[key].lower() if key in ("boot_id", "dmi_uuid") else row[key] for row in ranks]
        if context["topology"] == "two_physical_hosts":
            require(values[0] != values[1], "Two-host context has identical physical " + key)
        else:
            require(values[0] == values[1], "Colocated context has inconsistent physical " + key)
    require(context.get("master_addr") == ranks[0]["internal_ip"] == environment.get("MASTER_ADDR"), "Master address differs")
    port = context.get("master_port")
    require(type(port) is int and 1024 <= port <= 65400 and environment.get("MASTER_PORT") == str(port), "Invalid/mismatched master port")
    require(port == 32080, "E4 services must match the frozen scoped-netem port allowlist")
    require(context.get("dtype") == "bfloat16", "Direct cohort preserves the existing BF16 numerical protocol")
    require(environment.get("CUDA_VISIBLE_DEVICES") == ranks[rank]["gpu_uuid"], "Visibility must explicitly name this rank's GPU UUID")
    return rank


def physical_identity(context, rank):
    expected = context["ranks"][rank]
    actual = {"hostname": socket.gethostname(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip().lower(),
              "dmi_uuid": Path("/sys/class/dmi/id/product_uuid").read_text().strip().lower(),
              "internal_ip": expected["internal_ip"]}
    for key in PHYSICAL_KEYS:
        value = expected[key].lower() if key in ("boot_id", "dmi_uuid") else expected[key]
        require(actual[key] == value, "Actual physical identity differs: " + key)
    # Binding an unconnected datagram socket proves local address ownership
    # without sending traffic or guessing from a remotely resolved hostname.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind((expected["internal_ip"], 0))
    return actual


def load_context(args, plan, runtime_directory):
    name = os.environ.get("RESTATE_EXECUTION_CONTEXT")
    require(name is not None and Path(name).is_absolute(), "RESTATE_EXECUTION_CONTEXT must be an absolute JSON path")
    path = Path(name)
    before = sha(path)
    require(path.stat().st_size <= 1024**2, "Unexpectedly large direct context")
    context = json.loads(path.read_text())
    require(sha(path) == before, "Direct context changed during read")
    rank = validate_context(context, plan, os.environ, socket.gethostname())
    require(not args.analysis_only and args.source_result is None, "E5 analysis is outside this direct runner")
    sources = source_hashes(runtime_directory)
    require(context.get("source_sha256") == sources, "Direct runtime source closure differs from frozen context")
    require(context.get("plan_sha256") == sha(args.plan), "Direct plan differs from frozen context")
    for field, value in (("model_path", args.model), ("model_id", args.model_id),
                         ("model_revision", args.revision), ("dtype", args.dtype)):
        require(context.get(field) == value, "Direct model/precision argument differs: " + field)
    require(Path(args.model).is_absolute() and re.fullmatch(r"[0-9a-f]{40}", args.revision or ""), "Pinned absolute model path/revision required")
    require(re.fullmatch(r"[0-9a-f]{64}", str(context.get("model_manifest_sha256"))), "Missing model manifest SHA256")
    actual = physical_identity(context, rank)
    return context, before, rank, actual


def selected_cuda_identity(torch, context, rank):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one UUID-selected CUDA GPU must be visible")
    torch.cuda.set_device(0)
    require(torch.cuda.is_bf16_supported(), "Selected GPU cannot run the frozen BF16 protocol")
    props = torch.cuda.get_device_properties(0)
    selected = getattr(props, "uuid", None)
    method = "torch.cuda device properties"
    if selected is None:
        class UUID(ctypes.Structure):
            _fields_ = [("bytes", ctypes.c_ubyte * 16)]
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuInit.argtypes, lib.cuInit.restype = [ctypes.c_uint], ctypes.c_int
        lib.cuDeviceGet.argtypes, lib.cuDeviceGet.restype = [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
        getter = getattr(lib, "cuDeviceGetUuid_v2", None) or lib.cuDeviceGetUuid
        getter.argtypes, getter.restype = [ctypes.POINTER(UUID), ctypes.c_int], ctypes.c_int
        device, uid = ctypes.c_int(), UUID()
        require(lib.cuInit(0) == 0 and lib.cuDeviceGet(ctypes.byref(device), 0) == 0, "CUDA UUID setup failed")
        require(getter(ctypes.byref(uid), device) == 0, "CUDA selected UUID query failed")
        value = bytes(uid.bytes).hex()
        selected = "GPU-" + "-".join((value[:8], value[8:12], value[12:16], value[16:20], value[20:]))
        method = "CUDA driver API selected visible ordinal"
    selected = gpu_uuid(str(selected))
    require(selected == gpu_uuid(context["ranks"][rank]["gpu_uuid"]), "Actual selected CUDA UUID differs from direct context")
    expected = context["ranks"][rank]
    require("gpu_name" not in expected or props.name == expected["gpu_name"], "Actual CUDA GPU name differs from inventory")
    if context["mode"] != "smoke":
        require(props.total_memory >= 79_000_000_000, "Main 14B cohort retains the validated 80GB-class memory requirement")
    return {"gpu_uuid": selected, "logical_ordinal": 0, "uuid_query_method": method,
            "gpu": props.name, "gpu_memory_bytes": props.total_memory,
            "device_count": torch.cuda.device_count(), "native_bfloat16": True}


def peer_hello(context, rank, identity, physical, selected_uuid):
    return {"kind": "campaign_identity", "identity": identity, "hostname": physical["hostname"],
            "rank": rank, "run_id": context["run_id"], "scheduler": "direct",
            "topology": context["topology"], "physical_identity": physical, "gpu_uuid": selected_uuid}


def validate_peer(peer, tensors, context, local_rank, identity):
    other = 1 - local_rank
    expected = context["ranks"][other]
    physical = {key: expected[key].lower() if key in ("boot_id", "dmi_uuid") else expected[key] for key in PHYSICAL_KEYS}
    require(not tensors and peer.get("kind") == "campaign_identity" and peer.get("identity") == identity,
            "Direct peer model/plan/source identity differs")
    require(type(peer.get("rank")) is int and peer["rank"] == other
            and peer.get("run_id") == context["run_id"] and peer.get("scheduler") == "direct"
            and peer.get("topology") == context["topology"] and peer.get("hostname") == expected["hostname"]
            and peer.get("physical_identity") == physical and peer.get("gpu_uuid") == gpu_uuid(expected["gpu_uuid"]),
            "Direct peer physical/GPU/run identity differs")
    if context["topology"] == "two_physical_hosts":
        require(peer["hostname"] != context["ranks"][local_rank]["hostname"], "Cross-host peer must reside on the other physical host")
    return {"passed": True, "rank": other, "hostname": peer["hostname"], "gpu_uuid": peer["gpu_uuid"],
            "physical_identity": physical, "scope": "Observed worker identity matched the explicit direct allocation context"}
