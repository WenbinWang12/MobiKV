"""Prefill/decode timing of the experiment model under an optional MPS SM cap; one process per setting.

Same load path as mh_runner (bf16, eager attention, cuda:0). The cap comes from the client environment
(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) and only applies when an MPS daemon is running. Prints one JSON line.
"""
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM


def main():
    model = AutoModelForCausalLM.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False,
                                                 dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval()
    ids = torch.randint(1000, 30000, (1, 4096), generator=torch.Generator().manual_seed(2026102903)).to("cuda:0")

    def prefill():
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model(ids, use_cache=True)
        torch.cuda.synchronize()
        return time.perf_counter() - start, out

    prefill()
    samples = [prefill()[0] for _ in range(3)]
    out = prefill()[1]
    kv, token, steps = out.past_key_values, ids[:, -1:], []
    with torch.no_grad():
        for _ in range(24):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(token, past_key_values=kv, use_cache=True)
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - start)
            kv = out.past_key_values
    print(json.dumps(dict(mps_percentage=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
                          prefill_4096_s=statistics.median(samples), prefill_samples_s=samples,
                          decode_step_s=statistics.median(steps[4:]), gpu=torch.cuda.get_device_name(0))))


if __name__ == "__main__":
    main()
