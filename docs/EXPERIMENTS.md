# Experiment map

| Paper component | Public entry point |
| --- | --- |
| Full/cached-prefix recompute, transfer, hybrid repair | `mobikv/measurement/ct_hybrid.py` |
| RPP with one helper | `mobikv/measurement/ct_pp.py` |
| RPP with two helpers | `mobikv/measurement/ct_pp2.py` |
| Trace-driven MobiKV, published baselines, RPP and DF | `mobikv/simulator.py` |
| Main/robustness aggregation | `mobikv/analysis/analyze_a9.py` |
| Real-GPU SV-3 replay | `mobikv/measurement/sv3_replay.py` and `mobikv/analysis/analyze_sv3.py` |

The final system keeps the mechanism flags explicit:

```text
--policy mobikv --route_first p2c --cont_min 0 --beta 1
--pp 1 --pp_cold 1 --pp_ctrl dpp --pp_V 30 --pp_eps 0.05
--pp_min_rel 0.25 --pp_qmin 1024 --pp_grid 32 --bg_dfb 1
```

The six adapted published baselines are Follow-Me Cloud (Taleb and Ksentini, IEEE Network 2013), MDP-based service migration (Wang et al., IEEE/ACM ToN 2019), Lyapunov-based service placement (Ouyang et al., IEEE JSAC 2018), Preble (Srivatsa et al., ICLR 2025), Mooncake's scheduler (Qin et al., USENIX FAST 2025), and LoongServe-style ESP (Wu et al., ACM SOSP 2024). Full references are in the repository README. All policies use the same hybrid repair primitive when a KV range is missing.
