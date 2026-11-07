from mobikv import simulator


def test_simulator_exposes_final_switches():
    source = open(simulator.__file__, encoding="utf-8").read()
    for flag in ("bg_dfb", "link_het", "helper_fail", "bg_slow", "pp_ctrl"):
        assert flag in source
