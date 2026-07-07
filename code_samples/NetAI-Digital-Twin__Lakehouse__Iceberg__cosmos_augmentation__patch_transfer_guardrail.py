"""Patch cosmos transfer.py to expose --disable_guardrail (CLI omits it though the
pipeline supports it). Lets us skip the guardrail, which otherwise needs the gated
Cosmos-Guardrail1 + Meta Llama-Guard-3-8B. Idempotent. Run on the cluster."""
p = "/scratch/autodr_test/cosmos-transfer1/cosmos_transfer1/diffusion/inference/transfer.py"
t = open(p).read()
if "--disable_guardrail" in t:
    print("already patched")
else:
    anchor = '        help="Offload guardrail models after inference",\n    )\n'
    assert anchor in t, "argparse anchor not found"
    add = ('    parser.add_argument(\n'
           '        "--disable_guardrail",\n'
           '        action="store_true",\n'
           '        help="Disable guardrail safety checks (avoids gated guardrail weights)",\n'
           '    )\n')
    t = t.replace(anchor, anchor + add, 1)
    n = t.count("offload_guardrail_models=cfg.offload_guardrail_models,\n")
    t = t.replace("offload_guardrail_models=cfg.offload_guardrail_models,\n",
                  "offload_guardrail_models=cfg.offload_guardrail_models,\n"
                  "            disable_guardrail=cfg.disable_guardrail,\n")
    open(p, "w").write(t)
    print(f"patched: added --disable_guardrail + wired into {n} pipeline call(s)")
