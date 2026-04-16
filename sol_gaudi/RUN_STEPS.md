# Run BFCL on ASU SOL Gaudi — step by step

Quick reference. For full operational detail see `README.md`; for hardware reference see `GAUDI_EVAL_GUIDE.md`.

```bash
# 1. SSH to SOL (requires ASU campus network or Sol VPN; first login prompts
#    for host-key acceptance — type "yes")
ssh <asurite>@login.sol.rc.asu.edu

# 2. Clone + switch to the Gaudi branch (on /scratch, not $HOME)
cd /scratch/$USER
git clone https://github.com/HectorHernandez1/gorilla_bfcl.git
cd gorilla_bfcl
git checkout bfcl_gaudi

# 3. (Optional) Override defaults for your account/QoS/SIF path/email
#    Skip this if you're on the class account with default settings —
#    scripts fall back to config.env.example automatically.
# cp sol_gaudi/config.env.example sol_gaudi/config.env
# $EDITOR sol_gaudi/config.env

# 4. Bring-up + smoke test (qwen3_4b on simple_python)
./sol_gaudi/quickstart.sh
# builds bfcl_gaudi mamba env, resolves vllm_gaudi.sif, submits 1 job

# 5. Watch it
./sol_gaudi/manage_bfcl_gaudi.sh status
./sol_gaudi/manage_bfcl_gaudi.sh logs <JOB_ID>

# 6. Once smoke passes, launch the full sweep
./sol_gaudi/manage_bfcl_gaudi.sh submit-all
# or one at a time: submit qwen3_32b / submit gemma4_31b

# 7. Read results
./sol_gaudi/manage_bfcl_gaudi.sh results
# or inspect berkeley-function-call-leaderboard/score/data_overall.csv
```

Model keys: `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `qwen3_32b`, `gemma4_31b`.

Always run the `qwen3_4b` smoke test before `submit-all` — the ASU workshop deck only validated Qwen 2.5 on HPU-vLLM, so Qwen3 support is inferred from architecture, not confirmed.
