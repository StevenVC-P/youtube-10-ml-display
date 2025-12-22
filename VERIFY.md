Retro ML Desktop – lineage + video provenance smoke checks

1) DB lineage fields
- `sqlite3 ml_experiments.db "SELECT run_id, base_run_id, branch_id, parent_run_id, start_timestep, target_timestep, config_path FROM experiment_runs ORDER BY start_time DESC LIMIT 5;"` – confirm lineage fields are populated for new legs/branches.
- `sqlite3 ml_experiments.db "PRAGMA table_info(experiment_runs);"` – verify added columns (branch_id, parent_* , config_hash/path, start/target_timestep, metadata_json).

2) Start → continue → branch
- Launch UI: `python tools/retro_ml_desktop/main_simple.py`.
- Start a short run (e.g., Breakout, PPO, 0.02h) → confirm `models/checkpoints/<run_id>/run_metadata.json` has base_run_id=run_id, leg_index=0, branch_id=main, config_hash, target_timestep.
- Continue the run (Continue mode) → verify new run_id, same base_run_id/branch_id, leg_index incremented, start_timestep≈prior end, target_timestep≈prior end + leg span.
- Branch from same checkpoint (Branch mode) with a new branch label → verify run_metadata shows new branch_id, parent_run_id, hyperparam_diff, and base_run_id preserved.

3) Video generation + manifests
- Generate 30–60s video for latest leg: `python training/post_training_video_generator.py --model-dir models/checkpoints/<run_id>/milestones --config models/checkpoints/<run_id>/config_effective.yaml --total-seconds 60 --output-dir outputs/<run_id>/milestones`.
- Check filename schema: `<env>__<algo>__<runName>__base-<base>__leg-<idx>__branch-<branch>__ckpt-<id>__mode-eval__seed-<seed>__ts-...`. Confirm adjacent manifest: `<video>.manifest.json` with checkpoint/config hashes, base/leg/branch, seeds, mode, git commit.

4) Overlay fields (visual check on generated video)
- Top overlay shows run name + run_id, base_run_id, leg_index, branch_id, checkpoint id, mode, seed, git short hash.
- Second line shows algo + key hparams (lr, gamma, clip, ent), epsilon/mode, step/target, current reward.
- Segment banner at video start: `SEG <idx> | seed=<seed> | mode=eval`.

5) Smoke test loop
- Start a quick run (0.02h), wait for checkpoints.
- Continue it once; confirm dashboard progress starts near prior completion (e.g., ~50% if doubling target).
- Generate video; confirm manifest present and overlay fields populated.

Artifacts to inspect
- `models/checkpoints/<run_id>/run_metadata.json` – lineage + hashes.
- `models/checkpoints/<run_id>/config_effective.yaml` (hash in metadata).
- Video + `<video>.manifest.json` in output_dir.
