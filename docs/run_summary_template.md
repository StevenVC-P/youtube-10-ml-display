# Run Summary Template

Ready-to-fill structure for summarizing training runs using `ml_experiments.db` + stored artifacts. Replace `{{placeholders}}` via a small script or manual fill. All items are derived from existing tables in `tools/retro_ml_desktop` data flow.

## 1) Intro / Snapshot
- Run: `{{run_id}}` — `{{experiment_name}}` (`{{custom_name_or_leg}}`)
- Game / Algo / Preset: `{{game}}` / `{{algorithm}}` / `{{preset}}`
- Status: `{{status}}` (started: `{{start_time}}`, ended: `{{end_time_or_in_progress}}`, duration: `{{duration_h}}` h)
- Progress: `{{current_timestep}} / {{total_timesteps}}` (`{{progress_pct}}%`)
- Outcome headline: Best reward `{{best_reward}}`, final `{{final_reward}}`, convergence at `{{convergence_timestep}}`

## 2) Outcomes
- Milestones: 10% `{{rew_10}}`, 25% `{{rew_25}}`, 50% `{{rew_50}}`, 75% `{{rew_75}}`, 100% `{{rew_100}}`
- Plateau / improvement callout: `{{milestone_trend_note}}`
- Solved/threshold hit: `{{solved_flag}}`

## 3) Training Efficiency
- Avg/Min/Max FPS: `{{avg_fps}} / {{min_fps}} / {{max_fps}}`
- Steps per second trend: `{{sps_trend_note}}`
- ETA trend: `{{eta_note}}`
- Downtime: pauses/restarts `{{pause_count}}` (lost time `{{pause_time_h}}` h)

## 4) Resource Profile
- GPU util avg/peak: `{{gpu_avg}}% / {{gpu_max}}%`, VRAM avg/peak: `{{vram_avg}} / {{vram_max}} MB`
- CPU avg/peak: `{{cpu_avg}}% / {{cpu_max}}%`, RAM avg/peak: `{{mem_avg}} / {{mem_max}} MB`
- Bottlenecks: `{{resource_bottleneck_note}}`

## 5) Learning Dynamics & Stability
- Losses: policy `{{policy_loss_trend}}`, value `{{value_loss_trend}}`, entropy `{{entropy_loss_trend}}`
- KL / explained_variance: `{{kl_note}}`, `{{explained_variance_note}}`
- Reward shape: `{{reward_trend_note}}`
- Anomalies: backward timestep jumps `{{backward_jump_count}}`, gaps `{{metric_gap_note}}`, cliffs `{{reward_cliff_note}}`
- Failures: `{{error_summary_if_failed}}`

## 6) Config Highlights
- Timesteps: target `{{total_timesteps}}`; eval/save freq: `{{eval_freq}} / {{save_freq}}`
- Parallelism: envs `{{n_envs}}`, frame_stack `{{frame_stack}}`, action_repeat `{{action_repeat}}`
- LR/Batch/N-steps: `{{learning_rate}} / {{batch_size}} / {{n_steps}}`; gamma `{{gamma}}`
- Device/seed: `{{device}} / {{seed}}`
- Notable overrides/custom params: `{{custom_params_note}}`

## 7) Artifacts
- Model: `{{model_path}}`
- Logs/TensorBoard: `{{log_path}}`, `{{tensorboard_path}}`
- Videos: post-training `{{video_artifacts_paths}}`; training segments `{{training_videos_summary}}`
- Video generation status: `{{video_generation_status}}`

## 8) Lineage / Comparison
- Base run / leg: `{{base_run_id}}` (leg `{{leg_number}}`)
- Changes vs lineage: `{{lineage_diff_note}}` (from `lineage_json`)
- Delta vs prior: reward `{{reward_delta}}`, FPS `{{fps_delta}}`, config deltas `{{config_delta_list}}`

## 9) Next Actions
- Gate: `{{go_no_go}}`
- Recommendations (pick top 2): `{{recommendations}}`

---

## Useful queries (SQLite)

Run info:
```sql
SELECT run_id, experiment_name, status, start_time, end_time, current_timestep,
       best_reward, final_reward, convergence_timestep, config_json
FROM experiment_runs
WHERE run_id = :run_id;
```

Latest experiment snapshot:
```sql
SELECT game, algorithm, preset, progress_pct, current_timestep,
       elapsed_time, estimated_time_remaining, latest_metrics_json, lineage_json
FROM experiments
WHERE experiment_id = (SELECT base_run_id FROM experiment_runs WHERE run_id = :run_id LIMIT 1)
   OR experiment_id = :run_id;
```

Metrics stats:
```sql
SELECT COUNT(*) AS total_metrics,
       MIN(timestep) AS first_step,
       MAX(timestep) AS last_step,
       AVG(fps) AS avg_fps, MIN(fps) AS min_fps, MAX(fps) AS max_fps,
       AVG(gpu_percent) AS avg_gpu, MAX(gpu_percent) AS max_gpu,
       AVG(gpu_memory_mb) AS avg_vram, MAX(gpu_memory_mb) AS max_vram,
       AVG(cpu_percent) AS avg_cpu, MAX(cpu_percent) AS max_cpu,
       AVG(memory_mb) AS avg_mem, MAX(memory_mb) AS max_mem,
       MIN(timestamp) AS first_time, MAX(timestamp) AS last_time
FROM training_metrics
WHERE run_id = :run_id;
```

Milestone rewards (adjust steps):
```sql
SELECT timestep, episode_reward_mean
FROM training_metrics
WHERE run_id = :run_id AND episode_reward_mean IS NOT NULL
  AND (progress_pct IN (10,25,50,75,100) OR timestep IN (
      (SELECT MAX(timestep) FROM training_metrics WHERE run_id=:run_id AND progress_pct<=10),
      (SELECT MAX(timestep) FROM training_metrics WHERE run_id=:run_id AND progress_pct<=25),
      (SELECT MAX(timestep) FROM training_metrics WHERE run_id=:run_id AND progress_pct<=50),
      (SELECT MAX(timestep) FROM training_metrics WHERE run_id=:run_id AND progress_pct<=75),
      (SELECT MAX(timestep) FROM training_metrics WHERE run_id=:run_id AND progress_pct<=100)
  ))
ORDER BY timestep;
```

Backward timestep jumps:
```sql
SELECT m1.id AS id1, m1.timestep AS t1, m2.id AS id2, m2.timestep AS t2
FROM training_metrics m1
JOIN training_metrics m2 ON m2.id = m1.id + 1
WHERE m1.run_id = :run_id AND m2.run_id = :run_id AND m2.timestep < m1.timestep;
```

Videos:
```sql
SELECT * FROM video_artifacts WHERE experiment_id = :experiment_id;
SELECT * FROM training_videos WHERE run_id = :run_id ORDER BY segment_number;
SELECT * FROM video_generation_progress WHERE run_id = :run_id;
```

Lineage diff idea:
- Load `lineage_json` and `config_json`; list keys that differ and the before/after values.

Error summary idea:
- Tail `log_path` and scan for CUDA/OOM keywords; include `process_manager` `error_message` if status = failed.

Use this file as a reference for a future automated generator (LLM-assisted or scripted) without needing new schema changes.
