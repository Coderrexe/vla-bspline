# Figure provenance

| figure | generator | source evidence |
|---|---|---|
| `headline_results.pdf` | `scripts/analysis/plot_headline_results.py` | `outputs/language_clause_exact/locked_3seed_factorial_2327539.json` plus `outputs/robocasa_confirm_n40/official_phase_50scene_stats.json` |
| `kettle_phase_storyboard.pdf` | `scripts/analysis/plot_kettle_video_forensics.py` | target-split seed-1001 videos from jobs 2327930 (original labels, failure) and 2327934 (official phases, success) |
| `robocasa_phase_progress_seed1000.pdf` | `scripts/analysis/plot_robocasa_phase_progress.py` | validated two-seed artifact; training seed 1000, n=50/task/condition |
| `robocasa_phase_progress_seed1001.pdf` | `scripts/analysis/plot_robocasa_phase_progress.py` | validated two-seed artifact; training seed 1001, n=20/task/condition on disjoint scenes |

The storyboard samples video frames at environment steps 0, 150, 200, 400, and
the final frame. Videos were recorded every five environment steps. The source videos
are the matched seed-1011 confirmation rollouts from jobs 2332999/2333000; their
SHA-256 values are `1be6d4c415795942cb42d74c74e62500df0ec245880bc0f65b5a20a4461f08d6`
(original) and `97aa38a375c1375dba5952be0ad5b35246d2655719ee897b39bf7a5fc926bed3`
(official phases). Both rollouts have initial-observation SHA-256
`3119615e98e76ba06b6ebf10c0ade7e076c09fcd06794b8884293eb40ecc033c`.

The progress figures use the immutable analysis snapshot
`robocasa_phase_replication_analysis_20260825_v2` and validated cluster artifact
`artifacts/robocasa_phase_replication/robocasa_phase_replication_stats_20260825_2347716.json`
(SHA-256 `13ae9ea02644728e66af4b59829432b2c59f23d74694d724abb457973581655c`).
The artifact binds all 12 input JSON hashes and reports all stage diagnostics, not
only favorable ones.
