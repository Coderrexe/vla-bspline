# RoboCasa counterfactual-steering artifact transfer

The existing 600/1,500-episode slot-decorrelation result is scientifically useful,
but it currently exists only in the team report. Please transfer the following as a
single immutable directory (tarball or shared project path); do not regenerate it.

## Required artifacts

- The complete 600- and 1,500-episode LeRobot datasets, including `meta/`, data
  Parquets, videos, episode/task tables, and the exact language strings.
- The counterfactual RoboCasa environment patch and scripted demonstration
  controller, plus their source commit or SHA-256 manifest.
- The original/confounded, 600-episode, 1,500@35k, and 1,500@70k checkpoints,
  including `model.safetensors`, policy config, train config, optimizer/training
  state, and logs.
- The evaluation script and the exact scene/slot seed list used for every reported
  cell; include raw per-episode outcomes rather than only aggregate percentages.
- Package/environment versions and the resume-training patch that skips the
  inappropriate `relative_actions_processor` override for the spline policy.

## Verification performed after transfer

1. Hash every source, dataset table/video, checkpoint, and raw result before use.
2. Verify that target slot and language are balanced and that the physical action
   data are identical wherever the comparison claims only a language change.
3. Re-score the existing checkpoints on at least 100 locked scenes per slot using
   commanded-object first grasp-and-lift (sustained lift ≥3 cm for five steps before
   any distractor grasp) as primary; approach distance remains diagnostic.
4. Train waypoint A and corrected-contract spline C on the exact same 1,500 episodes,
   seeds 1000–1002, updates, sampling, and evaluation scenes. Use matched deployment
   cadence and report both slot-specific competence and their balanced average.
5. Treat a smaller slot gap caused by both slots failing as a regression, not better
   grounding. Report no-grasp, wrong-first-grasp, and correct-first-grasp coverage.

This matched A/C factorial is the shortest route from the existing team result to a
paper-grade claim that the compact spline action representation improves language
grounding rather than merely benefiting from counterfactual data collection.
