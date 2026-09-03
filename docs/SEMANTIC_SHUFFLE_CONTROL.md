# Semantic-label permutation control

Status (Aug. 21): the read-only Misha coverage audit completed as job 2319578;
the overlay and GPU trainings have not been launched. With permutation seed
1701, 966/975 used appended labels are deranged, covering 79,971/86,148
appended frames (92.83%) and 3,709/3,973 contiguous runs (93.36%). There are no
normalized-text collisions. Eighteen mapped pairs have token Jaccard at least
0.8. Exact SmolVLM tokenizer/truncation audit job 2319582 then confirmed that
all 966 mapped pairs have distinct token-id sequences and none are truncated at
the policy's 48-token limit (artifact SHA-256
`a08219c525c0e780ad05ce141696ee10bc0ccad07084711d7cc6197ace5fe0de`).
Coverage is at least 81.6% on every Long task except task 4, where
singleton/shared-support labels limit the intervention to 36.6%; task 0, the
historical positive mechanism cell, is 87.1% deranged. The immutable report is
`outputs/language_dataset_audit/semshuffle_analysis_s1701_2319578.json` on
Misha (SHA-256
`7a25341fed91355605c90a7568014ebda01a4bf6be83035955a1981397bb109f`).
Overlay build job 2319588 was submitted with the predeclared 90% overall
deranged-frame threshold; GPU training remains gated on the primary factorial.

## Why this is the next causal control

The running language factorial asks whether the historical granular-language
dataset helps each action head. It does not distinguish useful segment semantics
from regularization caused by changing prompts at the same temporal density. A
matched negative control should therefore keep every demonstration, task-index
switch, segment duration, and caption vocabulary fixed while breaking only the
caption-to-motion meaning.

`build_language_shuffle_control.py` does this without copying frame or video
data. It permutes appended caption strings in `meta/tasks.parquet`; all frame
`task_index` arrays remain in the original source through a symlink overlay.
The mapping is a deterministic seeded derangement within groups of caption IDs
that have the **exact same original-task support set**. Original task identity
comes from the episode `tasks` field, not the known-stale episode task-index
statistics.

This construction exactly preserves:

- frames, observations, actions, images, episode membership, and order;
- the full task-index schedule and every language-run boundary/length;
- the number and global vocabulary of appended captions;
- each original task's granular-caption vocabulary;
- each original task's and the global multisets of caption frame frequencies
  and run lengths.

It does not preserve which caption string receives which frequency—that is the
intended semantic intervention. It also cannot derange a used label whose exact
task-support class contains only that label. Such labels remain unchanged and
the analyzer reports their frame/run coverage, making the full-suite comparison
an intention-to-treat estimate diluted by those segments. Labels shared across
tasks are safe when there is another label with the identical support set; they
are never reassigned across support sets.

## Interpretation

For a fixed head, the estimand is:

> success with correctly aligned legacy granular captions minus success with
> same-support shuffled captions, with frames and prompt-switch timing fixed.

A positive difference is evidence that caption semantics—not merely prompt
switches, caption diversity, or temporal regularization—drive the benefit. It is
still not evidence of production-exact event alignment because `libero_l10gran`
uses legacy Molmo boundaries. B-spline specificity requires the full A/C cross:
both heads trained on aligned and shuffled language under the same seeds and
evaluated with the locked evaluator. A C-only contrast is insufficient.

The derangement is string-exact, not guaranteed meaning-exact: two different
captions can be paraphrases. The analyzer records normalized-text collisions and
token-Jaccard overlap for every mapping. The review should also pass each pair
through the exact frozen policy tokenizer and prompt template: distinct strings
could theoretically collapse after normalization or truncation. Review that
audit and pre-register a minimum deranged-frame coverage before allocating
GPUs. If singleton coverage is large, do not relax across support groups: that
would change per-task vocabulary and confound task identity with semantic
alignment. A comprehensive control would instead require rewriting frame task
IDs with task-specific label duplicates, losing the clean
zero-copy/task-table-only property.

## Safe execution proposal

Nothing in this proposal has been synced to or submitted on Misha. After review:

1. Run the CPU analyzer only and inspect coverage/collisions.
2. If coverage is adequate, build the compact overlay with
   `cluster/build_language_shuffle_control.sbatch`; it refuses overwrite and
   records hashes, the complete mapping, source paths, and source dependency.
   The proposal intentionally refuses to build unless a reviewed
   `MIN_DERANGED_FRACTION` is supplied.
3. Train the A/C × three-seed cross using
   `cluster/train_language_shuffle_control.sbatch`, gated on the reviewed
   builder. Evaluate with exactly the locked state/RNG/hardware/cadence protocol
   used by the aligned-language factorial.
4. Treat the overlay as dependent on the immutable source. The training script
   rechecks the derived task-table hash; the source schedule hash should also be
   regenerated before launch if the source dataset may have changed.

Because current scratch file-count headroom is tight, the overlay uses only
top-level/meta symlinks plus one rewritten parquet and one manifest. It should
not be uploaded as an independent dataset without materializing and re-auditing
all dependencies.

One fixed, pre-registered permutation across three training seeds estimates
training variance cleanly. A confirmatory second permutation is valuable if the
first contrast is large, because three optimizer seeds do not capture
permutation uncertainty; crossing every training seed with many permutations is
the statistically strongest but much more expensive design.
