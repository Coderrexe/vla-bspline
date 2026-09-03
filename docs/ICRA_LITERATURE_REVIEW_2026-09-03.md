# ICRA 2027 literature and paper-structure review

*Verified 3 September 2026. This is a manuscript-design document, not the BibTeX
file. Every venue label below is linked to a primary source.*

## Submission constraints

The [official ICRA 2027 call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
sets the deadline at **15 September 2026, 11:59 PM Pacific** and permits **eight
pages total**, including figures, tables, acknowledgments, and references. Review is
double anonymous. There is no supplementary PDF: only an optional video, limited to
180 seconds and 20 MB. Video uploads are open through 9 September and again from
17--22 September; manuscript uploads remain available while video uploads are paused
from 10--16 September. The paper therefore has to be self-contained.

The same call requires disclosure in the acknowledgments when generative AI created
paper content, figures, or code; ordinary grammar editing is outside the intended
scope. We should keep a short internal log of what assistance was used and make the
final disclosure match the actual workflow.

## What accepted ICRA papers show about the bar

| paper | verified venue | evaluation shape | structural lesson |
|---|---|---|---|
| [hPGA-DP](https://arxiv.org/abs/2507.05695) | ICRA 2026 | five robosuite tasks plus two real tasks; a compact real-robot section | a focused hardware page can be sufficient when it tests the main claim rather than acting as a demo appendix |
| [Action Coherence Guidance](https://arxiv.org/abs/2510.22201) | ICRA 2026 | RoboCasa, DexMimicGen, and two SO-101 tasks | one clear failure mode, one clean intervention, and a compact cross-domain table can carry an eight-page paper |
| [Inference-Time Policy Steering](https://arxiv.org/abs/2411.16627) | ICRA 2025 | three simulated and real-world benchmarks and several steering interfaces | steering papers need an alignment metric or counterfactual intervention, not success alone |
| [Discrete Policy](https://arxiv.org/abs/2409.18707) | ICRA 2025 | simulation and multiple real embodiments | a representation paper succeeds when the representation is tied to a conspicuous capability and headline delta |
| [Flow Policy Optimization](https://arxiv.org/abs/2510.09976) | ICRA 2026 | LIBERO and simulated ALOHA, without a real-robot requirement in the abstract | hardware is valuable but not an absolute venue requirement; a coherent, strong simulation result can be accepted |

The supplied hPGA-DP paper is exactly eight pages. Its main body uses roughly one
page for the introduction, under one page for related work, about two pages for the
method, two pages for evaluation and discussion, and the remainder for references.
Its hardware evidence is deliberately narrow: two tasks with a table and qualitative
frames. ACG similarly uses two real tasks. These papers support a **focused
0.6--0.9-page hardware section**, not an attempt to build a large new benchmark in
twelve days.

The common successful pattern is:

1. State one failure of current policies in the first paragraph.
2. Name one mechanism that addresses it and show the mechanism in Figure 1.
3. Put the strongest quantitative result in the abstract and introduction.
4. Use the evaluation to answer two or three explicit questions, not to list every
   experiment chronologically.
5. Give hardware the same intervention and metric as simulation.

Our statistical depth is already above these examples: the primary language result
uses three training seeds, exact replay, 6,000 rollouts, and a hierarchical confidence
interval. The manuscript should exploit that credibility without spending a page on
experimental bookkeeping.

### Full-paper visual and hardware audit

The following conclusions come from inspecting the complete eight-page PDFs, not only
their abstracts.

| paper | page-one visual strategy | main-result presentation | hardware footprint |
|---|---|---|---|
| hPGA-DP | no teaser; the architecture appears on page 3 | one combined task-image/table/training-curve figure | approximately one page across pages 5--6; two tasks, 200 demonstrations each |
| ACG | polished half-page conceptual diagram plus a three-domain gain plot | one cross-domain table followed by targeted coherence and ablation figures | two tasks, 50/40 demonstrations, 10 trials repeated three times; hardware shares the main table |
| ITPS | large photographic montage that communicates all three steering interfaces | alignment-versus-validity tables plus trajectory filmstrips | two kitchen skills, 60 demonstrations each; roughly one page including the behavior graph and result table |
| FPO | no page-one figure; dense technical introduction | one clean four-suite table, learning curves, and latent/action diagnostics | none; accepted with simulation only |

There is no single cosmetic template. Clear papers range from conservative IEEE plots
to polished graphical abstracts. The strongest pattern for our kind of capability
paper is the ACG/ITPS pattern: **show the intervention and resulting behavior before
the reader reaches the equations**. PACE, while a concurrent preprint rather than an
accepted-paper calibration point, sets the best current visual bar: its full-width
first-page figure combines the fixed-horizon failure, method block, headline numbers,
and a phase-by-phase rollout. Our Figure 1 should target this level of information
design without copying its style.

The supplied hardware photograph is the same Yale dual-xArm7-on-linear-actuator
platform shown in hPGA-DP. Its 3D-printed non-cuboid stacking and drawer fixtures also
match that paper's established task infrastructure. This materially lowers setup risk:
we can reuse the calibrated platform and reliable physical interactions while asking
a different question about clause composition and timing. The accepted hPGA-DP paper
confirms that two well-chosen tasks and about one page of hardware are credible at
ICRA; ACG and ITPS show that 40--60 demonstrations per task and 10 paired trials per
run can also be publishable when the intervention is clear.

### Direct quality assessment of our package

Our current evidence is in the accepted-paper range before hardware:

- FPO's accepted simulation-only headline is 87.2% average LIBERO success and a
  5.1-point LIBERO-Long gain over its strongest reported comparison. Our primary
  representation-specific language interaction is **+9.33 points**, positive across
  three training seeds, layered on much larger within-head clause gains.
- ACG reports an average 9.6-point improvement across its combined domains with three
  repeats and 10 real trials per task. Our primary simulation result has substantially
  more evaluation depth and exact paired replay.
- ITPS succeeds through a clean behavioral intervention and an alignment/success
  decomposition. Our clause-order experiment provides the analogous causal evidence,
  but its final-conjunction drop means the paper must distinguish first-subgoal
  steering from full reversed-order completion.
- hPGA-DP demonstrates that the exact available hardware platform and 3D-printed
  fixtures can support an accepted two-task, one-page physical evaluation.

The package is therefore not held back by the quantity of results. Its acceptance
risks are (i) explaining the novelty relative to three concurrent spline papers,
(ii) preventing reviewers from attributing the whole language gain to relabeling,
and (iii) presenting a clean physical confirmation. The matched A/C interaction,
learned event-clock ablation, and CALVIN timing result directly answer the first two;
a concise drawer/stack hardware study answers the third.

## Closest technical literature

### Continuous and spline action representations

| work | what it establishes | boundary for our claim |
|---|---|---|
| [B-spline Policy (BSP)](https://arxiv.org/abs/2607.09648) | predicts spline parameters in diffusion/ACT policies, resamples continuously, and manually rescales time in simulation and hardware | do not claim the first spline action head, continuous sampling, or decode-time rescaling; distinguish **learned event duration and language-addressable units** |
| [Spline Policy](https://arxiv.org/abs/2606.07386) | broad structured spline representation across policy families, editable constraints, arbitrary temporal resolution, and VLA/real-robot studies | do not frame the paper as “structured splines for policies”; its abstract and paper do not present our executable-clause factorial or learned event clock |
| [BEAST](https://arxiv.org/abs/2506.06072) | B-spline action tokenization with fixed-length tokens across several architectures and many tasks | compression and smoothness are supporting properties, not our novelty |
| [FAST](https://arxiv.org/abs/2501.09747) | frequency-space action tokenization for efficient autoregressive VLAs | contrast frequency-domain token compression with our geometric shape/time factorization |
| [PACE](https://arxiv.org/abs/2606.00537) | training-free phase-aware selection of execution horizons from low-speed transitions | contrast action-derived phase candidates with a supervised event duration that also schedules language clauses |

The safe, strong distinction is not that other spline methods cannot change execution
rate. It is that our head turns a compact curve **plus learned physical duration** into
an event-aligned execution unit, then tests whether that unit is easier to address by
piecewise language than a conventional waypoint chunk. The matched A/C factorial is
the key evidence: clauses help both heads, but help the spline-duration head by an
additional **9.33 percentage points**, positive on all three training seeds with a
hierarchical 95% CI of **[+0.67,+18.0]**.

### Language grounding and long-horizon execution

| work | what it establishes | distinction or connection |
|---|---|---|
| [VLAs as Tools](https://arxiv.org/abs/2605.13119) | a VLM planner invokes specialized VLA tools and receives progress feedback for long-horizon tasks | our clauses remain inside one end-to-end VLA with no tool library or separate planner; the head comparison isolates representation-specific benefit |
| [Long-VLA](https://arxiv.org/abs/2508.19958) | phase-aware inputs improve long-horizon VLA manipulation | our phases are executable language interventions linked to action units, not only an input mask |
| [CAST](https://arxiv.org/abs/2508.13446) | counterfactual language/action data improves fine-grained instruction following | supports Quinten's decorrelation direction; our primary evidence tests sequential order and later-clause execution |
| [LIBERO-CF / CAG](https://arxiv.org/abs/2602.17659) | exposes vision-over-language shortcuts and adds counterfactual inference guidance | motivates alternate-goal and matched-prompt scoring rather than interpreting wrong-prompt suppression alone |
| [pi0.5](https://arxiv.org/abs/2504.16054) | semantic subtask prediction and heterogeneous co-training support long tasks | a relevant pretrained-backbone reference; not necessary to reproduce within this deadline to establish the head-level causal result |
| [SmolVLA](https://arxiv.org/abs/2506.01844) | efficient open VLA backbone used by our controlled study | state clearly that the contribution is the action/program interface, not a new foundation model |

Two framing disciplines matter. First, clauses alone are not new: claim that the
spline-duration representation **amplifies** their benefit and exposes a deployable
event clock. Second, our current reverse-order probe establishes requested first-goal
steering, but not full reversed-program mastery; the proposed held-out-composition and
hardware experiments are the clean route to the stronger recombination claim.

### Timing, cadence, and language steering

The CALVIN result supplies a second capability of the same representation rather than
a disconnected historical project. Factorizing geometry from time permits a learned
curve to be retimed without retraining: the best decode recovers **+0.379 average
chain length at 1.42x realized speed** over the unretimed spline and is **+0.127** over
the waypoint policy on matched chains. The language-speed result then connects the two
halves of the paper: “quickly”/“slowly” changes action magnitude by about +/-16%, and
quick versus plain improves average chain length by **+0.172** at n=600.

This timing evidence should be positioned after executable clauses. It is important
because it demonstrates that the language-addressable unit has a physical clock, but
the manuscript should not compete with BSP on the generic claim “splines run faster.”

## Manuscript implication

The strongest defensible paper is:

> **An event-aligned spline action head factorizes trajectory geometry from physical
> duration, turning action chunks into compact units that can be scheduled and
> addressed by executable language. Piecewise language improves long-horizon success
> for both heads and significantly more for the spline head; the same factorization
> also provides language and decode-time control over execution timing.**

This statement preserves all useful earlier work. LIBERO suite parity proves that the
new target is not paid for with broad competence; CALVIN establishes the timing
capability; LIBERO-Long supplies the strongest causal language result; RoboCasa tests
semantic phase execution in a harder kitchen domain; the rate and density studies
explain when the representation helps. They should be ranked by role, not divided into
“old” and “new” campaigns.

## References to add to the manuscript BibTeX first

- B-spline Policy, Spline Policy, BEAST, FAST, PACE.
- SmolVLA, pi0.5, LIBERO, CALVIN, RoboCasa365.
- VLAs as Tools, Long-VLA, CAST, LIBERO-CF/CAG.
- ACT and Diffusion Policy for action-chunk context.
- ITPS for intervention-based steering evaluation.

Before final submission, verify author names, titles, versions, and venues from the
paper PDF or official project page when producing each BibTeX entry. Do not copy venue
metadata from search snippets.
