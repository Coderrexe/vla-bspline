# Project showcase release — 26 September 2026

Public project website: **https://vla-bspline.vercel.app/**

The repository README and a new isolated `website/` directory present the research
for external readers. Policy code, training code, experiment outputs, and the
manuscript were not modified for this release.

## What visitors can explore

- The research premise: executable language clauses, compact event-aligned spline
  actions, and an explicit physical clock.
- Real RoboCasa365 KettleBoiling and RinseSinkBasin videos, including synchronized
  original-label/phase-label comparisons from identical initial observations.
- A synthetic interactive cubic spline showing independent duration and sampling
  changes. This visualization does not connect to a model or robot.
- Three-seed LIBERO clause results, CALVIN retiming, duration-scheduled policy
  queries, RoboCasa replication, and expandable supporting studies.
- The hardware integration and an explicitly labeled human lamp demonstration.
- Author credits, the official APOLLO Lab mark, technical links, and CSV export.

The README includes a looping animation, a results figure, a method diagram,
benchmark tables, reproduction entry points, and a tested standalone spline example.
No publication acceptance, autonomous hardware success, or state-of-the-art claim
has been added.

## Evidence

`website/public/results.json` records selected values and source hashes.
`website/public/media-manifest.json` records video hashes, scene seeds, initial
observation identity, outcomes, phase switches, and simulator-step timing.
Raw experiment inputs stay outside the static site. The primary numbers are drawn
from `paper/results.md`, the locked clause factorial, CALVIN seeded statistics,
and official-phase confirmation artifacts.

## Checks completed

| Check | Outcome |
|---|---|
| Static build, local asset paths, anchor IDs, numerical export checks | Pass |
| Desktop/tablet/phone widths: 1440, 1024, 768, 390, 360 px | No horizontal overflow |
| Paired video playback, seek, final-frame hold, task switching | Pass |
| Result tabs, keyboard navigation, 13-row CSV export | Pass |
| Duration/rate sliders and invariant plotted path | Pass |
| Reduced-motion behavior | Pass |
| axe WCAG 2 A/AA + 2.1 AA checks, including expanded details | No detected violations |
| Browser console and local asset requests | No page errors or failed assets |
| README local links | All targets exist |
| `tests/test_event_targets.py` in a separate Python environment | 6 passed |
| README spline example at 12/24/48 actions | Same endpoint displacement |

The browser checks are reproducible with `website/tools/browser_check.py`.
The site uses Node 22 for build/deployment and has no production package dependencies.

## Sharing and maintenance

The Vercel page is independently public. During release, the GitHub origin
`Coderrexe/vla-bspline` was reachable with the local Git credentials but returned
404 to unauthenticated visitors. Repository visibility must be decided before
sharing a GitHub link externally. No repository privacy setting was changed.

Use `website/README.md` for local preview, deployment, media provenance, and asset
regeneration. Author portraits are optional; the current author row uses initials.
The website does not expose unpublished manuscript PDFs, robot credentials,
private endpoints, model weights, or the raw demonstration directory.
