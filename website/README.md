# VLA B-Spline project website

A static, responsive research presentation with real policy videos, synchronized
comparisons, evidence-backed charts, and an interactive cubic B-spline decoder.
The site has no production JavaScript dependencies, tracking scripts, or backend.

## Local preview

Use Node.js 22 (the deployment version). No package installation is needed.

```bash
cd website
npm run dev
# http://127.0.0.1:4173
```

```bash
npm run check
npm run build
npm run preview
```

## Deploy to Vercel

Set the project root to **website**, framework to **Other**, build command to
**npm run build**, and output directory to **dist**. The checked-in Vercel config
sets the build and security headers. Alternatively, from this directory:

```bash
vercel link --project vla-bspline
vercel --prod
```

Only `public/` is copied into `dist/`. Do not deploy the repository root: it contains
unpublished research artifacts and local experiment material. No model weights,
robot endpoints, SSH configuration, or raw dataset directory is deployed.

## Content and assets

| File | Purpose |
|---|---|
| `public/index.html` | Semantic page content and accessible controls |
| `public/style.css` | Responsive layout, self-hosted fonts, reduced-motion support |
| `public/main.js` | Paired video playback, result charts, CSV export, spline explorer |
| `public/results.json` | Selected numbers with source artifact hashes |
| `public/media-manifest.json` | Video origins, hashes, outcomes, matched-scene identities |
| `public/media/` | Curated real videos, posters, README animation, vector figures |
| `tools/prepare_assets.py` | Regenerates evidence exports and derived presentation media |
| `tools/prepare_portraits.py` | Crops supplied author photos into circular, metadata-free PNGs |
| `tools/check.mjs` | Link, data, provenance, and public-build checks |

The media generator requires the original local `outputs/` and lamp demonstration,
plus Pillow, NumPy and imageio-ffmpeg. Those source directories are intentionally
not included in the website. Generated public files are committed so cloning or
deploying the website does not require the private experiment artifacts.

```bash
# From the repository root, in a Python environment with those dependencies:
python website/tools/prepare_assets.py
```

The public site uses training seed 1000's scene 1011 for the paired KettleBoiling
videos and scene 1015 for RinseSinkBasin. Recorded video time is mapped to simulator
steps using five environment steps per saved frame and four frames per second.
The paired viewer freezes a completed video's final frame; it never stretches one
trajectory to match the other's length. Playback-rate changes are display controls,
not policy retiming experiments. The hardware video is explicitly labeled as a
human demonstration, not a learned-policy success.

The interactive curve is a synthetic 2D clamped cubic B-spline. Its duration and
sampling controls implement the illustrative decode equation. It does not connect
to a model, simulator, or robot.

## Credits

- Research: Simba Shi, Quinten Jin, Xiatao Sun, in collaboration with Yale APOLLO Lab.
- Lab mark: the [official APOLLO Lab website](https://apollo-lab-yale.github.io/assets/theme/images/apollo-lab-logo.png), used for affiliation attribution.
- DM Sans and Instrument Serif: Google Fonts; SIL Open Font License files are included in `public/fonts/`.
- Rollouts and demonstration recordings: this project. Simulation assets belong to their upstream benchmark authors.
- The [Spatial-MemER website](https://spatial-memer.vercel.app/) informed the editorial emphasis on an approachable research explanation and visible demonstrations. This site's code, layout, figures, and interactions are newly implemented.

Author portraits are derived from the user-supplied `photos/` originals. Run
`python website/tools/prepare_portraits.py` from a Pillow environment to regenerate
them. Only the cropped public PNGs are needed for deployment or the README;
original photographs remain untouched. There is no publication badge or manuscript
link while the paper is in preparation.
