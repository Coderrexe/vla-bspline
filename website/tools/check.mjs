import { readFile, stat, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import assert from "node:assert/strict";

const root = resolve(import.meta.dirname, "../public");
const html = await readFile(resolve(root, "index.html"), "utf8");
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
assert.equal(new Set(ids).size, ids.length, "Duplicate element IDs");
for (const [, href] of html.matchAll(/(?:href|src|poster)="([^"]+)"/g)) {
  if (href.startsWith("#") && href !== "#")
    assert(ids.includes(href.slice(1)), `Missing anchor ${href}`);
  if (href.startsWith("/")) await stat(resolve(root, "." + href));
}
const results = JSON.parse(
  await readFile(resolve(root, "results.json"), "utf8"),
);
assert.deepEqual(
  results.language.rows.map((r) => [r.before, r.after, r.n]),
  [
    [21, 124, 300],
    [35, 166, 300],
  ],
);
assert(Math.abs(results.language.interaction_pp - 9.333333333333334) < 1e-8);
for (const row of results.language.rows) {
  assert.equal(
    row.seeds.reduce((n, s) => n + s.original_successes, 0),
    row.before,
  );
  assert.equal(
    row.seeds.reduce((n, s) => n + s.clause_successes, 0),
    row.after,
  );
}
const manifest = JSON.parse(
  await readFile(resolve(root, "media-manifest.json"), "utf8"),
);
assert.equal(manifest.videos.length, 5);
for (const i of [0, 2]) {
  assert.equal(
    manifest.videos[i].initial_observation_sha256,
    manifest.videos[i + 1].initial_observation_sha256,
  );
  assert.equal(manifest.videos[i].success, true);
  assert.equal(manifest.videos[i + 1].success, false);
}
assert.equal(manifest.videos[4].kind, "human teleoperation");
const files = await readdir(root, { recursive: true });
assert(
  !files.some((p) => /\.(pt|pth|parquet|ckpt|env|tex|pdf)$/.test(p)),
  "Unexpected research/private artifact in public deployment",
);
assert(!html.includes("localhost"), "Local URL in production page");
console.log(
  "PASS: local assets, anchors, IDs, public data, paired video evidence, deployment contents.",
);
