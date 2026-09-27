const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const repository = "https://github.com/Coderrexe/vla-bspline/blob/main/";
const colors = {
  ink: "#173c3a",
  orange: "#b7603f",
  muted: "#68756d",
  line: "#d8ded2",
  gray: "#aebba3",
};
let results;
let currentResult = "language";

function notify(message) {
  $("#toast").textContent = message;
  $("#toast").classList.add("visible");
  window.setTimeout(() => $("#toast").classList.remove("visible"), 3500);
}

// Keyboard behavior for both compact tab interfaces.
$$('[role="tablist"]').forEach((list) => {
  list.addEventListener("keydown", (event) => {
    const tabs = [...list.querySelectorAll('[role="tab"]')];
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    let target;
    if (event.key === "ArrowRight") target = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft")
      target = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") target = 0;
    if (event.key === "End") target = tabs.length - 1;
    if (target !== undefined) {
      event.preventDefault();
      tabs[target].focus();
      tabs[target].click();
    }
  });
});

// The hero uses the actual recorded fixed-clause switch times: 128 and 242 steps.
const hero = $("#hero-video");
let heroManuallyPaused = false;
function syncHeroButton() {
  $("#hero-play").innerHTML = hero.paused
    ? 'Play <span aria-hidden="true">▶</span>'
    : 'Pause <span aria-hidden="true">Ⅱ</span>';
  $("#hero-play").setAttribute(
    "aria-label",
    hero.paused ? "Play hero video" : "Pause hero video",
  );
}
$("#hero-play").addEventListener("click", () => {
  heroManuallyPaused = !hero.paused;
  if (hero.paused)
    hero
      .play()
      .catch(() => notify("Use the video play button to start playback."));
  else hero.pause();
});
hero.addEventListener("play", syncHeroButton);
hero.addEventListener("pause", syncHeroButton);
hero.addEventListener("timeupdate", () => {
  const step = hero.currentTime * 20;
  const phase = step < 128 ? 0 : step < 242 ? 1 : 2;
  $$(".program-step").forEach((el, i) =>
    el.classList.toggle("is-active", i === phase),
  );
});
new IntersectionObserver(
  ([entry]) => {
    if (entry.isIntersecting && !reduceMotion.matches && !heroManuallyPaused)
      hero.play().catch(() => {});
    else hero.pause();
  },
  { threshold: 0.25 },
).observe(hero);

// Two source videos, one shared recording-time axis. Do not stretch the successful
// video to fill the baseline: hold its last frame when it ends.
const beforeVideo = $("#baseline-video");
const afterVideo = $("#phase-video");
const demoCases = {
  kettle: {
    seed: 1011,
    duration: 75,
    beforeSteps: 1500,
    afterSteps: 516,
    before:
      "Places the kettle, but does not complete burner actuation within 1,500 steps.",
    after: "Completes the pick → place → burner sequence in 516 steps.",
    schedule: "Identical fixed clause switch steps (128, 242).",
  },
  rinse: {
    seed: 1015,
    duration: 67.5,
    beforeSteps: 1350,
    afterSteps: 444,
    before:
      "Does not complete the ordered faucet-and-wash task within 1,350 steps.",
    after: "Turns on the water and completes the basin task in 444 steps.",
    schedule: "Identical fixed clause switch step (162).",
  },
};
let currentDemo = "kettle";
let comparisonPlaying = false;
let comparisonFrame;
function timeString(time) {
  return `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, "0")}`;
}
function duration() {
  return Number.isFinite(beforeVideo.duration)
    ? beforeVideo.duration
    : demoCases[currentDemo].duration;
}
function setComparisonState(playing) {
  comparisonPlaying = playing;
  $("#comparison-play").innerHTML = playing
    ? 'Pause comparison <span aria-hidden="true">Ⅱ</span>'
    : 'Play comparison <span aria-hidden="true">▶</span>';
}
function pauseComparison() {
  beforeVideo.pause();
  afterVideo.pause();
  setComparisonState(false);
  window.cancelAnimationFrame(comparisonFrame);
}
function syncComparison() {
  const position = beforeVideo.currentTime;
  $("#comparison-time").textContent =
    `${timeString(position)} / ${timeString(duration())}`;
  $("#comparison-scrub").value = duration() ? (position / duration()) * 100 : 0;
  if (Number.isFinite(afterVideo.duration)) {
    const target = Math.min(position, Math.max(0, afterVideo.duration - 0.04));
    if (Math.abs(afterVideo.currentTime - target) > 0.25)
      afterVideo.currentTime = target;
    if (position >= afterVideo.duration - 0.04) afterVideo.pause();
    else if (comparisonPlaying && afterVideo.paused)
      afterVideo.play().catch(() => {});
  }
  if (comparisonPlaying)
    comparisonFrame = requestAnimationFrame(syncComparison);
}
async function playComparison() {
  if (beforeVideo.ended) {
    beforeVideo.currentTime = 0;
    afterVideo.currentTime = 0;
  }
  const speed = Number($("#comparison-speed").value);
  beforeVideo.playbackRate = speed;
  afterVideo.playbackRate = speed;
  try {
    await beforeVideo.play();
    if (
      !Number.isFinite(afterVideo.duration) ||
      beforeVideo.currentTime < afterVideo.duration - 0.04
    )
      await afterVideo.play();
    setComparisonState(true);
    syncComparison();
  } catch {
    pauseComparison();
    notify("Video playback could not start. Please try again.");
  }
}
$("#comparison-play").addEventListener("click", () =>
  comparisonPlaying ? pauseComparison() : playComparison(),
);
$("#comparison-restart").addEventListener("click", () => {
  pauseComparison();
  beforeVideo.currentTime = 0;
  afterVideo.currentTime = 0;
  $("#comparison-scrub").value = 0;
  syncComparison();
  playComparison();
});
$("#comparison-speed").addEventListener("change", (event) => {
  beforeVideo.playbackRate = Number(event.target.value);
  afterVideo.playbackRate = Number(event.target.value);
});
$("#comparison-scrub").addEventListener("input", (event) => {
  if (beforeVideo.readyState === 0) {
    beforeVideo.load();
    afterVideo.load();
    notify("Loading the recordings. Press play to begin.");
    return;
  }
  beforeVideo.currentTime = (Number(event.target.value) / 100) * duration();
  if (afterVideo.readyState > 0)
    afterVideo.currentTime = Math.min(
      beforeVideo.currentTime,
      afterVideo.duration - 0.04,
    );
  if (!comparisonPlaying) syncComparison();
});
beforeVideo.addEventListener("ended", () => {
  pauseComparison();
  syncComparison();
});
beforeVideo.addEventListener("loadedmetadata", () => {
  if (!comparisonPlaying) syncComparison();
});
$$("[data-demo]").forEach((tab) =>
  tab.addEventListener("click", () => {
    if (tab.dataset.demo === currentDemo) return;
    pauseComparison();
    currentDemo = tab.dataset.demo;
    $$("[data-demo]").forEach((el) => {
      const selected = el === tab;
      el.setAttribute("aria-selected", String(selected));
      el.tabIndex = selected ? 0 : -1;
    });
    $("#demo-panel").setAttribute("aria-labelledby", tab.id);
    const item = demoCases[currentDemo];
    beforeVideo.poster = `/media/${currentDemo}-baseline.webp`;
    afterVideo.poster = `/media/${currentDemo}.webp`;
    beforeVideo.src = `/media/${currentDemo}-baseline.mp4`;
    afterVideo.src = `/media/${currentDemo}.mp4`;
    beforeVideo.load();
    afterVideo.load();
    $("#baseline-caption").textContent = item.before;
    $("#phase-caption").textContent = item.after;
    $("#demo-protocol").textContent =
      `Selected paired scene ${item.seed}, training seed 1000. Identical initial observation hash; ${item.schedule.toLowerCase()} Videos record every five simulation steps at 4 fps. The successful rollout holds its final frame while the baseline continues.`;
    $("#comparison-time").textContent = `0:00 / ${timeString(item.duration)}`;
    $("#comparison-scrub").value = 0;
  }),
);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    hero.pause();
    pauseComparison();
  }
});
reduceMotion.addEventListener("change", () => {
  if (reduceMotion.matches) {
    hero.pause();
    animationRunning = false;
    updateMotionButton();
  }
});

// Clamped uniform cubic B-spline, evaluated by Cox–de Boor. This is a schematic
// command path with eight controls, not an empirical trajectory.
const controls = [
  [55, 255],
  [99, 244],
  [141, 90],
  [224, 61],
  [317, 274],
  [396, 285],
  [485, 112],
  [563, 118],
];
const knots = [0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1];
function basis(i, degree, u) {
  if (degree === 0) return knots[i] <= u && u < knots[i + 1] ? 1 : 0;
  const a = knots[i + degree] - knots[i];
  const b = knots[i + degree + 1] - knots[i + 1];
  return (
    (a ? ((u - knots[i]) / a) * basis(i, degree - 1, u) : 0) +
    (b ? ((knots[i + degree + 1] - u) / b) * basis(i + 1, degree - 1, u) : 0)
  );
}
function point(u) {
  if (u >= 1) return controls.at(-1);
  const result = [0, 0];
  controls.forEach((p, i) => {
    const w = basis(i, 3, u);
    result[0] += w * p[0];
    result[1] += w * p[1];
  });
  return result;
}
function circleMarkup(p, radius) {
  return `<circle cx="${p[0].toFixed(3)}" cy="${p[1].toFixed(3)}" r="${radius}"/>`;
}
$("#control-polygon").setAttribute(
  "d",
  controls.map((p, i) => `${i ? "L" : "M"}${p.join(",")}`).join(" "),
);
$("#control-points").innerHTML = controls
  .map((p) => circleMarkup(p, 4))
  .join("");
$("#spline-path").setAttribute(
  "d",
  Array.from(
    { length: 241 },
    (_, i) => `${i ? "L" : "M"}${point(i / 240).join(",")}`,
  ).join(" "),
);
let alpha = 1;
let animationRunning = !reduceMotion.matches;
let progress = 0;
let lastFrame = 0;
let explorerVisible = false;
let explorerFrame;
function updateSamples() {
  alpha = Number($("#duration-control").value);
  const rho = Number($('input[name="rate"]:checked').value);
  const count = Math.max(2, Math.round(alpha * rho * 24));
  $("#sample-points").innerHTML = Array.from({ length: count + 1 }, (_, i) =>
    circleMarkup(point(i / count), 2.7),
  ).join("");
  $("#duration-label").textContent = `${alpha.toFixed(2)}×`;
  $("#duration-stat").textContent = `${(alpha * 1.2).toFixed(2)} s`;
  $("#samples-stat").textContent = String(count);
}
function positionBead() {
  const p = point(progress);
  for (const id of ["#spline-bead", "#spline-bead-halo"]) {
    $(id).setAttribute("cx", p[0]);
    $(id).setAttribute("cy", p[1]);
  }
}
function animate(time) {
  const delta = lastFrame ? Math.min(time - lastFrame, 50) : 0;
  lastFrame = time;
  if (animationRunning) progress = (progress + delta / (1200 * alpha)) % 1;
  positionBead();
  if (explorerVisible && animationRunning)
    explorerFrame = requestAnimationFrame(animate);
}
function updateMotionButton() {
  $("#motion-toggle").textContent = animationRunning
    ? "Pause animation"
    : "Play animation";
  $("#motion-toggle").setAttribute(
    "aria-label",
    animationRunning ? "Pause spline animation" : "Play spline animation",
  );
}
$("#motion-toggle").addEventListener("click", () => {
  animationRunning = !animationRunning;
  updateMotionButton();
  lastFrame = 0;
  cancelAnimationFrame(explorerFrame);
  if (animationRunning && explorerVisible)
    explorerFrame = requestAnimationFrame(animate);
});
new IntersectionObserver(
  ([entry]) => {
    explorerVisible = entry.isIntersecting;
    lastFrame = 0;
    cancelAnimationFrame(explorerFrame);
    if (explorerVisible && animationRunning)
      explorerFrame = requestAnimationFrame(animate);
  },
  { threshold: 0.1 },
).observe($("#spline-plot"));
$("#duration-control").addEventListener("input", updateSamples);
$$('input[name="rate"]').forEach((input) =>
  input.addEventListener("change", updateSamples),
);
updateSamples();
positionBead();
updateMotionButton();

// Charts share the exported, source-bound research data with the README assets.
function chartSVG({
  groups,
  series,
  maximum,
  ticks,
  axisLabel,
  format,
  description,
}) {
  const compact = window.matchMedia("(max-width: 500px)").matches;
  const width = compact ? 400 : 630,
    height = compact ? 320 : 355,
    left = compact ? 35 : 51,
    right = 10,
    top = 33,
    bottom = 67;
  const plotWidth = width - left - right,
    plotHeight = height - top - bottom;
  const labelText = (value) => (format ? format(value) : String(value));
  let content = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${description}"><text x="${left}" y="14" class="chart-label">${axisLabel}</text>`;
  ticks.forEach((tick) => {
    const y = top + plotHeight * (1 - tick / maximum);
    content += `<path d="M${left} ${y}H${width - right}" stroke="${colors.line}"/><text x="${left - 10}" y="${y + 4}" text-anchor="end" class="chart-label">${tick}</text>`;
  });
  const groupWidth = plotWidth / groups.length,
    barWidth = Math.min(46, groupWidth / 5);
  groups.forEach((group, index) => {
    const center = left + groupWidth * (index + 0.5);
    series.forEach((s, j) => {
      const val = s.values[index];
      const barHeight = (val / maximum) * plotHeight;
      const x =
        center + (j - (series.length - 1) / 2) * (barWidth + 10) - barWidth / 2;
      content += `<rect x="${x}" y="${top + plotHeight - barHeight}" width="${barWidth}" height="${barHeight}" fill="${s.color}"/><text x="${x + barWidth / 2}" y="${top + plotHeight - barHeight - 9}" text-anchor="middle" class="chart-value">${labelText(val)}</text>`;
    });
    content += `<text x="${center}" y="${height - bottom + 24}" text-anchor="middle" class="chart-label">${group}</text>`;
  });
  const legendStep = compact ? 173 : 184;
  const legendWidth = series.length * legendStep;
  const legendStart = (width - legendWidth) / 2;
  series.forEach((s, i) => {
    const x = legendStart + i * legendStep;
    content += `<rect x="${x}" y="${height - 16}" width="9" height="9" fill="${s.color}"/><text x="${x + 16}" y="${height - 7}" class="chart-label">${s.label}</text>`;
  });
  return content + "</svg>";
}
function renderResult(key) {
  currentResult = key;
  $$("[data-result]").forEach((tab) => {
    const selected = tab.dataset.result === key;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  $("#result-panel").setAttribute("aria-labelledby", `tab-${key}`);
  if (!results) return;
  let content;
  if (key === "language") {
    content = {
      kicker: "LIBERO-LONG / T0 + T4",
      title: "Clauses help both heads.<br>The spline head benefits more.",
      description:
        "With identical clause inputs at inference, changing training labels improves waypoint success by 34.3 points and spline success by 43.7 points.",
      value: `+${results.language.interaction_pp.toFixed(2)} pp`,
      valueLabel: "spline × language interaction",
      note: "Positive on all three seeds. Hierarchical 95% CI: [+0.67, +18.0] pp. This compares training labels under a fixed clause schedule, not whole-caption inference against clause inference.",
      protocol:
        "Three training seeds · 100 rollouts per seed and condition · 71 shared demonstrations",
      source: results.language.source,
      chart: {
        groups: ["Seed 1000", "Seed 1001", "Seed 1002"],
        series: results.language.rows.map((r, i) => ({
          label: r.head,
          color: i ? colors.ink : colors.orange,
          values: r.seeds.map((s) => s.clause_minus_original * 100),
        })),
        maximum: 60,
        ticks: [0, 15, 30, 45, 60],
        axisLabel: "Gain from clause training (percentage points)",
        format: (v) => `+${v.toFixed(0)}`,
        description:
          "Clause gains: waypoint +31, +30, +42 percentage points; spline +41, +42, +48 across training seeds 1000, 1001, 1002.",
      },
    };
  } else if (key === "clock") {
    content = {
      kicker: "CALVIN / THREE SEEDS × 1,000 CHAINS",
      title: "A different clock.<br>The same trained policy.",
      description:
        "Retiming the frozen spline decoder increases average completed chain length from 1.31 to 1.69, at 1.42× realized pace.",
      value: `+${results.calvin.delta.toFixed(3)}`,
      valueLabel: "mean completed subtasks per chain",
      note: "95% CI: [0.318, 0.441]. This is a decoder intervention relative to the native spline. A separate matched interpolation screen finds that retiming also benefits waypoints; it does not establish a spline-specific retiming advantage.",
      protocol:
        "3,000 official five-task chains per condition · paired comparison · no retraining",
      source: results.calvin.source,
      chart: {
        groups: ["Seed 1000", "Seed 1001", "Seed 1002"],
        series: [
          {
            label: "Native spline",
            color: colors.gray,
            values: results.calvin.native_seeds,
          },
          {
            label: "Retimed spline",
            color: colors.ink,
            values: results.calvin.retimed_seeds,
          },
        ],
        maximum: 2,
        ticks: [0, 0.5, 1, 1.5, 2],
        axisLabel: "Mean completed subtasks per chain",
        format: (v) => v.toFixed(2),
        description:
          "CALVIN mean chain lengths before and after retiming: seed 1000 1.422 to 1.734; seed 1001 1.239 to 1.579; seed 1002 1.273 to 1.759.",
      },
    };
  } else {
    content = {
      kicker: "LIBERO / DURATION-SCHEDULED REPLANNING",
      title: "Ask the policy<br>when it matters.",
      description:
        "Replanning four steps before the predicted motion event cuts policy queries while preserving observed task success on Object and Long.",
      value: "2.24–2.68×",
      valueLabel: "fewer policy calls per episode",
      note: "Object success: 93% → 93%. Long: 61% → 62%. These are counts over 100 episodes per condition, not a formal equivalence test. A tuned fixed cadence is also competitive; fewer queries do not imply a faster individual forward pass.",
      protocol:
        "100 episodes per suite and condition · same checkpoint · compare against fixed K=5",
      source: results.queries.source,
      chart: {
        groups: ["Object · 93% → 93%", "Long · 61% → 62%"],
        series: [
          {
            label: "Fixed every 5 steps",
            color: colors.gray,
            values: results.queries.rows.map((r) => r.before),
          },
          {
            label: "Predicted duration − 4",
            color: colors.ink,
            values: results.queries.rows.map((r) => r.after),
          },
        ],
        maximum: 90,
        ticks: [0, 30, 60, 90],
        axisLabel: "Policy calls per episode · lower is better",
        format: (v) => v.toFixed(1),
        description:
          "Object: 31.3 to 11.7 calls, 93% success in both conditions. Long: 73.8 to 33.0 calls, success 61% to 62%.",
      },
    };
  }
  $("#result-kicker").textContent = content.kicker;
  $("#result-title").innerHTML = content.title.replaceAll("<br>", "<br> ");
  $("#result-description").textContent = content.description;
  $("#result-value").textContent = content.value;
  $("#result-value-label").textContent = content.valueLabel;
  $("#result-note").textContent = content.note;
  $("#result-protocol").textContent = content.protocol;
  $("#result-source").href = content.source.startsWith("/")
    ? content.source
    : repository + content.source;
  $("#result-chart").innerHTML = chartSVG(content.chart);
}
$$("[data-result]").forEach((tab) =>
  tab.addEventListener("click", () => renderResult(tab.dataset.result)),
);
$$("[data-result-link]").forEach((link) =>
  link.addEventListener("click", () => renderResult(link.dataset.resultLink)),
);
window
  .matchMedia("(max-width: 500px)")
  .addEventListener("change", () => renderResult(currentResult));
fetch("/results.json")
  .then((response) => {
    if (!response.ok) throw new Error("Results unavailable");
    return response.json();
  })
  .then((data) => {
    results = data;
    renderResult(currentResult);
  })
  .catch(() => {
    $("#result-note").textContent +=
      " Interactive data is unavailable; the source ledger and static figure remain accessible.";
    $$("[data-result]").forEach((tab) => {
      if (tab.dataset.result !== "language") tab.disabled = true;
    });
  });
$("#download-results").addEventListener("click", () => {
  if (!results) {
    notify("Results are still loading. Please try again.");
    return;
  }
  const rows = [
    [
      "experiment",
      "condition",
      "training_seed_or_suite",
      "before",
      "after",
      "unit",
      "n_per_condition",
      "protocol",
    ],
  ];
  results.language.rows.forEach((head) =>
    head.seeds.forEach((seed) =>
      rows.push([
        "LIBERO clauses",
        head.head,
        seed.seed,
        seed.original_successes,
        seed.clause_successes,
        "successes",
        seed.n,
        "Same fixed clause input; training labels differ",
      ]),
    ),
  );
  results.calvin.native_seeds.forEach((v, i) =>
    rows.push([
      "CALVIN retiming",
      "event spline",
      1000 + i,
      v,
      results.calvin.retimed_seeds[i],
      "mean chain length",
      1000,
      "Native vs retimed; historical K=10 protocol",
    ]),
  );
  results.robocasa.rows.forEach((r) =>
    rows.push([
      "RoboCasa phases",
      "two-task aggregate",
      r.seed,
      r.before,
      r.after,
      "successes",
      r.n,
      "Fixed clause schedule; original vs official phase training labels",
    ]),
  );
  results.queries.rows.forEach((r) =>
    rows.push([
      "Policy queries",
      "event spline",
      r.suite,
      r.before,
      r.after,
      "calls per episode",
      100,
      "Fixed K=5 vs predicted duration minus 4",
    ]),
  );
  const csv = rows
    .map((row) =>
      row.map((v) => `"${String(v).replaceAll('"', '""')}"`).join(","),
    )
    .join("\n");
  const url = URL.createObjectURL(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "vla-bspline-selected-results.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify("Selected results downloaded with protocol labels.");
});
