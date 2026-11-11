// roblox-up-and-coming-forecast-01 entry (Node 22, standard library only).
//
// A Monte Carlo rank-diffusion forecaster. The Up-and-Coming chart is a
// computed order that moves fast (median 6 of the top 10 change between
// archive reads ~6 h apart) and its player counts do not predict rank moves
// (measured correlation of log playing growth with rank change ~ -0.02, and
// with the *next* rank change ~ -0.02), so this entry does not use playing
// counts as a signal at all. What it uses, all estimated from the history
// it is given at run time:
//
//  - per-rank-band drift and volatility of rank changes (a rank random
//    walk: top-10 games drift down slowly and move ~2.3 ranks per ~6 h,
//    lower bands diffuse faster and climb on average, a refresh effect);
//  - rank momentum: a game's recent rank velocity persists into its next
//    move (measured correlation ~0.22), so each game carries a velocity
//    that decays hour by hour and biases its drift;
//  - churn: games leave the chart (mostly from the bottom bands) and fresh
//    games enter at the bottom; when a game's creation date is known
//    (house-recorder reads carry details) it leaves for good at 28 days;
//  - the scored window is 72 hourly slots and one forecast is scored
//    against all of them, so the forecast is the time average over the
//    window of the simulated place-occupancy distribution: the honest
//    probabilities for a log score.
//
// Deterministic: a fixed-seed PRNG (mulberry32 + Box-Muller), no wall
// clock, no network, no model call.
//
// Local validation (local scorer rebuilt from the rules; practice answer
// rebuilt from the public MIT archive Kenny-Dhub/roblox-snapshots; the
// scorer reproduces the house baseline exactly at -43.512039):
// practice window -35.40 (baseline -43.51), second time-split -41.70
// (baseline -56.91). A momentum-carry variant scored worse on both
// splits and was rejected.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ---- tunables (coarse; tuned on the practice window + a second split) ----
// Defaults are the tuned values; the RBX_* environment variables exist only
// so the entry can be re-tuned locally, and are unset in the grader.
const envNum = (name: string, dflt: number) => {
  const v = process.env[name];
  return v === undefined || v === '' || Number.isNaN(Number(v)) ? dflt : Number(v);
};
const SIMS = envNum('RBX_SIMS', 2000); // Monte Carlo paths
const UNNAMED_RESERVE = envNum('RBX_UNNAMED', 0.30); // share of each place left for games we cannot name
const MIN_P = envNum('RBX_MIN_P', 0.004); // drop smaller rows
const MOMENTUM_FALLBACK = 0.2; // if the history is too thin to fit it
const MOMENTUM_MULT = envNum('RBX_MOM_MULT', 1); // scales the fitted momentum slope
const VOL_MULT = envNum('RBX_VOL_MULT', 1.15); // scales fitted band volatility
const VELOCITY_DECAY = envNum('RBX_VDECAY', 0.95); // per hour
const VELOCITY_CLIP = 1.5; // ranks per hour
const SEED = 20261111;

// ---- seeded PRNG ----
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);
let spareGauss: number | null = null;
function gauss(): number {
  if (spareGauss !== null) { const g = spareGauss; spareGauss = null; return g; }
  let u = 0, v = 0;
  while (u === 0) u = rand();
  v = rand();
  const r = Math.sqrt(-2 * Math.log(u));
  const t = 2 * Math.PI * v;
  spareGauss = r * Math.sin(t);
  return r * Math.cos(t);
}

// ---- types ----
interface ChartRow { rank: number; universe_id: number; playing: number | null; }
interface Snapshot { at: number; rows: ChartRow[]; details: Record<string, { created?: string }> | null; }
interface GameState {
  id: number; x: number; v: number; known: boolean;
  exitAt: number | null; // ms epoch when the game ages out (28 d after created), if known
}

function bandOf(rank: number): number {
  if (rank <= 10) return 0;
  if (rank <= 30) return 1;
  if (rank <= 60) return 2;
  return 3;
}
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

// ---- read input ----
const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number; output: string; window?: { slots: number; slot_minutes: number };
};
const HOURS = task.window ? task.window.slots * (task.window.slot_minutes / 60) : 72;
const PLACES = task.places;
const snaps: Snapshot[] = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n').filter((l) => l.trim() !== '')
  .map((l) => {
    const o = JSON.parse(l) as {
      at: string; up_and_coming: ChartRow[] | null;
      details: Record<string, { created?: string }> | null;
    };
    return { at: Date.parse(o.at), rows: o.up_and_coming ?? [], details: o.details ?? null };
  })
  .filter((s) => s.rows.length > 0)
  .sort((a, b) => a.at - b.at);
if (snaps.length === 0) throw new Error('no snapshots in history');
const last = snaps[snaps.length - 1]!;
const t0 = last.at;

// ---- per-game rank series, and velocity from a recency-weighted fit ----
const series = new Map<number, Array<{ t: number; rank: number }>>();
for (const s of snaps) for (const r of s.rows) {
  const arr = series.get(r.universe_id) ?? [];
  arr.push({ t: s.at, rank: r.rank });
  series.set(r.universe_id, arr);
}
function velocity(id: number): number {
  const pts = (series.get(id) ?? []).filter((p) => t0 - p.t <= 96 * 3600e3);
  if (pts.length < 2) return 0;
  const use = pts.slice(-8);
  // Weighted least squares of rank on time (hours), weights decay with age (half-life 24 h).
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const p of use) {
    const x = (p.t - t0) / 3600e3;
    const w = Math.pow(0.5, -x / 24);
    sw += w; sx += w * x; sy += w * p.rank; sxx += w * x * x; sxy += w * x * p.rank;
  }
  const den = sw * sxx - sx * sx;
  if (den === 0) return 0;
  return clamp((sw * sxy - sx * sy) / den, -VELOCITY_CLIP, VELOCITY_CLIP);
}

// ---- fit band drift / volatility / exit hazard / entry ranks from pairs ----
const BANDS = 4;
const driftSum = new Array(BANDS).fill(0); // sum of rank changes
const gapSum = new Array(BANDS).fill(0); // sum of gaps (h)
const sqSum = new Array(BANDS).fill(0); // sum of (drank - mu*gap)^2 / gap
const pairN = new Array(BANDS).fill(0);
const exitN = new Array(BANDS).fill(0); // games that left
const exposure = new Array(BANDS).fill(0); // game-hours at risk
const entryRanks: number[] = [];
const momX: number[] = []; const momY: number[] = [];
for (let i = 1; i < snaps.length; i++) {
  const a = snaps[i - 1]!, b = snaps[i]!;
  const gap = (b.at - a.at) / 3600e3;
  if (gap <= 0 || gap > 24) continue;
  const am = new Map(a.rows.map((r) => [r.universe_id, r]));
  const bm = new Map(b.rows.map((r) => [r.universe_id, r]));
  for (const r of a.rows) {
    const bd = bandOf(r.rank);
    exposure[bd] += gap;
    const nb = bm.get(r.universe_id);
    if (!nb) { exitN[bd]++; continue; }
    const dr = nb.rank - r.rank;
    driftSum[bd] += dr; gapSum[bd] += gap; pairN[bd]++;
  }
  for (const r of b.rows) if (!am.has(r.universe_id)) entryRanks.push(r.rank);
  // momentum triplets: need the read before a as well
  if (i >= 2) {
    const p = snaps[i - 2]!;
    const gap1 = (a.at - p.at) / 3600e3;
    if (gap1 <= 0 || gap1 > 24) continue;
    const pm = new Map(p.rows.map((r) => [r.universe_id, r]));
    for (const r of b.rows) {
      const ra = am.get(r.universe_id); const rp = pm.get(r.universe_id);
      if (!ra || !rp) continue;
      momX.push((ra.rank - rp.rank) / gap1);
      momY.push((r.rank - ra.rank) / gap);
    }
  }
}
const mu = new Array(BANDS).fill(0); // drift ranks/hour
const sd = new Array(BANDS).fill(0); // volatility ranks/sqrt(hour)
const hazard = new Array(BANDS).fill(0); // exits per game-hour
for (let b = 0; b < BANDS; b++) {
  mu[b] = gapSum[b] > 0 ? driftSum[b] / gapSum[b] : 0;
  hazard[b] = exposure[b] > 0 ? exitN[b] / exposure[b] : 0;
}
for (let i = 1; i < snaps.length; i++) {
  const a = snaps[i - 1]!, b = snaps[i]!;
  const gap = (b.at - a.at) / 3600e3;
  if (gap <= 0 || gap > 24) continue;
  const bm = new Map(b.rows.map((r) => [r.universe_id, r]));
  for (const r of a.rows) {
    const nb = bm.get(r.universe_id);
    if (!nb) continue;
    const bd = bandOf(r.rank);
    const dr = nb.rank - r.rank;
    sqSum[bd] += ((dr - mu[bd] * gap) ** 2) / gap;
  }
}
for (let b = 0; b < BANDS; b++) {
  sd[b] = pairN[b] > 0 ? Math.sqrt(Math.max(sqSum[b] / pairN[b], 0.05)) : 1;
  mu[b] = clamp(mu[b], -0.5, 0.5);
  sd[b] = clamp(sd[b], 0.3, 6);
  hazard[b] = clamp(hazard[b], 0, 0.05);
}
// momentum slope (least squares, clipped)
let momentum = MOMENTUM_FALLBACK;
if (momX.length > 200) {
  const mx = momX.reduce((s, x) => s + x, 0) / momX.length;
  const my = momY.reduce((s, y) => s + y, 0) / momY.length;
  let cxx = 0, cxy = 0;
  for (let k = 0; k < momX.length; k++) { cxx += (momX[k]! - mx) ** 2; cxy += (momX[k]! - mx) * (momY[k]! - my); }
  if (cxx > 0) momentum = clamp(cxy / cxx, 0, 0.4) * MOMENTUM_MULT;
}
entryRanks.sort((a, b) => a - b);
function sampleEntryRank(): number {
  if (entryRanks.length === 0) return 70 + Math.floor(rand() * 29);
  return entryRanks[Math.floor(rand() * entryRanks.length)]!;
}
const entryRatePerHour = entryRanks.length > 0
  ? entryRanks.length / Math.max((t0 - snaps[0]!.at) / 3600e3, 1)
  : 0.5;

// ---- initial state from the last chart ----
const createdOf = new Map<number, number>();
if (last.details) {
  for (const [id, d] of Object.entries(last.details)) {
    if (d && d.created) createdOf.set(Number(id), Date.parse(d.created));
  }
}
function makeInitial(): GameState[] {
  return last.rows.map((r) => {
    const c = createdOf.get(r.universe_id);
    return {
      id: r.universe_id, x: r.rank, v: velocity(r.universe_id), known: true,
      exitAt: c !== undefined ? c + 28 * 86400e3 : null,
    };
  });
}
let nextUnknownId = -1;

// ---- simulate ----
const counts = new Map<number, number[]>(); // id -> counts per place (1..PLACES)
function bump(id: number, place: number) {
  let arr = counts.get(id);
  if (!arr) { arr = new Array(PLACES).fill(0); counts.set(id, arr); }
  arr[place - 1]!++;
}
for (let sim = 0; sim < SIMS; sim++) {
  let live = makeInitial();
  for (let h = 1; h <= HOURS; h++) {
    const now = t0 + h * 3600e3;
    // move
    for (const g of live) {
      const b = bandOf(Math.max(1, Math.round(g.x)));
      g.x += mu[b]! + momentum * g.v + gauss() * sd[b]! * VOL_MULT;
      g.v *= VELOCITY_DECAY;
    }
    // exits: aging, then hazard
    let exits = 0;
    live = live.filter((g) => {
      if (g.exitAt !== null && now >= g.exitAt) { exits++; return false; }
      const b = bandOf(Math.max(1, Math.round(g.x)));
      if (rand() < hazard[b]!) { exits++; return false; }
      return true;
    });
    // entries: replace exits plus the background entry rate, jittered
    let nEnter = exits + Math.round(entryRatePerHour + (rand() - 0.5));
    if (live.length + nEnter < 60) nEnter = Math.max(0, 60 - live.length);
    for (let e = 0; e < nEnter; e++) {
      live.push({ id: nextUnknownId--, x: sampleEntryRank() + gauss() * 3, v: 0, known: false, exitAt: null });
    }
    // re-rank by latent position (ties broken by id for determinism)
    live.sort((a, b) => (a.x - b.x) || (a.id - b.id));
    for (let k = 0; k < Math.min(PLACES, live.length); k++) {
      const g = live[k]!;
      if (g.known) bump(g.id, k + 1);
    }
  }
}

// ---- build the forecast: time-averaged occupancy, scaled per place ----
const total = SIMS * HOURS;
const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
const perPlaceTop: Array<Array<{ id: number; p: number }>> = [];
for (let place = 1; place <= PLACES; place++) {
  const rows: Array<{ id: number; p: number }> = [];
  for (const [id, arr] of counts) {
    const p = arr[place - 1]! / total;
    if (p >= MIN_P) rows.push({ id, p });
  }
  rows.sort((a, b) => b.p - a.p || a.id - b.id);
  const sum = rows.reduce((s, r) => s + r.p, 0);
  const target = Math.min(sum, 1 - UNNAMED_RESERVE);
  const scale = sum > 0 ? target / sum : 0;
  const kept = rows.slice(0, 100).map((r) => ({ id: r.id, p: r.p * scale })).filter((r) => r.p >= MIN_P / 2);
  perPlaceTop.push(kept);
  for (const r of kept) probabilities.push({ universe_id: r.id, place, p: Math.round(r.p * 1e6) / 1e6 });
}
// top10: greedy by probability, one game per place
const used = new Set<number>();
const top10: number[] = [];
for (let place = 1; place <= PLACES; place++) {
  const cand = (perPlaceTop[place - 1] ?? []).find((r) => !used.has(r.id));
  if (cand) { top10.push(cand.id); used.add(cand.id); }
}
if (top10.length < PLACES) {
  for (const r of last.rows) {
    if (top10.length >= PLACES) break;
    if (!used.has(r.universe_id)) { top10.push(r.universe_id); used.add(r.universe_id); }
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
