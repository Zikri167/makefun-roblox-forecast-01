# Roblox Up-and-Coming forecast — Monte Carlo rank diffusion

Entry for `roblox-up-and-coming-forecast-01`: a pure-code, deterministic
forecaster (Node 22, standard library only, no network, no model call).
It reads `/input/history.ndjson` + `/input/task.json` and writes
`/output/forecast.json`.

## Idea

One forecast is scored against every hourly read of a 72-hour window, so
the log score is maximised by the **time-averaged** distribution of each
game's place over the window. The entry simulates that distribution:

- **Rank random walk, per rank band.** From the history it fits, per band
  (1–10, 11–30, 31–60, 61+), the hourly drift and volatility of rank
  changes. Top-10 games drift down slowly and move ~2.3 ranks per ~6 h;
  lower bands diffuse faster and climb on average (chart refresh).
- **Rank momentum.** A game's recent rank velocity (recency-weighted fit
  over its history) persists into its next move (measured correlation
  ~0.22); each game carries a velocity that decays ~5% per hour.
- **Churn.** Games leave the chart (exit hazard per band, fitted) and
  fresh games enter at the bottom. When a creation date is known
  (house-recorder reads carry `details`), a game ages out at 28 days.
- **Playing counts are NOT used.** Measured correlation of log playing
  growth with rank change is ~-0.02, concurrent and one-step-ahead: the
  chart is a computed order, not a live count.
- 2000 deterministic Monte Carlo paths (fixed-seed PRNG) × 72 hourly
  steps, re-ranking by latent position each hour. Place probabilities are
  the simulated occupancy, scaled so each place sums to 0.70, leaving
  0.30 unnamed for games the simulation cannot name (fresh entrants).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Runs in ~9 s on 1 CPU. Two runs on the same history are byte-identical.

Tuning knobs (env, defaults are the tuned values):
`RBX_SIMS` (2000), `RBX_UNNAMED` (0.30), `RBX_VOL_MULT` (1.15),
`RBX_MOM_MULT` (1), `RBX_VDECAY` (0.95), `RBX_MIN_P` (0.004).

## Local validation

The kit ships no scorer, so a local one was built from the rules
(`ln max(p, 0.0001)` per place, unnamed share split over the chart)
and the practice answer was rebuilt from the public MIT archive
`Kenny-Dhub/roblox-snapshots` (the 10 reads in the practice window).
The scorer reproduces the house baseline exactly (-43.512039, 6.4 hits,
53.5 agreement).

| forecast | practice window (10 reads) | 2nd split: history to 5 Nov, window 5–7 Nov |
|---|---|---|
| house baseline (starter) | -43.512 | -56.911 |
| this entry | **-35.401** | **-41.703** |

Practice: typical probability per place 0.029 vs baseline 0.013.

## License

MIT (see LICENSE).
