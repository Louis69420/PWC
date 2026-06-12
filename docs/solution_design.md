# Solution design — GameVault Acquisition Radar

## Two-layer architecture — model where data is deep, transparency where it is thin

1. **Review-level sentiment engine (ML).** 14,610 labelled Steam reviews are deep enough to train on. The engine scores *any* review text — including unlabelled feedback from forums, other storefronts, or console communities — so the approach scales beyond Steam's built-in labels. Output: a 0–1 sentiment score per review.
2. **Game-level acquisition scorecard (transparent index).** With only 23 games (18 with ≥100 reviews) a game-level ML model would not work. Instead each title gets a weighted composite of: sentiment health, momentum (last-6-months vs lifetime — the "lasting vs buzz" signal), engagement depth (median hours played), and community validation (helpful votes). Weights are adjustable in the dashboard, and thin-coverage titles carry a confidence flag.

