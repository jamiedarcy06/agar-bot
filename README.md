# agar-bot

A heuristic bot for an Agar.io-style competition I did as part of a uni course. Each team wrote a `choose_move(game) -> MovePlayer` function that got driven by a real per-object physics engine (splitting, viruses, merge cooldowns, vision limited to nearby entities) and played 8-way free-for-all matches against everyone else's bots.

This is just the bot logic — the engine itself was a course-specific package (`agario-kit`) that isn't public, so this file won't run standalone. It's here as a record of the final version, not a runnable project.

## How it works

No machine learning, just a scored steering vector recomputed every round:

- Every visible blob is bucketed as a threat or as prey based on its size relative to yours, weighted by distance and blended into a single flee/chase direction
- Nearby swarms (several fragments from one player) are handled separately from lone blobs — worth eating piece by piece if you're big enough, worth fleeing as a group otherwise
- Split-attacks: lunge-and-eat a target if the post-split fragment can actually reach it in time and nothing bigger is close enough to punish the temporary split
- Viruses are baited when you're big enough to survive popping them (and it drops something dangerous on top of a target), avoided otherwise
- When multiple threats are converging at once, a discrete search over candidate escape headings picks the direction with the best worst-case margin over a short lookahead, with hysteresis so it doesn't flicker between two similar options every round
- Wall/corner-aware routing so fleeing doesn't just run in a straight line into a dead end
- Falls back to food-seeking, weighted per food item by distance to your nearest blob, when nothing else is going on

## History

This was the ~101st iteration of the bot. Most of that was small, targeted fixes — a specific death from a match replay traced back to one bad assumption (a stale distance cutoff, a threat-detection gap, a corner the escape logic didn't account for), fixed, then A/B tested against the previous version over hundreds of simulated matches before keeping it. The original had all of that history written into the docstrings; I stripped it out for this repo since it's meaningless without the match data behind it.
