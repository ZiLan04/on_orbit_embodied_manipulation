---
name: sim-driven-policy-iteration
description: Use when a simulated robot policy's success rate is inadequate and you must improve it from simulation feedback — reading closed-loop evaluation artifacts, stratifying failures, sizing the evaluation set so a delta is actually measurable, applying one intervention at a time, and avoiding the reward-curve and small-sample traps. Companion to `starlink-cube-vla-pipeline`.
---

# Improving a policy from simulation feedback

A repeatable loop for raising a measured success rate. The method matters more than
the specific intervention: the usual failure is not a bad idea, it is **drawing a
conclusion from a measurement too noisy to support it**, then building on that
conclusion.

Follow the steps in order. Step 0 is not optional.

## Step 0 — Make the measurement trustworthy before changing anything

**This is where most iterations go wrong.** A headline number like
"37.5% success" is nearly always reported without its uncertainty, and at small
episode counts the uncertainty is enormous.

Compute the interval, not just the point estimate. For `s` successes in `n`
episodes with proportion `p = s/n`:

```
standard error  SE = sqrt(p * (1 - p) / n)
95% interval    p ± 1.96 * SE
```

Worked example from this project — the real first result:

```
6 successes / 16 episodes  ->  p = 0.375
SE = sqrt(0.375 * 0.625 / 16) = 0.121
95% interval = 0.375 ± 0.237  ->  [0.19, 0.61]
```

**A ±24 percentage-point interval cannot support any claim.** Reporting "37.5%"
without that interval is misleading, in either direction.

Rough sizing for a decision, comparing two arms:

| Claim you want to support | Episodes per arm |
|---|---|
| ~12 point difference (e.g. 37% vs 50%) | ~250 |
| ~5 point difference | ~1500 |

Two cheaper routes to a usable measurement — prefer them over blindly multiplying
episodes:

1. **Pair the evaluation.** Evaluate both arms on the *identical* seed set and
   identical initial conditions, then compare episode-by-episode rather than
   comparing two independent rates. Pairing removes the between-episode variance
   that dominates at small `n`. The evaluator takes a seed; reuse it across arms
   and record it in the run name.
2. **Stratify by the condition that actually drives outcome** (usually object
   position). If the object's position is the dominant factor, a balanced
   stratified set of 60 episodes can be far more informative than 200 random ones.

Record, for every evaluation you intend to compare: episode count, seed, the
checkpoint or dataset version, and the exact command. An unrecorded evaluation is
not evidence.

## Step 1 — Read the artifacts

Each closed-loop run directory contains:

| File | What it gives you |
|---|---|
| `summary.json` | `episodes`, `successes`, `success_rate`, and a per-episode list with `success`, `steps`, and the object's position |
| `actions.jsonl` | every action applied, per step — for checking saturation, sign conventions, and whether the policy moves at all |
| `closed-loop.mp4` | what actually happened visually |
| `evaluator.log` | scene construction, IK errors, exceptions |
| `policy-server.log` | whether the server loaded the intended checkpoint and what metadata it advertised |

Read `summary.json` first, then the per-episode entries — the structure of the
failures is in there.

## Step 2 — Find the structure in the failures

A success rate alone tells you nothing actionable. Turn it into a condition.

- **Stratify by object position.** `summary.json` carries a position per episode.
  Bucket by region and compute success per bucket. A pattern like "succeeds
  centrally, fails at the edges" points at reach or workspace limits; a uniform
  spread points at the policy or the data.
- **Compare failure step counts.** Episodes ending at the step cap usually mean
  the policy never converges; episodes ending early usually mean it drove
  somewhere wrong and stopped. In verified runs, successes ended around step
  104–124 against a cap of 128.
- **Check the action stream.** `actions.jsonl` reveals whether translation
  channels are saturated (clipping is hiding the policy's real intent), whether
  the gripper channel ever crosses its close threshold (if it never does, the
  gripper policy is dead), and whether the two action spaces are consistent.
- **Verify the arm moved at all.** End-effector path length across an episode
  distinguishes "policy is wrong" from "policy is disconnected".
- **Watch the video.** The frames settle arguments that numbers only hint at.

Write the finding as a falsifiable statement about a condition, e.g. "failures
concentrate where the object is more than N from the base; central episodes
succeed". Not "the policy is bad".

## Step 3 — State one hypothesis and one intervention

Change **one** thing. If you change two, you cannot attribute the delta.

The intervention menu available in this codebase, roughly in ascending cost:

| Intervention | Cost | Addresses |
|---|---|---|
| Evaluation-set fix (more episodes, paired seeds, stratified) | minutes | measurement noise, not the policy |
| Data filtering / labeling correction | minutes–hours | wrong or mislabeled targets |
| Data coverage expansion (wider object randomization, recovery states) | hours | failures concentrated in under-sampled conditions |
| Action horizon change | hours | policy plans too short/long for the task dynamics |
| Reward reshaping → retrain (RL only) | days | policy optimizes the wrong objective |
| Unfreezing more of the model, longer training | days | capacity or under-training |

Prefer the cheapest intervention that addresses the finding from Step 2. If the
finding is "measurement too noisy", the correct intervention is Step 0, not a
model change.

## Step 4 — Re-evaluate identically, then compare

- Same seed set, same episode count, same conditions as the baseline.
- Same metric definition.
- Record the delta **with its interval**, not as a bare number.
- If the intervals overlap heavily, the result is **inconclusive** — say so, and
  either gather more episodes or pick a bigger effect to chase. Do not narrate an
  inconclusive result as an improvement.

One completed pass of this loop — evaluate, diagnose, change one thing,
re-evaluate — is the unit of progress.

## Guardrails

**Never judge a run by its training reward curve.** A reward curve that rises
smoothly while the task never succeeds is the exact failure mode already observed
on this project: `ep_rew_mean` climbed to +783 while task success stayed at 0/25
and the deterministic error sat frozen at 0.2267 m. The policy had learned to farm
a term that rewarded being *far but well-oriented*. Only deterministic evaluation
with a success criterion decides.

**Verify gated reward terms actually fire.** When a reward contains threshold
bonuses, check on a known-successful trajectory that each term contributes
something non-zero at least once. In that same failed experiment, similarity
topped out at 0.92–0.93 against a 0.95 threshold, so three bonus terms
contributed **exactly zero gradient** — they were inert the entire run. A term
that is always zero is not a weak signal, it is no signal.

**Prefer dense progress terms over threshold gates.** Reward the *reduction* of
error each step (a difference of errors, which is dense everywhere) and treat
bonuses as a supplement, not the mechanism.

**Keep a reference trajectory.** A scripted or otherwise known-good episode is
the ground truth for three things: seeding behavior cloning, sanity-checking a
reward, and confirming that a failure is the policy's fault rather than the
environment's.

**Isolate variables.** Running a new environment and a new policy at once means a
failure cannot be attributed. Exercise a new environment with a known-good driver
first.

**Do not edit deployed code to fix a bug.** The deployed tree is a sync target;
fix the source and re-deploy, or the fix disappears at the next sync.

## Worked example (a real analysis from this project)

**Baseline**: VLA action head, `closed-loop-randomized-64-v3`, 58/64 = 90.6%,
95% CI [0.84, 0.98]. Large enough to support a ~10-point claim. (An earlier
16-episode variant scored 6/16 = 37.5% with a ±24-point interval — that number
could not support anything, which is the point of Step 0.)

**Step 2 — structure in the failures.** All 6 failures hit the **step cap of 128**,
while every success finished in 55–68 steps (mean 62.6). So the failure mode is
*stalling*, not a wrong-but-terminating action. Stratifying by object position:

```
x in [-0.46, -0.44):  12/16 = 75.0%
x in [-0.44, -0.42):  36/37 = 97.3%
x in [-0.42, -0.40):  10/11 = 90.9%

edge  (x < -0.44):    12/16 = 75.0%
middle(x >= -0.44):   46/48 = 95.8%
Fisher exact p = 0.030
```

Two checks before believing it:

- **Rule out a sampling artifact.** Several failures sat at x ≈ -0.4487, suspiciously
  close to the sampled minimum of -0.4493. If the sampler piled episodes at its
  boundary, the cluster would be an artifact of the sampler, not the policy.
  The x values turned out evenly spread (only 7 of 64 below -0.448), so the
  cluster is real.
- **Check whether the teacher has the same trouble.** The scripted oracle scores
  **32/35 = 91.4%** on a randomized validation set and in at least one episode
  needed **169 steps with a regrasp** — more than the policy's 128-step cap.

**Step 3 — the finding, and the outcome of testing it.** The step-cap hypothesis was
tested and **refuted**. Seeded runs at cap 128 and cap 256 gave byte-identical
results: same 6 failures, same 58/64, and the failures genuinely consumed the
extra budget (128 → 256 steps). Successes were identical too (55–68 steps, mean
62.6), which also establishes that **a seeded evaluation here is fully
deterministic** — paired comparison has zero confound. Doubling a budget that is
already ~4× the typical success duration changed nothing.

Refuting it cost 12 minutes and was worth more than another day of guessing. That
is the point: a cheap falsifiable test beats a plausible story.

**What the refutation then exposed.** Reading the per-step `actions.jsonl` for
successes and failures side by side showed something the success rate hid:

- the applied action is **nearly identical in every episode** (translation norm
  1.0531–1.0533 across all 64, action[0] ≈ 0.3154 throughout);
- the gripper is **closed from step 0 in every episode**;
- successes stop after ~3 cm of end-effector travel, failures continue in the
  *same direction* for ~13 cm;
- so success and failure are decided by the **environment's done condition**, not
  by any difference in policy behaviour.

A near-constant action stream is not a policy that "fails at the edge" — it is a
policy that is barely modulating its output at all. Check this before blaming
coverage or capacity.

**Why that connects to a data problem, not a model problem.** The checkpoint's
`dataset_statistics.json` has action ranges of x ∈ [0.2895, 0.3167] and
**z with `min == max == 1.0`**. Under `min_max` un-normalization a channel with
`min == max` maps every input to the same value, so that channel's output is
discarded entirely. The applied action's z is exactly 1.0 in every recorded step,
consistent with that. Before concluding anything about policy quality, verify the
normalization statistics are non-degenerate, and dump the **pre-un-normalization**
model output to see whether it is saturated.

**Lesson.** When success rate stops responding to the obvious knobs, stop tuning
and start reading the action stream. A metric that is stable across 64 episodes
with different initial conditions is a signal that the actions themselves carry
little information — which points at the data contract, not at the policy.

**Also unresolved and worth knowing:** the cube positions reported in
`summary.json` (x ≈ −0.41 … −0.45) do not line up with the end-effector positions
in `actions.jsonl` (x ≈ −0.35 … −0.37) under a simple offset. Before using either
to argue about workspace coverage, establish which frame each is in. An earlier
attempt in this project to compare them directly produced a misleading conclusion.
