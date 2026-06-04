# Scenarios: Where We Started, and Where We're Going

<small>4 Jun 2026 - Hannah Westra</small>

When we first introduced scenarios in PyRIT, the pitch was simple: most of our users were assembling the same set of pieces — a target, a curated dataset of objectives, a few attack strategies, a scorer — and running them in a loop. Doing that by hand worked, but it didn't compose well. Two people running the "same" red-teaming pass could end up with subtly different setups, and there was no clean unit of work to point at when you wanted to say, "run this configuration, then compare it against that one." Scenarios were our answer: a single object that bundles objectives plus attack strategies plus scoring, that you can run end-to-end with `await scenario.run_async()` and that produces a `ScenarioResult` you can save, share, and diff.

The first scenarios — `FoundryScenario` (now `RedTeamAgent`), `Encoding`, and a starter `ContentHarms` — shipped around v0.10.0 / v0.11.0, alongside the `Scenario`, `AtomicAttack`, and `ScenarioStrategy` core. Since then we haven't really paused to write about scenarios on the blog, even though a lot has changed under the hood. This post is the catch-up.

## Sharpening the core (v0.13.0)

A lot of the work in v0.13.0 was making the scenario primitives behave the same way the rest of PyRIT does — fewer special cases, more shared infrastructure, and a clearer separation between *what an attack is* and *how it's wired into a scenario*.

**`AttackTechnique` is the new unit of composition.** Previously, an `AtomicAttack` wrapped a fully-constructed attack strategy and an `AtomicAttack` was the thing scenarios composed. That worked, but it conflated "the strategy" with "everything you need to instantiate the strategy" — every scenario had to know how to build its own attacks from raw configuration. The [`AttackTechnique` abstraction](https://github.com/microsoft/PyRIT/pull/1592) splits those apart. A technique now carries the attack strategy plus its `seed_technique` configuration as one self-describing bundle, and the scenario composes techniques rather than re-implementing the wiring each time. This is a breaking change — if you had your own scenario subclass calling `AtomicAttack(attack=...)`, the `attack=` keyword is now deprecated in favor of `attack_technique=`. The old signature still works through v0.16.0.

**`AttackTechniqueRegistry` is the catalog.** Once techniques are first-class objects, they need a place to live so the CLI, the scenario, and your notebook can all discover them the same way. [PR #1611](https://github.com/microsoft/PyRIT/pull/1611) introduced `AttackTechniqueRegistry`: a central, tag-queryable catalog of every attack technique known to PyRIT. Scenarios pick from it (via `TagQuery.any_of("default")`, for example), the CLI lists from it, and you can register your own. The registry is what lets new scenarios declare *what techniques they support* in a single line instead of hand-rolling an enum.

**Attack arguments are standardized.** [PR #1608](https://github.com/microsoft/PyRIT/pull/1608) was a follow-up cleanup: every attack strategy now takes the same shape of constructor args, which is why composing them inside techniques actually feels boring instead of bespoke.

**Smaller polish** that's worth knowing about:

- `include_baseline` moved from the `Scenario` constructor to `initialize_async` ([#1700](https://github.com/microsoft/PyRIT/pull/1700)) — you decide whether to run the baseline at run time, not at construction time.
- `BASELINE_POLICY` was renamed to `BASELINE_ATTACK_POLICY` ([#1763](https://github.com/microsoft/PyRIT/pull/1763)) so the name actually says what it controls.
- The `FoundryScenario` alias is gone ([#1623](https://github.com/microsoft/PyRIT/pull/1623)); use `RedTeamAgent`.
- And finally — possibly the most useful change for new users — we added [scenario doc pages for the seven scenarios that didn't have them](https://github.com/microsoft/PyRIT/pull/1558), and migrated the old cookbooks into the main docs tree. There's now a clean landing surface at [`doc/code/scenarios/`](../code/scenarios/0_scenarios.ipynb).

## Better reporting (v0.14.0)

v0.14.0 builds on that foundation and pushes on what you actually *see* after a scenario finishes.

**Attribution that survives resumes.** [Better Scenario Tracking](https://github.com/microsoft/PyRIT/pull/1758) ties every persisted `AttackResult` row back to the scenario run that produced it, with a proper attribution model in memory. The practical payoff: if you stop a long-running scenario and resume it with `scenario_result_id=...`, the analytics line up. Re-runs don't double-count. Cross-run comparisons stop being a manual labeling exercise.

**Sorted per-group breakdown.** Scenario printers now [sort the per-group results by success rate](https://github.com/microsoft/PyRIT/pull/1809), so the categories the target is most vulnerable on float to the top of the report instead of being buried mid-list.

**`SequentialAttack` as a compound primitive.** A new building block ([#1819](https://github.com/microsoft/PyRIT/pull/1819)) that runs a list of attacks in order with a configurable `SequenceCompletionPolicy` — try them all, or stop on the first success / first decisive result. Handy any time you want to stack attacks in priority order (e.g. try a cheap one first and only escalate to an expensive multi-turn one if needed).

## Adaptive scenarios (also v0.14.0)

This is the section that's been a long time coming.

### The problem with running everything against everything

The existing scenarios are exhaustive by design: pick N techniques and M objectives, and you run `N × M` attacks. That's the right default when you're benchmarking — you want to know how every technique fares on every objective. But it's also expensive, and in real red-teaming operations it doesn't match how we actually think. We *know* certain techniques tend to dominate for certain harm categories. Spending the same budget on techniques that have never landed against a given category, on every single objective, is mostly burning tokens to re-confirm what we already knew.

The trick is doing something about it without giving up reproducibility, without bolting in a custom scheduler, and without breaking the scenario abstraction that everything else in PyRIT relies on.

### Enter `TextAdaptive`

`TextAdaptive` is a new scenario that, for each objective, picks up to `max_attempts_per_objective` techniques in priority order, runs them one at a time, and stops as soon as one succeeds. Budget goes from `O(techniques × objectives)` to `O(max_attempts × objectives)` — and you choose `max_attempts`.

Concretely: say you select five techniques (`prompt_sending`, `crescendo`, `tap`, `pair`, `encoding`) against a dataset of misinformation objectives. Today, every technique runs against every objective. With `TextAdaptive` and `max_attempts_per_objective=3`, for each new misinformation objective the bandit asks memory which of those techniques have been working lately on similar runs, picks the top three (with some exploration), and stops as soon as one of them succeeds. Techniques that have historically failed against misinformation slide to the back of the queue without you maintaining a hand-written priority list.

```python
from pyrit.scenario.scenarios.adaptive import (
    TextAdaptive,
    EpsilonGreedyTechniqueSelector,
)

scenario = TextAdaptive(
    selector=EpsilonGreedyTechniqueSelector(epsilon=0.3, random_seed=42),
)
scenario.set_params_from_args(args={"max_attempts_per_objective": 5})
await scenario.initialize_async(objective_target=target)
result = await scenario.run_async()
```

The interesting part is *how* the techniques get picked.

### The selector: epsilon-greedy, Laplace-smoothed, optimistic on new techniques

`EpsilonGreedyTechniqueSelector` is a small bandit. For each objective it asks: *given what we've seen historically, which techniques should I try first?*

- With probability ε it **explores** — picks uniformly at random from the available techniques. Default ε is 0.2.
- Otherwise it **exploits** — picks the technique with the best Laplace-smoothed estimate, `(successes + 1) / (decided + 1)`. The +1 smoothing means an unseen technique starts at an optimistic `1.0` instead of an undefined `0/0`, so brand-new techniques are eligible to be tried instead of getting permanently ranked below techniques with any history.
- Ties are broken by random choice over the tied winners (using the same per-decision RNG), so the selector doesn't keep deterministically picking the first-listed technique when several look equally good.

If you supply a `random_seed`, picks are deterministic for a given objective and run context: a per-decision RNG is derived from `SHA-256(seed | objective)`, so resumes pick up where they left off without re-shuffling. Without a seed, exploration is intentionally nondeterministic.

### Stateless, memory-backed, and resumable

The thing that makes this approach feel native to PyRIT, rather than tacked on, is that the **selector doesn't hold counts in memory**. It queries `MemoryInterface` for the historical `AttackResult` rows that match its scope, computes success rates on the fly, and forgets them. That sounds expensive but it's the right call for three reasons:

1. **Learning accumulates across runs by default.** Each adaptive attempt is persisted to memory with a label identifying which technique ran. The next run — yours or someone else's against the same memory backend — reads those rows when scoring techniques, so over time the bandit's estimates reflect what's actually been working in your environment.
2. **Resumes use the same machinery.** Hand a `scenario_result_id` to an existing run and the regular scenario-resume flow skips objectives that already completed. For the objectives that haven't, the selector reads memory the same way it would on a fresh run, so a seeded resume picks the same techniques it would have picked originally.
3. **You decide what the selector learns from.** A `SelectorScope` controls whether to query across all history (the default), the current run only, results from a specific set of scenario / attack classes (`attack_classes`), a specific set of `targeted_harm_categories`, or any extra label filter you want to add. If you want to keep image and text bandits independent down the road, pin `attack_classes`. If you want every run to start fresh, use `SelectorScope.current_run()`.

One thing worth being explicit about: the selector picks all `max_attempts_per_objective` techniques up front for each objective, before the first attempt runs. So within a single objective, attempt 2 isn't re-ranked based on attempt 1's outcome. Online re-ranking between attempts in the same objective would be a reasonable next step; today, the granularity is per-objective.

### Baseline is preserved

Adaptive selection makes results harder to compare against "just send the prompt" if you don't think about it carefully — the bandit's job is to look better than random. To keep an honest comparison point, `prompt_sending` is **excluded from the adaptive technique pool** and instead runs as the baseline via `BASELINE_ATTACK_POLICY=Enabled`. By default, you see the no-attack number alongside the adaptive number, so the lift the adaptive scenario is actually buying you is right there in the report. (You can still opt out per-run with `initialize_async(include_baseline=False)` if you don't want it.)

### How the pieces fit together

```{mermaid}
flowchart LR
    SCENARIO["TextAdaptive (Scenario)"]
    DISPATCHER["AdaptiveDispatchAttack (one per dataset, shared selector)"]
    TECH["AttackTechnique[]"]
    MEMORY[("MemoryInterface (AttackResult rows)")]
    SELECTOR["EpsilonGreedyTechniqueSelector (stateless)"]

    SCENARIO -->|one AtomicAttack per dataset| DISPATCHER
    DISPATCHER -->|ask for top-K per objective| SELECTOR
    SELECTOR -.->|query historical success rates| MEMORY
    DISPATCHER -->|run chosen techniques in priority order, stop on success| TECH
    TECH -->|persist results with technique label| MEMORY
    MEMORY -.->|feed future selections| SELECTOR
```

The dispatcher (`AdaptiveDispatchAttack`) is a regular `AttackStrategy`: per-objective it asks the selector for the ordered technique list, then loops through them in priority order with its own per-attempt logic — merging the technique's `seed_technique` into the seed group, stamping a memory label for the chosen technique, and breaking out as soon as one attempt succeeds. The outer `AttackResult` it returns is a fresh copy of the winning inner result with the per-attempt trail stamped onto `metadata`, so the adaptive trail is recoverable without inventing a new result type. Inner techniques persist their own raw rows the way they always have. That last part is the important one: adaptive runs look like normal runs in memory and in your dashboards. Everything that already works with scenarios (the printers, the analytics, the CLI, the resume flow) keeps working without changes.

### What's not in this release

`AdaptiveScenario` is modality-agnostic on purpose. The text subclass is the one shipping in v0.14.0 — image and audio variants are scaffolded and on the roadmap, but they need their own attack-technique catalogs to be useful, and we'd rather ship them when the technique pool is there than have a one-attack adaptive run that doesn't really *have* anything to adapt. If you want to experiment with your own modality or your own selector strategy in the meantime, `TechniqueSelector` is a small protocol with one method — building a contextual bandit, a Thompson sampler, or whatever else you want to try is a hundred-or-so lines.

## Where to go next

- The scenarios docs landing page: [`doc/code/scenarios/`](../code/scenarios/0_scenarios.ipynb).
- The walkthroughs for running scenarios end-to-end: [`pyrit_scan`](../scanner/1_pyrit_scan.ipynb) and [`pyrit_shell`](../scanner/2_pyrit_shell.md).
- The adaptive scenarios notebook (shipped alongside `TextAdaptive`) is the fastest way to see the bandit in action with a real target.

If you've got ideas for new techniques to register, a selector strategy you want to try, or a scenario for a modality we don't cover yet — those are all great places to contribute. The whole point of the v0.13.0 / v0.14.0 work was making scenarios composable enough that adding the next thing doesn't require rewriting the last one.

Thanks for reading!
