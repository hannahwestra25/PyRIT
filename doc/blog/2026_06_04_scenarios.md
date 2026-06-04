# Scenarios: Where We Started, and Where We're Going

<small>4 Jun 2026 - Hannah Westra</small>

When we first introduced scenarios in PyRIT a few releases back, the pitch was pretty simple: most of the operators we talked to were assembling the same set of pieces — a target, a curated dataset of objectives, a few attack strategies, a scorer — and running them in a loop. The framework gave them all the right Lego bricks, but every team was clicking those bricks together by hand. Two people running the "same" red-teaming pass against the same model could end up with subtly different setups. There was no clean unit of work to point at when you wanted to say, "run this exact configuration, then compare it against that one."

Scenarios were our answer. A scenario is a **pre-packaged red-teaming playbook** — a single object that bundles a curated set of objectives, a set of attack techniques to try against them, and the scoring + reporting logic to make sense of the results. You point one at a target and call `await scenario.run_async()`. Out comes a `ScenarioResult` that you can save, share, and diff. The first batch — `RedTeamAgent` (originally `FoundryScenario`), `Encoding`, and a starter `ContentHarms` — shipped around v0.10.0 / v0.11.0.

We haven't really paused to write about scenarios on the blog since then, even though a *lot* has changed. This post is the catch-up: what scenarios look like today, what got sharpened in v0.13.0 and v0.14.0, and the new adaptive scenarios that landed in v0.14.0.

## Where scenarios fit: framework, scanner, GUI

It helps to put scenarios in context with the rest of PyRIT first. There are three layers:

1. **The framework** — the Python library. Targets, attacks, converters, scorers, memory. Maximum flexibility; every knob exposed.
2. **The scanner** — a CLI layer on top of the framework, with two commands: `pyrit_scan` and `pyrit_shell`. The whole idea is to abstract the framework knobs away for the common case. Operators repeat the same workflows constantly, and the scanner makes that repetition trivial. There are three things you pass it: a **scenario**, a **target**, and an optional **config**.
3. **The GUI** — the newest layer, currently focused on chat, conversation analytics, and converter exploration.

Scenarios are the unit of work that the scanner is built around. You run

```
pyrit_scan airt.rapid_response --target my_model --strategies default
```

and the scanner picks up the rapid-response scenario, applies the configuration, runs it against your target, and shows you the results. But the same scenario works fine straight from the framework — you can instantiate it in a notebook and call `run_async()` directly. That symmetry is what makes scenarios useful: the operator running the CLI and the developer iterating in a notebook are running the *same code path*, against the same datasets, with the same scoring.

## What's in a scenario

Cracking one open, every scenario bundles four things:

- **Techniques — the *how*.** How are we going to attack? Maybe we just send the prompt directly. Maybe we wrap it in a role-play scenario. Maybe we escalate over multiple turns with Crescendo or TAP. Techniques include the attack strategy plus its converters, jailbreak templates, and adversarial-chat configuration — basically all the knobs that affect how the attack is crafted and delivered.
- **Datasets — the *what*.** What harmful content are we testing for? Hate speech, violence, fairness, leakage, scam content. Each scenario ships with curated datasets that match its scope.
- **Scoring and reporting.** Every response flows through scoring, and the printer rolls everything up into a readable summary.
- **Memory persistence.** Every prompt, response, and result gets persisted so you can come back later, compare runs, or pick up where you left off.

The whole point is that you don't have to wire any of this up yourself. Pick a scenario, point it at a target, and the scenario handles the rest.

## The scenarios you can run today

There are four flavors in the catalog right now. You can also bring your own — the abstractions are designed to be subclassed.

🟢 **Foundry — `RedTeamAgent`.** The integration with [Azure AI Foundry's red-teaming library](https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/openai/red-team-tools). Organized by complexity (easy / moderate / difficult) rather than harm type. Easy = converters like Base64, ROT13. Difficult = multi-turn attacks like TAP and Crescendo. Built on HarmBench with 25+ techniques. The "throw everything at the wall" approach.

🟣 **Garak — `Encoding`.** Inspired by the [Garak](https://github.com/leondz/garak) project. Very focused: can the model be tricked into decoding and repeating harmful content? Tests 17 encoding schemes (Base64, Braille, Morse, Leet Speak, …) against slur terms and XSS payloads. Single-turn only, with a custom `DecodingScorer`. Niche but important for encoding-bypass vulnerabilities.

🟡 **Benchmark — `AdversarialBenchmark`.** Not about testing a target — about comparing adversarial models. Feed it multiple red-teaming models and it measures which is most effective at generating attacks. No baseline run. Useful for "which adversarial model should we use?"

🔴 **AIRT — seven scenarios** built by the AI Red Team for real-world harm testing, each focused on one domain:

- `Jailbreak` — jailbreak templates (skeleton key, role-play, many-shot)
- `Cyber` — malware generation
- `Leakage` — PII, training-data, IP leaks
- `Psychosocial` — mental-health crisis, fake-therapist scenarios
- `Scam` — phishing and fraud generation
- `ContentHarms` — general-purpose; covers hate, violence, fairness, sexual content
- `RapidResponse` — the urgent-question scenario, worth its own paragraph

### A closer look: `RapidResponse`

`RapidResponse` is the scenario you're most likely to reach for when something urgent comes up — a new vulnerability drops, a new jailbreak is trending on Twitter, and leadership wants to know whether the model is exposed.

It crosses two axes: **techniques × harm categories**. The technique side pulls from seven core techniques (`prompt_sending`, `role_play`, `many_shot`, `TAP`, `crescendo_simulated`, `red_teaming`, `context_compliance`). The harm side covers seven AIRT datasets (`airt_hate`, `airt_fairness`, `airt_violence`, `airt_sexual`, `airt_harassment`, `airt_misinformation`, `airt_leakage`). By default it sends four prompts per dataset, configurable with `--max-dataset-size`.

It runs a **baseline** first — raw prompts, no converters, no tricks — as the control group. Then it crosses every selected technique with every selected dataset and runs them in parallel through the execution engine. Results are grouped by harm category rather than by technique, because when leadership asks "are we exposed to hate speech?", you want the answer organized that way.

Strategies are also tagged for convenience: `default` is `prompt_sending + many_shot` (quick check), `single_turn` and `multi_turn` carve out the obvious subsets, and `light` is a fast sweep across five mostly-cheap techniques. So `pyrit_scan airt.rapid_response --target my_model --strategies default` gives you a quick two-technique pass across all seven harm categories; `--strategies multi_turn` hits the harder stuff.

## What's improved in v0.13.0 and v0.14.0

A lot of the recent work has been less about new scenarios and more about making the underlying machinery sharper, so adding scenarios (and adding the *next* layer of capability on top of them) doesn't require rewriting the last one.

**A real registry for techniques.** v0.13.0 introduced [`AttackTechnique`](https://github.com/microsoft/PyRIT/pull/1592) as a first-class abstraction (the attack strategy plus its `seed_technique` configuration as one bundle) and [`AttackTechniqueRegistry`](https://github.com/microsoft/PyRIT/pull/1611) as a central, tag-queryable catalog. Scenarios pick techniques from the registry via tag queries like `TagQuery.any_of("default")`, the CLI lists from it, and you can register your own. This is the foundation that everything below builds on. Attack arguments were [standardized in the same release](https://github.com/microsoft/PyRIT/pull/1608), which is what makes composing techniques inside scenarios feel boring instead of bespoke.

**Better reporting in v0.14.0.** [Better Scenario Tracking](https://github.com/microsoft/PyRIT/pull/1758) ties every persisted `AttackResult` row back to the scenario run that produced it, so resumes and re-runs line up correctly and cross-run comparisons stop being a manual labeling exercise. Scenario printers now [sort the per-group breakdown by success rate](https://github.com/microsoft/PyRIT/pull/1809), so the harm categories the target is most vulnerable on float to the top of the report. And [`SequentialAttack`](https://github.com/microsoft/PyRIT/pull/1819) shipped as a compound primitive — useful any time you want to stack attacks in priority order (try a cheap one first, escalate to expensive multi-turn only if needed).

**Smaller polish worth knowing about.** `include_baseline` moved from the `Scenario` constructor to `initialize_async` ([#1700](https://github.com/microsoft/PyRIT/pull/1700)) so you decide at run time, not construction time. `BASELINE_POLICY` was renamed to `BASELINE_ATTACK_POLICY` ([#1763](https://github.com/microsoft/PyRIT/pull/1763)) so the name says what it controls. The `FoundryScenario` alias was removed in favor of `RedTeamAgent` ([#1623](https://github.com/microsoft/PyRIT/pull/1623)). And — possibly the most useful change for new users — [scenario doc pages for the seven scenarios that didn't have them](https://github.com/microsoft/PyRIT/pull/1558) finally exist.

## Adaptive scenarios (v0.14.0)

`RapidResponse` is thorough — but it's brute force. It runs every technique against every objective, and most of those attempts are wasted. Maybe Crescendo works great on your target and `prompt_sending` never gets through. You're paying for the wasted ones anyway.

v0.14.0 ships a new family of scenarios — **adaptive scenarios** — that fix exactly this. Today it's just one: `TextAdaptive`. Image and audio variants are scaffolded by a modality-agnostic base and will follow once their technique pools are deep enough to be useful.

The idea is simple: instead of running every technique against every objective, the scenario **picks which technique to try next per-objective, learns from what's worked, and stops as soon as one succeeds**. Budget goes from `O(techniques × objectives)` down to `O(max_attempts × objectives)`, where `max_attempts` defaults to 3.

Three pieces make it work:

- **The registry** — the same `AttackTechniqueRegistry` from v0.13.0. It's the catalog of available techniques the scenario can pick from.
- **The selector** — the brain. `EpsilonGreedyTechniqueSelector` decides what to try next using an explore/exploit tradeoff: most of the time it picks the technique with the best historical success rate; some of the time it tries something random to make sure it isn't missing a better option. New techniques get a fair shot before the selector settles on favorites.
- **Attack success rate (ASR)** — the feedback loop. Every attempt gets persisted to memory with a label identifying which technique ran. Next time the selector is asked to pick, it queries memory for those rows and ranks techniques by their track record.

The genuinely powerful part is that **it learns across runs**. The selector is stateless — it doesn't hold counts in memory, it queries `MemoryInterface` every time. If you ran `RapidResponse` last week and TAP worked 60% of the time, the adaptive scenario knows that on day one. It doesn't rediscover what you already learned. Every scan your org runs makes the next adaptive scan smarter. `SelectorScope` is the escape hatch when you want to narrow the scope — restrict to the current run, a specific scenario class, or a specific set of harm categories.

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

`prompt_sending` is excluded from the adaptive pool and runs as the baseline comparison instead (via `BASELINE_ATTACK_POLICY=Enabled`), so you still see the honest "no-attack" number alongside the adaptive number in the report.

Bottom line: **`RapidResponse` tells you "here's how every technique did" against this target. `TextAdaptive` tells you "here's the fastest path to breaking this model."** Both are useful; you reach for them at different moments.

## Where to go next

- The scenarios docs landing page: [`doc/code/scenarios/`](../code/scenarios/0_scenarios.ipynb).
- The end-to-end walkthroughs from the scanner side: [`pyrit_scan`](../scanner/1_pyrit_scan.ipynb) and [`pyrit_shell`](../scanner/2_pyrit_shell.md).
- The adaptive scenarios notebook (shipped alongside `TextAdaptive`) is the fastest way to see the bandit in action against a real target.

A few things on the roadmap that are worth flagging:

- **Scenarios in the GUI.** Today scenarios run from the framework or the scanner. We're working on bringing scenario configuration and result browsing into the PyRIT GUI so non-CLI users can run scans, inspect results, and compare runs visually.
- **More adaptive modalities.** `ImageAdaptive` and `AudioAdaptive` are scaffolded but waiting on their attack-technique catalogs to be deep enough that there's something meaningful to adapt over.
- **Bring-your-own selectors and techniques.** `TechniqueSelector` is a small protocol with one method — building a contextual bandit, a Thompson sampler, or whatever else you want is a hundred-or-so lines. New techniques register into `AttackTechniqueRegistry` the same way the built-in ones do.

That's the catch-up. Thanks for reading!
