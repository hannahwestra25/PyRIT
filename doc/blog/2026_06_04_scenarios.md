# Scenarios: Where We Started, and Where We're Going

<small>4 Jun 2026 - Hannah Westra</small>

When we first introduced scenarios in PyRIT a few releases back, the pitch was pretty simple: most of the red teamers we talked to were assembling the same set of pieces — a target, a curated dataset of objectives, a few attack strategies, a scorer — and running them in a loop. The framework gave them all the right Lego bricks, but every team was clicking those bricks together by hand. That made the everyday workflow harder than it needed to be: there was no clean unit of work to rerun after a model update, hand to a teammate, or diff against last quarter's results.

Enter scenarios. A scenario is a **pre-packaged red-teaming playbook** — a single object that bundles a curated set of objectives, a set of attack techniques to try against them, and the scoring + reporting logic to make sense of the results. You can run a scenario two ways: instantiate it in a notebook and `await scenario.run_async()` directly through the framework, or invoke it from the scanner CLI with `pyrit_scan` (or `pyrit_shell` for interactive exploration). Either way you get back a `ScenarioResult` you can save, share, and diff. The first batch — `RedTeamAgent` (originally `FoundryScenario`), `Encoding`, and a starter `ContentHarms` — shipped around 0.10.0 / 0.11.0.

We haven't really paused to write about scenarios on the blog since then, even though a *lot* has changed. This post is the catch-up: what scenarios look like today, what got sharpened in 0.13.0 and 0.14.0, and the new adaptive scenarios that landed in 0.14.0.

## What's in a scenario

Cracking one open, every scenario bundles five things:

- **Techniques — the *how*.** How are we going to attack? Maybe we just send the prompt directly. Maybe we wrap it in a role-play prompt template. Maybe we escalate over multiple turns with Crescendo or TAP. Techniques include the attack strategy plus its converters, jailbreak templates, and adversarial-chat configuration — basically all the knobs that affect how the attack is crafted and delivered.
- **Datasets — the *what*.** What harmful content are we testing for? Hate speech, violence, fairness, leakage, scam content. Each scenario ships with curated datasets that match its scope.
- **Strategies — the runtime knob.** Each scenario exposes a named set of strategies (e.g. `default`, `single_turn`, `multi_turn`, `light`) that pick which techniques actually run on this invocation. `--strategies default` is how a red teamer says "just the quick subset"; `--strategies multi_turn` is how they say "give me the harder ones." Strategies are built dynamically from the technique registry per-scenario, so adding a new tagged technique just shows up under the right strategies automatically — no scenario edit needed.
- **Scoring and reporting.** Every response flows through scoring, and the printer rolls everything up into a readable summary.
- **Memory persistence.** Every prompt, response, and result gets persisted so you can come back later, compare runs, or pick up where you left off.

Scenarios can also opt in to running a **baseline pass** — sending the raw, unmodified prompts as a control group so the report can show "what the model does without any attack" next to "what the model does once attacked." Baseline behavior is generic to the `Scenario` base class (toggled per run via `include_baseline` and governed by `BASELINE_ATTACK_POLICY`); individual scenarios just decide whether it makes sense for their workload.

The whole point is that you don't have to wire any of this up yourself. Pick a scenario, point it at a target, and the scenario handles the rest.

Concretely, the simplest run to create a RapidResponse scenario from the CLI looks like:

```bash
pyrit_scan airt.rapid_response --target my_target
```

<!-- TODO IMAGE: Capture a terminal session running `pyrit_scan airt.rapid_response
     --target my_target` (or a similar short scenario). The most useful frame
     shows the command at the top, the initializer log lines loading
     techniques / targets / datasets, and the first technique progress bars
     starting up. The final summary is already captured in
     2026_06_04_rapid_response_output.png — this image is about the
     "what does it look like to kick one off" moment. Save to
     doc/blog/2026_06_04_scan_run_scenario.png and uncomment the line below.
![pyrit_scan launching a scenario from the CLI](2026_06_04_scan_run_scenario.png)
-->

That one command does a lot. Before the scenario itself runs, **initializers** (`PyRITInitializer` subclasses such as `ScenarioTechniqueInitializer`, `TargetInitializer`, and `LoadDefaultDatasets`) populate the registries — every technique factory lands in `AttackTechniqueRegistry`, every configured target in `TargetRegistry`, every default dataset for the chosen scenario in memory. Only then does the CLI look up `airt.rapid_response` in the scenario registry, resolve `my_target` against `TargetRegistry`, instantiate `RapidResponse`, and call `run_async()`. You get back a `ScenarioResult` persisted to memory and pretty-printed at the end. No notebook glue required, and the same scenario class is what you'd `await` in a notebook if you preferred to drive it from Python.

## The scenarios you can run today

There are five flavors in the catalog right now. You can also bring your own — the abstractions are designed to be subclassed.

**Foundry — `RedTeamAgent`.** The integration with [Azure AI Foundry's Red Teaming Agent](https://learn.microsoft.com/en-us/azure/foundry/concepts/ai-red-teaming-agent). Organized by complexity (easy / moderate / difficult) rather than harm type. Easy = converters like Base64, ROT13. Difficult = multi-turn attacks like TAP and Crescendo. Built on HarmBench with 25+ techniques.

**Garak — `Encoding`.** Inspired by the [Garak](https://github.com/leondz/garak) project. Very focused: can the model be tricked into decoding and repeating harmful content? Tests 17 encoding schemes (Base64, Braille, Morse, Leet Speak, …) against slur terms and XSS payloads. Single-turn only, with a custom `DecodingScorer`. Niche but important for encoding-bypass vulnerabilities.

**Benchmark — `AdversarialBenchmark`** *(new in 0.14.0)***.** Not about testing a target — about comparing adversarial models. Feed it multiple red-teaming models and it measures which is most effective at generating attacks. No baseline run. Useful for "which adversarial model should we use?"

**AIRT — seven scenarios** built by the AI Red Team for real-world harm testing, each focused on one domain:

- `Jailbreak` — jailbreak templates (skeleton key, role-play, many-shot)
- `Cyber` — malware generation
- `Leakage` — PII, training-data, IP leaks
- `Psychosocial` — mental-health crisis, fake-therapist scenarios
- `Scam` — phishing and fraud generation
- `RapidResponse` *(new in 0.14.0)* — broad starter scan; gets its own paragraph below

**Adaptive — `TextAdaptive`** *(new in 0.14.0)***.** A new family of scenarios that pick techniques on the fly based on what's worked before. Big enough to get its own section [below](#adaptive-scenarios-0140).

### A closer look: `RapidResponse`

`RapidResponse` is the broadest of the AIRT scenarios — a comprehensive sweep across the most common techniques and the full AIRT harm-category catalog. It exists for the moment when you've onboarded a new model (or a new model release ships, or a new vulnerability hits the news) and the first question is "where are we exposed?" — not "which specific technique works." It's a natural jumping-off point: you run it to get a wide, shallow read on which categories the target handles well and which ones come back concerning, then pivot to the more focused AIRT scenarios (or to `TextAdaptive`) to dig into whatever came back interesting.

It runs seven core techniques (`prompt_sending`, `role_play`, `many_shot`, `TAP`, `crescendo_simulated`, `red_teaming`, `context_compliance`) across seven AIRT datasets (`airt_hate`, `airt_fairness`, `airt_violence`, `airt_sexual`, `airt_harassment`, `airt_misinformation`, `airt_leakage`). By default it sends four prompts per dataset, configurable with `--max-dataset-size`.

## What's improved in 0.13.0 and 0.14.0

A lot of the recent work has been less about building out our scenario library and more about making the underlying machinery sharper, so adding scenarios (and adding the *next* layer of capability on top of them) doesn't require rewriting the last one.

### A real abstraction for attacks in a scenario

Before 0.13.0 a scenario glued attacks together by hand, knowing how to construct each one and what arguments it needed. `AttackTechnique` replaced that with a single bundle: the attack strategy class, the `seed_technique` configuration that tells the attack how to mutate prompts (jailbreak template, encoding, role-play wrapper, etc.), plus any technique-specific defaults like which adversarial chat to use. A scenario now composes a *list* of `AttackTechnique`s and hands them to the executor — the scenario doesn't need to know the internals of TAP versus Crescendo versus a converter-based attack, just that it has techniques to run. Standardized attack arguments shipped in the same release, which is what lets every technique constructor speak the same dialect.

### A catalog those techniques live in

`AttackTechniqueRegistry` is where techniques register themselves with metadata: name, description, tags like `default` / `single_turn` / `multi_turn` / `light`, modality, what kinds of targets they work against. Scenarios pull techniques out via tag queries — `TagQuery.any_of("default")`, `TagQuery.all_of("multi_turn", "text")` — instead of importing each one by name. The scanner CLI uses the same registry to list and describe what's available. Population of the registry itself is handled by initializers (the canonical one is `ScenarioTechniqueInitializer`); to add your own technique, you write a factory and register it with a tag, and every scenario that queries by that tag picks it up for free.

The full path — from an executable attack algorithm all the way to a `ScenarioResult` — looks like this:

```mermaid
flowchart TB
    AS["AttackStrategy"]
    SC["seed configuration"]
    AT["AttackTechnique"]
    AS --> AT
    SC --> AT

    AT -->|"registered (with tags)"| Reg["AttackTechniqueRegistry"]
    Reg -->|"TagQuery"| Strat["ScenarioStrategy"]

    Strat -->|"--strategies"| Sc["Scenario<br/>(e.g. RapidResponse)"]
    DS[("scenario datasets")] --> Sc

    Sc -->|"technique × dataset"| AA["AtomicAttack"]
    AA -->|"AttackExecutor"| Res["ScenarioResult"]
```

A few things worth pulling out:

- **`AttackStrategy`** — the executable attack class (e.g. `CrescendoAttack`, `TAPAttack`, `PromptSendingAttack`). The algorithm itself.
- **`AttackTechnique`** — wraps a strategy *plus* its seed configuration (templates, converters, adversarial chat) into the unit the registry tracks.
- **`ScenarioStrategy`** — a per-scenario enum (`default`, `single_turn`, `multi_turn`, `light`, …) built dynamically from the registry at import time, so the names you see on `--strategies` are always in sync with whatever techniques are currently tagged.
- **`AtomicAttack`** — the runnable pairing of one technique with one dataset. What the executor actually executes, and the unit at which results get tracked, resumed, and labeled in memory.

### Configuration from the CLI and from YAML

0.14.0 added a generic mechanism for setting scenario parameters at run time — both from `pyrit_scan` arguments (`--max-dataset-size 10 --strategies multi_turn`) and from YAML config files passed with `--config`. The same `set_params_from_args` plumbing is exposed in Python too, so a notebook user can stash their parameters in YAML and load the same config the CLI does. This is what made parameter-heavy scenarios like `TextAdaptive` (selector, epsilon, max attempts per objective, scope) feasible to drive from the CLI.

### Parallel execution within a scenario

0.14.0 reworked how atomic attacks fan out inside a single scenario run so independent objectives, techniques, and datasets actually run concurrently against the target (respecting the target's rate limits and the scenario's concurrency caps). For wide scenarios like `RapidResponse` — 7 techniques × 7 harm categories × N prompts — this is the difference between watching a progress bar for an hour and finishing in minutes.

### Attribution that survives runs

Better Scenario Tracking added a scenario-run ID that gets stamped onto every `AttackResult` row the run produces. That sounds small but unlocks a lot.

When you resume a partially-completed scenario (via `scenario_result_id`), the framework can ask memory "which objectives already have results for *this* run?" and skip them without double-counting. Cross-run analytics like "how did `RedTeamAgent` do on this target across our last ten scans?" stop needing manual labeling. The printer can roll results up to the correct scenario invocation instead of mixing in unrelated history sitting in the same database. And this is what makes the adaptive selector's cross-run learning trustworthy — it can scope its history queries cleanly through `SelectorScope`.

## Adaptive scenarios (0.14.0)

`RapidResponse` is thorough — but it's brute force. It runs every technique against every objective, and which can result in wasted attempts: maybe Crescendo works great on your target and `prompt_sending` never gets through. You're paying for the misses anyway in latency, API rate-limit budget, and adversarial-chat tokens — and on a wide scan against a real target those costs are not theoretical.

0.14.0 ships a new family of scenarios — **adaptive scenarios** — that fix exactly this by leaning on the per-technique **attack success rate (ASR)** the framework already records in memory. Today it's just one: `TextAdaptive`. Image and audio variants are scaffolded by a modality-agnostic base and will follow once their technique pools are deep enough to be useful.

The idea is simple: instead of running every technique against every objective, the scenario **picks which technique to try next per-objective based on ASR, learns from what's worked, and stops as soon as one succeeds**. Budget goes from `O(techniques × objectives)` down to `O(max_attempts × objectives)`, where `max_attempts` defaults to 3.

Three pieces make it work:

- **The registry** — the same `AttackTechniqueRegistry` from 0.13.0. It's the catalog of available techniques the scenario can pick from.
- **The selector** — the brain. By default, adaptive scenarios use `EpsilonGreedyTechniqueSelector`, which decides what to try next using an explore/exploit tradeoff: most of the time it picks the technique with the best historical ASR; some of the time it tries something random to make sure it isn't missing a better option. New techniques get a fair shot before the selector settles on favorites. The selector is pluggable — `TechniqueSelector` is a small protocol with one method, so you can drop in a contextual bandit, a Thompson sampler, or any other policy by passing your own implementation to the scenario's `selector` argument.
- **The ASR feedback loop** — every attempt gets persisted to memory with a label identifying which technique ran. Next time the selector is asked to pick, it queries memory for those rows and ranks techniques by their track record.

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

![TextAdaptive output: per-objective technique trail and the wins/picks/rate summary](2026_06_04_text_adaptive_output.png)

In practice, `RapidResponse` and `TextAdaptive` complement each other. Run `RapidResponse` first to map which harm categories your target struggles with; reach for `TextAdaptive` when you want the fastest path through whatever came back interesting. Both pull from the same technique registry and the same ASR history, so every scan you run sharpens the next.

## Where to go next

- The scenarios docs landing page: [`doc/code/scenarios/`](../code/scenarios/0_scenarios.ipynb).
- The end-to-end walkthroughs from the scanner side: [`pyrit_scan`](../scanner/1_pyrit_scan.ipynb) and [`pyrit_shell`](../scanner/2_pyrit_shell.md).
- The [adaptive scenarios notebook](../code/scenarios/3_adaptive_scenarios.ipynb) is the fastest way to see the bandit in action against a real target.

A few things on the roadmap that are worth flagging:

- **Scenarios in the GUI.** Today scenarios run from the framework or the scanner. We're working on bringing scenario configuration and result browsing into the PyRIT GUI so non-CLI users can run scans, inspect results, and compare runs visually.
- **More adaptive modalities.** `ImageAdaptive` and `AudioAdaptive` are scaffolded but waiting on their attack-technique catalogs to be deep enough that there's something meaningful to adapt over.
- **Bring-your-own selectors and techniques.** `TechniqueSelector` is a small protocol with one method — building a contextual bandit, a Thompson sampler, or whatever else you want is a hundred-or-so lines. New techniques register into `AttackTechniqueRegistry` the same way the built-in ones do.

That's the catch-up. Thanks for reading!
