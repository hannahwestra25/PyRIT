# TargetCapabilities 🤝 TargetConfig

## Summary

This document describes a plan to unify **TargetCapabilities** (what a target can do) with **prompt normalizers/adapters** (how to adapt when it can’t), without changing memory semantics or target protocol handling.

---

## Current State

- Targets each have **TargetCapabilities**
- **TargetCapabilities** are declarative flags (e.g. `supports_multi_turn`, `supports_system_prompt`) used mostly for validation and ad‑hoc branching
- Conversations are often manipulated to allow targets to interpret conversations
- **Message normalizers** adapt message lists (system squash, history squash, etc.)

### Problems

- TargetCapabilities and MessageNormalizers are disconnected
- Capability responses differ across targets, attacks, and converters
- No centralized normalization pathway
- Normalization is distributed and implicit:
  - Targets rebuild and reinterpret message lists
  - Each target performs its own implicit normalization

### Architectural Issues

- Type identity is conflated with capability semantics
- `PromptChatTarget` acts as both an inheritance base and a capability signal
- Capability semantics are split across multiple layers
- Conversation‑level transformations live on targets instead of memory/conversation layers

**Symptoms**:

- `isinstance` checks and target‑specific branching
- Inconsistent normalization behavior
- Duplicated capability logic across attacks, converters, and scorers

---

## Proposal: TargetConfiguration

A single object that specifies:

- What the target supports (**TargetCapabilities**)
- What to do when it doesn’t (**CapabilityHandlingPolicy**)
- How to adapt (**NormalizationPipeline**)

Each target defines defaults that can be overridden at creation time.

Consumers (attacks, converters, scorers) validate requirements at creation time.

---

## Current vs Proposed Flow

| Current Flow | Proposed Flow |
|-------------|---------------|
| Attack / Converter / Scorer | Attack / Converter / Scorer |
| Optional ad‑hoc capability checks | Capability validation at construction |
| `send_prompt_async` | `send_prompt_async` |
| Target‑specific prompt interpretation | `TargetConfiguration.normalize_async` |
| Implicit normalization | Explicit normalization pipeline |
| Target‑specific wire formatting | Target‑specific wire formatting |
| API call | API call |

---

## Components

### TargetCapabilities

Declarative, immutable description of what the target natively supports.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TargetCapabilities:
    supports_multi_turn: bool = True
    supports_system_prompt: bool = True
    supports_json_schema: bool = False
    supports_json_output: bool = False
    supports_editable_history: bool = False  # never adaptable
```

### CapabilityHandlingPolicy

Per‑capability behavior when the target **lacks** a capability.

```python
@dataclass(frozen=True)
class CapabilityHandlingPolicy:
    behaviors: Mapping[CapabilityName, UnsupportedCapabilityBehavior] = field(
        default_factory=lambda: {
            CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
            CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.RAISE,
        }
    )
```

- `ADAPT` — apply the corresponding normalizer (squash system prompt, squash history)
- `RAISE` — raise `ValueError` if the capability is needed but missing

### ConversationNormalizationPipeline

Ordered chain of `MessageListNormalizer[Message]` instances, built automatically from capability gaps + policy.

```python
_NORMALIZER_REGISTRY: list[tuple[CapabilityName, MessageListNormalizer[Message]]] = [
    (CapabilityName.SYSTEM_PROMPT, GenericSystemSquashNormalizer()),
    (CapabilityName.MULTI_TURN, HistorySquashNormalizer()),
]
```

### TargetConfiguration

Composes all three: capabilities, policy, and pipeline.

```python
class TargetConfiguration:
    def __init__(self, *, capabilities, policy, normalizer_overrides=None): ...

    async def normalize_async(self, *, messages: list[Message]) -> list[Message]:
        return await self._pipeline.normalize_async(messages=messages)

    def ensure_can_handle(self, *, capability: CapabilityName) -> None: ...
```

---

## Implementation PRs

### PR 1 — TargetCapabilities & CapabilityHandlingPolicy

Add the declarative data classes and `CapabilityName` enum. No behavioral changes.

### PR 2 — ConversationNormalizationPipeline & Normalizers

Add `GenericSystemSquashNormalizer`, `HistorySquashNormalizer`, and the pipeline that chains them. All existing normalizer behavior preserved.

### PR 3 — TargetConfiguration & TargetRequirements

Wire `TargetConfiguration` onto `PromptTarget`. Expose `target.configuration`. Add `TargetRequirements` for consumer‑side validation. Backward‑compatible: no target uses `normalize_async` yet.

### PR 4 — Integrate `normalize_async` into Target Send Path

Replace ad‑hoc normalization inside each target's `send_prompt_async` with a single call to `self.configuration.normalize_async(messages=...)`. Remove duplicated normalization code from targets. Preserve canonical memory semantics (memory always stores the **original** messages; normalization is ephemeral and only affects what is sent to the wire).

#### Problem

Every chat target independently:

1. Fetches conversation from memory
2. Appends the current message
3. Performs its own normalization (or none at all)
4. Converts to wire format and calls the API

Steps 1–3 are identical across targets, but each reimplements them. Some targets normalize (e.g. `AzureMLChatTarget` uses a pluggable `message_normalizer`), some pass messages through unmodified (e.g. `OpenAIChatTarget`). The `TargetConfiguration.normalize_async` pipeline is fully implemented but never called.

#### Approach

**Phase A — Add normalization call to base flow**

Insert `self.configuration.normalize_async(messages=conversation)` into each target's `send_prompt_async`, between the memory‑fetch+append step and the wire‑formatting step. The normalized conversation is used **only** for building the API request body — it is never written back to memory.

```python
# BEFORE (current pattern in every chat target)
conversation = self._memory.get_conversation(conversation_id=message_piece.conversation_id)
conversation.append(message)
body = await self._construct_request_body(conversation=conversation, ...)

# AFTER
conversation = self._memory.get_conversation(conversation_id=message_piece.conversation_id)
conversation.append(message)
normalized = await self.configuration.normalize_async(messages=list(conversation))
body = await self._construct_request_body(conversation=normalized, ...)
```

Key detail: `list(conversation)` creates a shallow copy so the original `conversation` (backed by memory) is never mutated.

**Phase B — Remove per‑target normalization code**

| Target | Code to remove / change |
|--------|------------------------|
| `AzureMLChatTarget` | Remove `message_normalizer` constructor parameter and `self.message_normalizer` field. System‑squash behavior moves to `TargetConfiguration(policy=CapabilityHandlingPolicy(behaviors={CapabilityName.SYSTEM_PROMPT: ADAPT}))`. Wire‑format conversion (`ChatMessageNormalizer.normalize_to_dicts_async`) remains as it is target‑specific. |
| `OpenAIChatTarget` | No normalization code to remove — currently passes messages through. Now gains system‑squash and history‑squash for free when configured. |
| `OpenAIResponseTarget` | Same as `OpenAIChatTarget`. |
| `OpenAIRealtimeTarget` | `_get_system_prompt_from_conversation` currently manually extracts the system message. After normalization, this method can trust the pipeline has handled squashing and only needs the first user message. |

**Phase C — Deprecate / remove `AzureMLChatTarget.message_normalizer`**

The `message_normalizer` parameter was the only per‑target normalization hook. With `TargetConfiguration` handling all adaptations, this field is redundant:

1. Emit `DeprecationWarning` when `message_normalizer` is passed
2. Map legacy `GenericSystemSquashNormalizer` usage to `CapabilityHandlingPolicy(behaviors={CapabilityName.SYSTEM_PROMPT: ADAPT})`
3. Remove in a future release

#### Affected Targets

| Target | Changes |
|--------|---------|
| `OpenAIChatTarget` | Add `normalize_async` call before `_construct_request_body` |
| `OpenAIResponseTarget` | Add `normalize_async` call before `_construct_request_body` |
| `OpenAIRealtimeTarget` | Add `normalize_async` call; simplify `_get_system_prompt_from_conversation` |
| `AzureMLChatTarget` | Add `normalize_async` call; deprecate `message_normalizer` field |
| `WebSocketTarget` | Add `normalize_async` call if applicable |

#### Memory Semantics (Critical Invariant)

Normalization is **ephemeral** — it MUST NOT alter what is stored in memory.

```
Memory stores:     [system, user₁, assistant₁, user₂]     ← canonical history
Pipeline returns:  [user₂ (with system + history squashed)] ← sent to API only
```

- `get_conversation()` always returns the **original** message list from memory
- `normalize_async()` produces a **copy** for the API request
- Response messages are stored with the original `conversation_id`, maintaining a clean conversation thread
- Re‑fetching the conversation for a subsequent turn returns the full, un‑normalized history

#### Test Plan

- **Unit tests**: For each target, verify that `configuration.normalize_async` is called with the full conversation, and that the return value (not the original) is passed to the wire‑formatting method.
- **Unit tests**: Verify memory is not mutated — assert that `get_conversation()` returns the same messages after `send_prompt_async`.
- **Unit tests**: Verify `AzureMLChatTarget` backward compat — passing `message_normalizer` emits `DeprecationWarning` and still works.
- **Integration tests**: End‑to‑end with a target that lacks system prompt support — verify the pipeline squashes correctly and the API call succeeds.
- **Integration tests**: Multi‑turn with history squash — verify the full history is in memory but only the squashed version reaches the endpoint.

#### Migration Guide

**For target authors**: Remove any inline normalization logic from `send_prompt_async`. Declare capabilities accurately in your `TargetConfiguration`. The pipeline handles adaptation automatically.

**For users constructing targets**: Instead of:

```python
target = AzureMLChatTarget(
    ...,
    message_normalizer=GenericSystemSquashNormalizer(),
)
```

Use:

```python
target = AzureMLChatTarget(
    ...,
    target_configuration=TargetConfiguration(
        capabilities=TargetCapabilities(supports_system_prompt=False),
        policy=CapabilityHandlingPolicy(
            behaviors={CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT}
        ),
    ),
)
```