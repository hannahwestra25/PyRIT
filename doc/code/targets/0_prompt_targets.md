# Prompt Targets

Prompt Targets are endpoints for where to send prompts. For example, a target could be a GPT-4 or Llama endpoint. Targets are typically used with other components like [attacks](../executor/attack/0_attack.md), [scorers](../scoring/0_scoring.md), and [converters](../converters/0_converters.ipynb).

- An attack's main job is to change prompts to a given format, apply any converters, and then send them off to prompt targets (sometimes using various strategies). Within an attack, prompt targets are (mostly) swappable, meaning you can use the same logic with different target endpoints.
- A scorer's main job is to score a prompt. Often, these use LLMs, in which case, a given scorer can often use different configured targets.
- A converter's job is to transform a prompt. Often, these use LLMs, in which case, a given converter can use different configured targets.

Prompt targets are found [here](https://github.com/Azure/PyRIT/tree/main/pyrit/prompt_target/) in code.


## Send_Prompt_Async

The main entry method follow the following signature:

```
async def send_prompt_async(self, *, message: Message) -> Message:
```

A `Message` object is a normalized object with all the information a target will need to send a prompt, including a way to get a history for that prompt (in the cases that also needs to be sent). This is discussed in more depth [here](../memory/3_memory_data_types.md).

## Target Capabilities

Every `PromptTarget` declares a `TargetCapabilities` object that describes what the target supports. Attacks, scorers, and converters use these flags to validate that a target is compatible before execution, raising a clear error at construction time rather than failing mid-run.

| Capability | Type | Description |
|---|---|---|
| `supports_multi_turn` | `bool` | Target accepts conversation history across multiple turns. Required by multi-turn attacks (e.g., PAIR, TAP, Crescendo). |
| `supports_editable_history` | `bool` | Target allows prepended conversation history to be injected into memory. Required by attacks that seed a conversation before starting (e.g., TAP, FlipAttack, ContextCompliance). |
| `supports_multi_message_pieces` | `bool` | Target accepts a single request with multiple pieces (e.g., text + image in one turn). |
| `supports_json_output` | `bool` | Target can be instructed to return valid JSON (e.g., via a `response_format` parameter). |
| `supports_json_schema` | `bool` | Target can constrain output to a specific JSON schema. |
| `input_modalities` | `frozenset` | The combinations of data types the target accepts as input (e.g., `{"text"}`, `{"text", "image_path"}`). |
| `output_modalities` | `frozenset` | The data types the target can produce as output (e.g., `{"text"}`, `{"audio_path"}`). |

Capabilities are defined at the class level via `_DEFAULT_CAPABILITIES` and can be overridden per instance using the `custom_capabilities` constructor parameter. This is useful for targets like `HTTPTarget` or `PlaywrightTarget` where capabilities depend on the specific deployment being wrapped.

Here are some examples:

| Example | `supports_multi_turn` | `supports_editable_history` | Notes |
|---|---|---|---|
| **OpenAIChatTarget** | Yes | Yes | Full chat target; supports multi-turn and injected history. |
| **OpenAIImageTarget** | No | No | Image generation; single-turn only. |
| **OpenAITTSTarget** | No | No | Text-to-speech; single-turn only. |
| **HTTPTarget** | No (default) | No (default) | Configurable via `custom_capabilities` if the wrapped app supports it. |
| **AzureBlobStorageTarget** | No | No | Storage target; not conversational. |

## Multi-Modal Targets

Like most of PyRIT, targets can be multi-modal.

- [OpenAI Chat Target](./1_openai_chat_target.ipynb) (*text + image --> text*)
- [OpenAI Image Target](./3_openai_image_target.ipynb) (*text --> image* or *text + image --> image*)
- [OpenAI Video Target](./4_openai_video_target.ipynb) (*text --> video*)
- [OpenAI TTS Target](./5_openai_tts_target.ipynb) (*text --> audio*)
