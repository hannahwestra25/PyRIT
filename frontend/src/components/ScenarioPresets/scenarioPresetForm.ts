import { buildParametersFromForm, getInitialFormValues } from '@/components/Parameters/parameterForm'
import {
  dynamicScenarioParameters,
  initialScenarioConfigState,
  uniqueTechniqueOptions,
  type ScenarioConfigFields,
  type ScenarioConfigFormState,
} from '@/components/Scenarios/scenarioConfigForm'
import type { RegisteredScenario, ScenarioPreset } from '@/types'

/**
 * Preset-specific form logic: the translation between a stored `ScenarioPreset`
 * and the scenario-config form state shared with the launch form. The launch
 * form never sees a preset, so none of this belongs in `scenarioConfigForm`.
 *
 * Every scenario-owned field on a preset is tri-state: an explicit value pins it,
 * and an omitted field means "whatever the scenario defaults to at run time". The
 * editor has to prefill omitted fields to render a control for them, so it writes a
 * field back only when the operator moved it off that default or the document already
 * pinned it. Round-tripping a preset through the editor otherwise freezes this
 * deployment's current defaults into the document on the first unrelated edit.
 */

/** Mirrors the server-side `ScenarioPreset.name` pattern so the editor fails fast instead of on a 422. */
const PRESET_NAME_PATTERN = /^[a-z][a-z0-9_]{0,63}$/

export function validatePresetName(name: string): string | null {
  if (name.length === 0) {
    return 'Name is required.'
  }
  if (!PRESET_NAME_PATTERN.test(name)) {
    return 'Use lowercase letters, digits and underscores, starting with a letter (max 64 characters).'
  }
  return null
}

/**
 * Technique names a preset may legitimately pin. Mirrors the server's allow-list:
 * the concrete techniques plus the scenario's aggregates (`all`, `easy`, ...), which
 * the selector has no checkbox for but which remain valid stored values.
 */
function presetSelectableTechniques(scenario: RegisteredScenario): Set<string> {
  const concrete = uniqueTechniqueOptions(scenario).techniques.map((technique) => technique.name)
  return new Set([...concrete, ...scenario.aggregate_techniques])
}

/**
 * Techniques the preset pins that this deployment's scenario no longer offers.
 * The editor cannot render a checkbox for them, so saving would silently drop
 * them — surfacing the names lets the operator decide instead.
 */
export function unknownPresetTechniques(
  scenario: RegisteredScenario,
  preset: ScenarioPreset,
): string[] {
  if (!preset.techniques) {
    return []
  }
  const available = presetSelectableTechniques(scenario)
  return preset.techniques.filter((name) => !available.has(name))
}

/**
 * Starting state for a preset that pins nothing yet. Identical to the launch form's
 * except the dataset cap stays blank. Unlike the technique and baseline controls, a
 * number input can represent "unset" directly, so the editor uses blank to mean
 * "track the scenario" rather than prefilling the default and comparing it back out.
 * The scenario's own cap is still shown as hint text beneath the field.
 */
export function initialPresetConfigState(scenario: RegisteredScenario): ScenarioConfigFormState {
  return { ...initialScenarioConfigState(scenario), maxDatasetSize: '' }
}

/**
 * Parameters the preset stores that this editor renders no control for — a key the
 * scenario no longer declares, or one of the common parameters the launch form owns.
 * They are carried through untouched rather than dropped on an unrelated edit.
 */
export function uneditableScenarioParams(
  scenario: RegisteredScenario,
  preset: ScenarioPreset | null,
): Record<string, unknown> {
  const stored = preset?.scenario_params
  if (!stored) {
    return {}
  }
  const editable = new Set(dynamicScenarioParameters(scenario).map((parameter) => parameter.name))
  return Object.fromEntries(Object.entries(stored).filter(([name]) => !editable.has(name)))
}

/**
 * Expands a stored preset into editable form state. Fields the preset omits
 * mean "use the scenario default", so they fall back to the same initial state
 * the launch form starts from.
 */
export function presetToConfigState(
  scenario: RegisteredScenario,
  preset: ScenarioPreset,
): ScenarioConfigFormState {
  const defaults = initialPresetConfigState(scenario)
  const filters = preset.dataset_filters ?? {}
  const available = presetSelectableTechniques(scenario)
  const pinned = preset.techniques?.filter((name) => available.has(name))
  return {
    techniques: pinned && pinned.length > 0 ? pinned : defaults.techniques,
    includeBaseline: scenario.baseline_policy === 'forbidden'
      ? false
      : preset.include_baseline ?? defaults.includeBaseline,
    datasetOverride: (preset.dataset_names ?? []).join(', '),
    maxDatasetSize: preset.max_dataset_size == null
      ? defaults.maxDatasetSize
      : String(preset.max_dataset_size),
    harmCategoriesFilter: (filters.harm_categories ?? []).join(', '),
    dataTypesFilter: (filters.data_types ?? []).join(', '),
    scenarioParamValues: getInitialFormValues(
      dynamicScenarioParameters(scenario),
      preset.scenario_params ?? null,
    ),
  }
}

interface PresetIdentity {
  name: string
  scenarioName: string
  description: string
}

interface PresetWriteContext {
  scenario: RegisteredScenario
  /** The document being edited, or `null` when creating. A field it already pins stays pinned. */
  previous: ScenarioPreset | null
}

interface ScenarioOwnedDefaults {
  techniques: string[]
  includeBaseline: boolean
  scenarioParams: Record<string, unknown>
}

/** What the scenario would resolve each field to on its own, i.e. what pinning nothing means. */
function scenarioOwnedDefaults(scenario: RegisteredScenario): ScenarioOwnedDefaults {
  const parameters = dynamicScenarioParameters(scenario)
  const built = parameters.length > 0
    ? buildParametersFromForm(parameters, getInitialFormValues(parameters))
    : null
  return {
    techniques: uniqueTechniqueOptions(scenario).defaultTechniques,
    includeBaseline: scenario.baseline_policy !== 'forbidden' && scenario.include_baseline_by_default,
    // A scenario whose untouched form cannot resolve (a required parameter with no
    // default) has no implicit value to compare against, so every key counts as edited.
    scenarioParams: built?.ok ? built.parameters ?? {} : {},
  }
}

/** Both sides originate from `buildParametersFromForm`, so a structural compare is sound. */
function matchesDefault(value: unknown, fallback: unknown): boolean {
  return JSON.stringify(value) === JSON.stringify(fallback)
}

function presetScenarioParams(
  config: ScenarioConfigFields,
  { scenario, previous }: PresetWriteContext,
  defaults: Record<string, unknown>,
): Record<string, unknown> | null {
  const stored = previous?.scenario_params ?? {}
  const edited = Object.entries(config.scenario_params ?? {}).filter(
    ([name, value]) => (
      Object.prototype.hasOwnProperty.call(stored, name) || !matchesDefault(value, defaults[name])
    ),
  )
  const merged = {
    ...Object.fromEntries(edited),
    ...uneditableScenarioParams(scenario, previous),
  }
  return Object.keys(merged).length > 0 ? merged : null
}

/**
 * Combines the preset's identity with the resolved scenario config, keeping fields the
 * operator left at the scenario's default unset so the preset keeps tracking them.
 */
export function configToPreset(
  { name, scenarioName, description }: PresetIdentity,
  config: ScenarioConfigFields,
  context: PresetWriteContext,
): ScenarioPreset {
  const { scenario, previous } = context
  const defaults = scenarioOwnedDefaults(scenario)
  const preset: ScenarioPreset = { name, scenario_name: scenarioName }

  const trimmedDescription = description.trim()
  if (trimmedDescription.length > 0) {
    preset.description = trimmedDescription
  }
  if (previous?.techniques != null || !matchesDefault(config.techniques, defaults.techniques)) {
    preset.techniques = config.techniques
  }
  // A forbidden baseline renders the checkbox disabled and forced off, so the form value
  // carries no operator intent — only whatever the document already said.
  const includeBaseline = scenario.baseline_policy === 'forbidden'
    ? previous?.include_baseline
    : config.include_baseline
  if (
    includeBaseline != null
    && (previous?.include_baseline != null || includeBaseline !== defaults.includeBaseline)
  ) {
    preset.include_baseline = includeBaseline
  }
  if (config.dataset_names) {
    preset.dataset_names = config.dataset_names
  }
  if (config.max_dataset_size !== undefined) {
    preset.max_dataset_size = config.max_dataset_size
  }
  if (config.dataset_filters) {
    preset.dataset_filters = config.dataset_filters
  }
  const scenarioParams = presetScenarioParams(config, context, defaults.scenarioParams)
  if (scenarioParams) {
    preset.scenario_params = scenarioParams
  }
  return preset
}
