import { getInitialFormValues } from '@/components/Parameters/parameterForm'
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
  const available = new Set(
    uniqueTechniqueOptions(scenario).techniques.map((technique) => technique.name),
  )
  return preset.techniques.filter((name) => !available.has(name))
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
  const defaults = initialScenarioConfigState(scenario)
  const filters = preset.dataset_filters ?? {}
  const available = new Set(
    uniqueTechniqueOptions(scenario).techniques.map((technique) => technique.name),
  )
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

/**
 * Combines the preset's identity with the resolved scenario config.
 * `ScenarioPreset` forbids unknown fields, so only the keys
 * `buildScenarioConfig` produces may be spread in here.
 */
export function configToPreset(
  { name, scenarioName, description }: PresetIdentity,
  config: ScenarioConfigFields,
): ScenarioPreset {
  const trimmedDescription = description.trim()
  return {
    name,
    scenario_name: scenarioName,
    ...(trimmedDescription.length > 0 ? { description: trimmedDescription } : {}),
    ...config,
  }
}
