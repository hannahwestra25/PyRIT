import { buildScenarioConfig } from '@/components/Scenarios/scenarioConfigForm'
import { makeScenario } from '@/test-utils/scenarioFixtures'
import type { Parameter, ScenarioPreset } from '@/types'

import {
  configToPreset,
  presetToConfigState,
  unknownPresetTechniques,
  validatePresetName,
} from './scenarioPresetForm'

function makePreset(overrides: Partial<ScenarioPreset> = {}): ScenarioPreset {
  return {
    name: 'nightly_probe',
    scenario_name: 'foundry.red_team_agent',
    ...overrides,
  }
}

const ITERATION_PARAMETER: Parameter = {
  name: 'max_turns',
  type: 'int',
  required: false,
  default: 5,
  description: 'Turn budget.',
}

describe('validatePresetName', () => {
  it('rejects an empty name', () => {
    expect(validatePresetName('')).toBe('Name is required.')
  })

  it.each([
    ['nightly_probe'],
    ['a'],
    ['a1_b2'],
    [`a${'b'.repeat(63)}`],
  ])('accepts %s', (name) => {
    expect(validatePresetName(name)).toBeNull()
  })

  it.each([
    ['Nightly'],
    ['1nightly'],
    ['nightly-probe'],
    ['nightly probe'],
    ['nightly.probe'],
    [`a${'b'.repeat(64)}`],
  ])('rejects %s', (name) => {
    expect(validatePresetName(name)).not.toBeNull()
  })
})

describe('unknownPresetTechniques', () => {
  it('returns nothing when the preset pins no techniques', () => {
    expect(unknownPresetTechniques(makeScenario(), makePreset())).toEqual([])
  })

  it('names only the techniques the scenario no longer offers', () => {
    const scenario = makeScenario({ all_techniques: ['crescendo', 'default_technique'] })
    const preset = makePreset({ techniques: ['crescendo', 'retired_attack'] })

    expect(unknownPresetTechniques(scenario, preset)).toEqual(['retired_attack'])
  })

  it('treats an aggregate technique as unavailable because it is not selectable', () => {
    const scenario = makeScenario({ aggregate_techniques: ['all', 'default'] })
    const preset = makePreset({ techniques: ['all'] })

    expect(unknownPresetTechniques(scenario, preset)).toEqual(['all'])
  })
})

describe('presetToConfigState', () => {
  it('falls back to the scenario defaults for every omitted field', () => {
    const scenario = makeScenario()
    const state = presetToConfigState(scenario, makePreset())

    expect(state.techniques).toEqual(['default_technique'])
    expect(state.includeBaseline).toBe(true)
    expect(state.datasetOverride).toBe('')
    expect(state.maxDatasetSize).toBe('')
    expect(state.harmCategoriesFilter).toBe('')
    expect(state.dataTypesFilter).toBe('')
  })

  it('expands the stored fields a preset does carry', () => {
    const scenario = makeScenario()
    const state = presetToConfigState(scenario, makePreset({
      techniques: ['crescendo'],
      include_baseline: false,
      dataset_names: ['harmbench', 'xstest'],
      max_dataset_size: 25,
      dataset_filters: { harm_categories: ['violence'], data_types: ['text'] },
    }))

    expect(state.techniques).toEqual(['crescendo'])
    expect(state.includeBaseline).toBe(false)
    expect(state.datasetOverride).toBe('harmbench, xstest')
    expect(state.maxDatasetSize).toBe('25')
    expect(state.harmCategoriesFilter).toBe('violence')
    expect(state.dataTypesFilter).toBe('text')
  })

  it('drops pinned techniques the scenario no longer offers', () => {
    const scenario = makeScenario({ all_techniques: ['crescendo', 'default_technique'] })
    const state = presetToConfigState(
      scenario,
      makePreset({ techniques: ['crescendo', 'retired_attack'] }),
    )

    expect(state.techniques).toEqual(['crescendo'])
  })

  it('falls back to the scenario defaults when every pinned technique is gone', () => {
    const scenario = makeScenario({ all_techniques: ['crescendo', 'default_technique'] })
    const state = presetToConfigState(scenario, makePreset({ techniques: ['retired_attack'] }))

    expect(state.techniques).toEqual(['default_technique'])
  })

  it('keeps baseline off when the scenario forbids it, whatever the preset stored', () => {
    const scenario = makeScenario({ baseline_policy: 'forbidden' })
    const state = presetToConfigState(scenario, makePreset({ include_baseline: true }))

    expect(state.includeBaseline).toBe(false)
  })

  it('seeds dynamic parameter values from the stored scenario params', () => {
    const scenario = makeScenario({ supported_parameters: [ITERATION_PARAMETER] })
    const state = presetToConfigState(
      scenario,
      makePreset({ scenario_params: { max_turns: 9 } }),
    )

    expect(state.scenarioParamValues.max_turns).toBe('9')
  })
})

describe('configToPreset', () => {
  function buildConfig(overrides: Partial<Parameters<typeof buildScenarioConfig>[0]> = {}) {
    const result = buildScenarioConfig({
      techniques: ['crescendo'],
      dynamicParameters: [],
      scenarioParamValues: {},
      datasetOverride: '',
      maxDatasetSize: '',
      harmCategoriesFilter: '',
      dataTypesFilter: '',
      includeBaseline: false,
      ...overrides,
    })
    if (!result.ok) {
      throw new Error(result.error)
    }
    return result.config
  }

  it('omits a blank description rather than storing an empty string', () => {
    const preset = configToPreset(
      { name: 'nightly_probe', scenarioName: 'foundry.red_team_agent', description: '   ' },
      buildConfig(),
    )

    expect(preset).not.toHaveProperty('description')
  })

  it('trims a description it does keep', () => {
    const preset = configToPreset(
      { name: 'nightly_probe', scenarioName: 'foundry.red_team_agent', description: '  nightly  ' },
      buildConfig(),
    )

    expect(preset.description).toBe('nightly')
  })

  it('carries only the scenario-owned keys the config produced', () => {
    const preset = configToPreset(
      { name: 'nightly_probe', scenarioName: 'foundry.red_team_agent', description: '' },
      buildConfig(),
    )

    expect(Object.keys(preset).sort()).toEqual(
      ['include_baseline', 'name', 'scenario_name', 'techniques'],
    )
  })

  it('round-trips a fully populated preset back through the form state', () => {
    const scenario = makeScenario()
    const original = makePreset({
      techniques: ['crescendo'],
      include_baseline: false,
      dataset_names: ['harmbench'],
      max_dataset_size: 25,
      dataset_filters: { harm_categories: ['violence'] },
    })
    const state = presetToConfigState(scenario, original)

    const roundTripped = configToPreset(
      { name: original.name, scenarioName: original.scenario_name, description: '' },
      buildConfig({
        techniques: state.techniques,
        includeBaseline: state.includeBaseline,
        datasetOverride: state.datasetOverride,
        maxDatasetSize: state.maxDatasetSize,
        harmCategoriesFilter: state.harmCategoriesFilter,
        dataTypesFilter: state.dataTypesFilter,
      }),
    )

    expect(roundTripped).toEqual(original)
  })
})
