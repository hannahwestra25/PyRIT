import { useId, useMemo } from 'react'

import { Checkbox, Text, ToggleButton } from '@fluentui/react-components'

import type { ScenarioTechniqueSummary } from '@/types'

import { useScenarioTechniqueSelectorStyles } from './ScenarioTechniqueSelector.styles'
import { buildSelectableTechniques, type SelectableTechnique } from './scenarioConfigForm'
import { techniqueSetName } from './scenarioTechniqueSets'

interface ScenarioTechniqueSelectorProps {
  techniqueOptions: ScenarioTechniqueSummary[]
  selectedTechniques: string[]
  includeBaseline: boolean
  isBaselineForbidden: boolean
  disabled: boolean
  onTechniquesChange: (techniques: string[]) => void
  onIncludeBaselineChange: (includeBaseline: boolean) => void
}

/**
 * Technique picker shared by the scenario launch form and the preset editor.
 *
 * Baseline is rendered as a pseudo-technique so an operator sees one list, but it
 * travels as its own flag because the backend models it as `include_baseline`.
 */
export default function ScenarioTechniqueSelector({
  techniqueOptions,
  selectedTechniques,
  includeBaseline,
  isBaselineForbidden,
  disabled,
  onTechniquesChange,
  onIncludeBaselineChange,
}: ScenarioTechniqueSelectorProps) {
  const styles = useScenarioTechniqueSelectorStyles()
  const titleId = useId()

  const selectableTechniques = useMemo<SelectableTechnique[]>(
    () => buildSelectableTechniques(techniqueOptions, isBaselineForbidden),
    [isBaselineForbidden, techniqueOptions],
  )

  const isTechniqueSelected = (technique: SelectableTechnique): boolean => (
    technique.isBaseline ? includeBaseline : selectedTechniques.includes(technique.name)
  )

  const handleTechniqueChange = (technique: SelectableTechnique, checked: boolean): void => {
    if (technique.isBaseline) {
      onIncludeBaselineChange(checked)
      return
    }
    if (checked) {
      if (!selectedTechniques.includes(technique.name)) {
        onTechniquesChange([...selectedTechniques, technique.name])
      }
      return
    }
    onTechniquesChange(selectedTechniques.filter((name) => name !== technique.name))
  }

  const handleTagChange = (tag: string): void => {
    const members = selectableTechniques.filter(
      (technique) => !technique.disabled && technique.tags.includes(tag),
    )
    const shouldSelect = members.some((technique) => !isTechniqueSelected(technique))
    const memberNames = new Set(
      members.filter((technique) => !technique.isBaseline).map((technique) => technique.name),
    )
    const selected = new Set(selectedTechniques)
    for (const name of memberNames) {
      if (shouldSelect) selected.add(name)
      else selected.delete(name)
    }
    onTechniquesChange(
      techniqueOptions.map((technique) => technique.name).filter((name) => selected.has(name)),
    )
    if (members.some((technique) => technique.isBaseline)) {
      onIncludeBaselineChange(shouldSelect)
    }
  }

  return (
    <section className={styles.section} aria-labelledby={titleId}>
      <Text id={titleId} as="h2" size={400} weight="semibold">
        Techniques
      </Text>
      <Text size={200} className={styles.hint}>
        Select individual techniques, or use a tag to select or clear all techniques with that tag.
      </Text>
      {selectedTechniques.length === 0 && (
        <Text className={styles.errorText} role="alert">
          Select at least one attack technique.
        </Text>
      )}
      <div className={styles.techniqueList} role="group" aria-label="Techniques">
        {selectableTechniques.map((technique) => {
          const selected = isTechniqueSelected(technique)
          return (
            <div className={styles.techniqueOption} key={technique.name}>
              <Checkbox
                className={styles.selectionControl}
                label={technique.name}
                checked={selected}
                disabled={disabled || technique.disabled}
                onChange={(_, data) => handleTechniqueChange(technique, data.checked === true)}
                data-testid={technique.isBaseline ? 'baseline-checkbox' : `technique-${technique.name}`}
              />
              <div className={styles.techniqueDetails}>
                {technique.description && (
                  <Text size={200} className={styles.hint}>{technique.description}</Text>
                )}
                {technique.tags.length > 0 && (
                  <div className={styles.techniqueTags} aria-label={`${technique.name} tags`}>
                    {technique.tags.map((tag) => {
                      const tagMembers = selectableTechniques.filter(
                        (candidate) => !candidate.disabled && candidate.tags.includes(tag),
                      )
                      const tagSelected = tagMembers.length > 0 && tagMembers.every(isTechniqueSelected)
                      return (
                        <ToggleButton
                          className={styles.techniqueTag}
                          key={tag}
                          size="small"
                          appearance="outline"
                          checked={tagSelected}
                          disabled={disabled || tagMembers.length === 0}
                          onClick={() => handleTagChange(tag)}
                          aria-label={`${tagSelected ? 'Clear' : 'Select'} ${techniqueSetName(tag)} techniques`}
                        >
                          {techniqueSetName(tag)}
                        </ToggleButton>
                      )
                    })}
                  </div>
                )}
                {technique.disabled && (
                  <Text size={200} className={styles.hint}>
                    This scenario does not support a baseline comparison.
                  </Text>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
