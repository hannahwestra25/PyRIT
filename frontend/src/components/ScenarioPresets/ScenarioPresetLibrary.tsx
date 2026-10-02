import { useCallback, useEffect, useState } from 'react'

import {
  Badge,
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  MessageBar,
  MessageBarBody,
  Spinner,
  Text,
} from '@fluentui/react-components'
import {
  AddRegular,
  ArrowSyncRegular,
  DeleteRegular,
  EditRegular,
  PlayRegular,
} from '@fluentui/react-icons'
import { Link, useNavigate } from 'react-router'

import { scenarioPresetsApi } from '@/services/api'
import { toApiError } from '@/services/errors'
import type { ScenarioPresetResponse, TargetInstance } from '@/types'

import LaunchPresetDialog from './LaunchPresetDialog'
import { useScenarioPresetLibraryStyles } from './ScenarioPresetLibrary.styles'
import { NEW_PRESET_ROUTE, presetEditorRoutePath } from './presetRoutes'

interface ScenarioPresetLibraryProps {
  targets: TargetInstance[]
  defaultObjectiveTarget: TargetInstance | null
  defaultAdversarialTarget: TargetInstance | null
  labels: Record<string, string>
}

function techniqueSummary(item: ScenarioPresetResponse): string {
  const { techniques } = item.preset
  if (!techniques || techniques.length === 0) {
    return 'Scenario default techniques'
  }
  return `${techniques.length} technique${techniques.length === 1 ? '' : 's'}`
}

/**
 * Lists the saved scenario presets. Presets that reference something this
 * deployment does not have are still listed — the server reports the problems
 * as advisory issues — but cannot be launched here.
 */
export default function ScenarioPresetLibrary({
  targets,
  defaultObjectiveTarget,
  defaultAdversarialTarget,
  labels,
}: ScenarioPresetLibraryProps) {
  const styles = useScenarioPresetLibraryStyles()
  const navigate = useNavigate()
  const [items, setItems] = useState<ScenarioPresetResponse[]>([])
  const [source, setSource] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [launching, setLaunching] = useState<ScenarioPresetResponse | null>(null)
  const [pendingDelete, setPendingDelete] = useState<ScenarioPresetResponse | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [generation, setGeneration] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false

    const loadPresets = async (): Promise<void> => {
      try {
        const response = await scenarioPresetsApi.list(controller.signal)
        if (cancelled) return
        setItems(response.items)
        setSource(response.source)
        setError(null)
      } catch (err: unknown) {
        if (cancelled || controller.signal.aborted) return
        setError(toApiError(err).detail)
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadPresets()

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [generation])

  const refresh = useCallback(() => {
    setLoading(true)
    setError(null)
    setGeneration((current) => current + 1)
  }, [])

  const handleDelete = async (): Promise<void> => {
    if (!pendingDelete) {
      return
    }
    setDeleting(true)
    setActionError(null)
    try {
      await scenarioPresetsApi.remove(pendingDelete.preset.name)
      setPendingDelete(null)
      refresh()
    } catch (err) {
      setActionError(toApiError(err).detail)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <section
      className={styles.root}
      data-testid="scenario-preset-library"
      aria-labelledby="scenario-preset-library-title"
    >
      <div className={styles.header}>
        <div className={styles.headerText}>
          <Text id="scenario-preset-library-title" as="h1" size={600} weight="semibold">
            Scenario presets
          </Text>
          <Text size={300} className={styles.subtitle}>
            Reusable scenario configurations. A preset pins what to test; you choose the target at
            launch.
          </Text>
          <Text size={200} className={styles.subtitle}>
            <Link to="/scanner">Back to the scanner catalog</Link>
          </Text>
        </div>
        <div className={styles.headerActions}>
          <Button
            className={styles.touchTarget}
            appearance="primary"
            icon={<AddRegular />}
            onClick={() => navigate(NEW_PRESET_ROUTE)}
            data-testid="new-preset-btn"
          >
            New preset
          </Button>
          <Button
            className={styles.touchTarget}
            appearance="subtle"
            icon={<ArrowSyncRegular />}
            onClick={refresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </div>
      </div>

      {source !== '' && (
        <Text size={200} className={styles.sourceLine}>
          Stored in {source}
        </Text>
      )}

      {actionError && (
        <MessageBar intent="error">
          <MessageBarBody>{actionError}</MessageBarBody>
        </MessageBar>
      )}

      {loading ? (
        <div className={styles.centeredState}>
          <Spinner label="Loading presets..." />
        </div>
      ) : error ? (
        <div className={styles.centeredState} data-testid="error-state">
          <MessageBar intent="error">
            <MessageBarBody>{error}</MessageBarBody>
          </MessageBar>
          <Button
            className={styles.touchTarget}
            appearance="primary"
            icon={<ArrowSyncRegular />}
            onClick={refresh}
            data-testid="retry-btn"
          >
            Retry
          </Button>
        </div>
      ) : items.length === 0 ? (
        <div className={styles.centeredState} data-testid="empty-state">
          <Text size={400}>No presets saved yet</Text>
          <Text size={200}>
            Create one to reuse a scenario configuration across targets and runs.
          </Text>
        </div>
      ) : (
        <div className={styles.list}>
          {items.map((item) => {
            const runnable = item.issues.length === 0
            return (
              <div
                key={item.preset.name}
                className={styles.card}
                data-testid={`preset-card-${item.preset.name}`}
              >
                <div className={styles.cardText}>
                  <div className={styles.cardTitleRow}>
                    <Link to={presetEditorRoutePath(item.preset.name)}>
                      <Text weight="semibold">{item.preset.name}</Text>
                    </Link>
                    {!runnable && (
                      <Badge appearance="tint" color="warning">
                        Not runnable here
                      </Badge>
                    )}
                  </div>
                  <Text size={200} className={styles.cardMeta}>
                    {item.preset.scenario_name} · {techniqueSummary(item)}
                  </Text>
                  {item.preset.description && (
                    <Text size={200}>{item.preset.description}</Text>
                  )}
                  {!runnable && (
                    <ul className={styles.issueList} data-testid={`preset-issues-${item.preset.name}`}>
                      {item.issues.map((issue) => (
                        <li key={`${issue.field}:${issue.message}`}>
                          <Text size={200}>{issue.message}</Text>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div className={styles.cardActions}>
                  <Button
                    className={styles.touchTarget}
                    appearance="primary"
                    icon={<PlayRegular />}
                    disabled={!runnable}
                    onClick={() => setLaunching(item)}
                    data-testid={`launch-preset-${item.preset.name}`}
                  >
                    Launch
                  </Button>
                  <Button
                    className={styles.touchTarget}
                    appearance="subtle"
                    icon={<EditRegular />}
                    onClick={() => navigate(presetEditorRoutePath(item.preset.name))}
                    data-testid={`edit-preset-${item.preset.name}`}
                  >
                    Edit
                  </Button>
                  <Button
                    className={styles.touchTarget}
                    appearance="subtle"
                    icon={<DeleteRegular />}
                    onClick={() => {
                      setActionError(null)
                      setPendingDelete(item)
                    }}
                    data-testid={`delete-preset-${item.preset.name}`}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {launching && (
        <LaunchPresetDialog
          preset={launching.preset}
          targets={targets}
          defaultObjectiveTarget={defaultObjectiveTarget}
          defaultAdversarialTarget={defaultAdversarialTarget}
          labels={labels}
          onDismiss={() => setLaunching(null)}
        />
      )}

      {pendingDelete && (
        <Dialog
          open
          onOpenChange={(_, data) => { if (!data.open) setPendingDelete(null) }}
        >
          <DialogSurface>
            <DialogBody>
              <DialogTitle>Delete {pendingDelete.preset.name}?</DialogTitle>
              <DialogContent className={styles.dialogBody}>
                <Text>
                  This permanently removes the preset. Runs already started from it are unaffected.
                </Text>
              </DialogContent>
              <DialogActions>
                <Button
                  appearance="secondary"
                  onClick={() => setPendingDelete(null)}
                  disabled={deleting}
                >
                  Cancel
                </Button>
                <Button
                  appearance="primary"
                  onClick={handleDelete}
                  disabled={deleting}
                  data-testid="confirm-delete-preset"
                >
                  {deleting ? 'Deleting...' : 'Delete'}
                </Button>
              </DialogActions>
            </DialogBody>
          </DialogSurface>
        </Dialog>
      )}
    </section>
  )
}
