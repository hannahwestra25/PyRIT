import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import { MemoryRouter } from 'react-router'

import { scenarioPresetsApi } from '@/services/api'
import type { PresetIssue, ScenarioPreset, ScenarioPresetResponse } from '@/types'

import ScenarioPresetLibrary from './ScenarioPresetLibrary'

jest.mock('@/services/api', () => ({
  scenarioPresetsApi: {
    list: jest.fn(),
    remove: jest.fn(),
    resolve: jest.fn(),
  },
  scenariosApi: {
    startRun: jest.fn(),
  },
}))

const mockList = scenarioPresetsApi.list as jest.Mock
const mockRemove = scenarioPresetsApi.remove as jest.Mock

const mockNavigate = jest.fn()

jest.mock('react-router', () => ({
  ...jest.requireActual('react-router'),
  useNavigate: () => mockNavigate,
}))

const NIGHTLY_PRESET: ScenarioPreset = {
  name: 'nightly_probe',
  scenario_name: 'foundry.red_team_agent',
  techniques: ['crescendo'],
}

function makeItem(
  preset: ScenarioPreset = NIGHTLY_PRESET,
  issues: PresetIssue[] = [],
): ScenarioPresetResponse {
  return { preset, version: 'v1', issues }
}

function renderLibrary() {
  return render(
    <FluentProvider theme={webLightTheme}>
      <MemoryRouter>
        <ScenarioPresetLibrary
          targets={[]}
          defaultObjectiveTarget={null}
          defaultAdversarialTarget={null}
          labels={{}}
        />
      </MemoryRouter>
    </FluentProvider>,
  )
}

beforeEach(() => {
  jest.clearAllMocks()
  mockList.mockResolvedValue({ source: 'presets.yaml', items: [] })
})

describe('ScenarioPresetLibrary', () => {
  it('shows an empty state and where presets are stored', async () => {
    renderLibrary()

    expect(await screen.findByTestId('empty-state')).toBeInTheDocument()
    expect(screen.getByText('Stored in presets.yaml')).toBeInTheDocument()
  })

  it('lists each saved preset with its scenario and technique count', async () => {
    mockList.mockResolvedValue({
      source: 'presets.yaml',
      items: [
        makeItem(),
        makeItem({ name: 'weekly_probe', scenario_name: 'foundry.encoding', techniques: ['base64'] }),
      ],
    })

    renderLibrary()

    expect(await screen.findByTestId('preset-card-nightly_probe')).toBeInTheDocument()
    expect(screen.getByText('foundry.red_team_agent · 1 technique')).toBeInTheDocument()
    expect(screen.getByTestId('preset-card-weekly_probe')).toBeInTheDocument()
  })

  it('reports scenario default techniques when the preset pins none', async () => {
    mockList.mockResolvedValue({
      source: 'presets.yaml',
      items: [makeItem({ name: 'nightly_probe', scenario_name: 'foundry.red_team_agent' })],
    })

    renderLibrary()

    expect(
      await screen.findByText('foundry.red_team_agent · Scenario default techniques'),
    ).toBeInTheDocument()
  })

  it('disables launch for an unresolvable preset and names the missing reference', async () => {
    mockList.mockResolvedValue({
      source: 'presets.yaml',
      items: [makeItem(NIGHTLY_PRESET, [
        { field: 'scenario_name', message: 'Unknown scenario "foundry.retired".' },
      ])],
    })

    renderLibrary()

    expect(await screen.findByTestId('launch-preset-nightly_probe')).toBeDisabled()
    expect(screen.getByText('Not runnable here')).toBeInTheDocument()
    expect(screen.getByText('Unknown scenario "foundry.retired".')).toBeInTheDocument()
  })

  it('keeps launch enabled when the server reports no issues', async () => {
    mockList.mockResolvedValue({ source: 'presets.yaml', items: [makeItem()] })

    renderLibrary()

    expect(await screen.findByTestId('launch-preset-nightly_probe')).toBeEnabled()
    expect(screen.queryByTestId('preset-issues-nightly_probe')).not.toBeInTheDocument()
  })

  it('surfaces a load failure with a retry that refetches', async () => {
    const user = userEvent.setup()
    mockList.mockRejectedValueOnce(new Error('storage unavailable'))

    renderLibrary()

    expect(await screen.findByTestId('error-state')).toBeInTheDocument()
    expect(screen.getByText('storage unavailable')).toBeInTheDocument()

    mockList.mockResolvedValue({ source: 'presets.yaml', items: [makeItem()] })
    await user.click(screen.getByTestId('retry-btn'))

    expect(await screen.findByTestId('preset-card-nightly_probe')).toBeInTheDocument()
  })

  it('requires confirmation before deleting and refreshes afterwards', async () => {
    const user = userEvent.setup()
    mockList.mockResolvedValue({ source: 'presets.yaml', items: [makeItem()] })
    mockRemove.mockResolvedValue(undefined)

    renderLibrary()

    await user.click(await screen.findByTestId('delete-preset-nightly_probe'))
    expect(mockRemove).not.toHaveBeenCalled()

    mockList.mockResolvedValue({ source: 'presets.yaml', items: [] })
    const dialog = await screen.findByRole('dialog')
    await user.click(within(dialog).getByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(mockRemove).toHaveBeenCalledWith('nightly_probe'))
    expect(await screen.findByTestId('empty-state')).toBeInTheDocument()
  })

  it('cancels a delete without calling the server', async () => {
    const user = userEvent.setup()
    mockList.mockResolvedValue({ source: 'presets.yaml', items: [makeItem()] })

    renderLibrary()

    await user.click(await screen.findByTestId('delete-preset-nightly_probe'))
    const dialog = await screen.findByRole('dialog')
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))

    expect(mockRemove).not.toHaveBeenCalled()
    expect(screen.getByTestId('preset-card-nightly_probe')).toBeInTheDocument()
  })

  it('reports a delete failure without removing the card', async () => {
    const user = userEvent.setup()
    mockList.mockResolvedValue({ source: 'presets.yaml', items: [makeItem()] })
    mockRemove.mockRejectedValue(new Error('Admin access required.'))

    renderLibrary()

    await user.click(await screen.findByTestId('delete-preset-nightly_probe'))
    const dialog = await screen.findByRole('dialog')
    await user.click(within(dialog).getByRole('button', { name: 'Delete' }))

    expect(await screen.findByText('Admin access required.')).toBeInTheDocument()
    expect(screen.getByTestId('preset-card-nightly_probe')).toBeInTheDocument()
  })

  it('navigates to the editor for create and edit', async () => {
    const user = userEvent.setup()
    mockList.mockResolvedValue({ source: 'presets.yaml', items: [makeItem()] })

    renderLibrary()

    await user.click(await screen.findByTestId('edit-preset-nightly_probe'))
    expect(mockNavigate).toHaveBeenCalledWith('/scanner/presets/nightly_probe/edit')

    await user.click(screen.getByTestId('new-preset-btn'))
    expect(mockNavigate).toHaveBeenCalledWith('/scanner/presets/new')
  })

  it('opens the launch dialog only for a runnable preset', async () => {
    const user = userEvent.setup()
    mockList.mockResolvedValue({ source: 'presets.yaml', items: [makeItem()] })

    renderLibrary()

    await user.click(await screen.findByTestId('launch-preset-nightly_probe'))

    expect(await screen.findByTestId('confirm-launch-preset')).toBeInTheDocument()
  })
})
