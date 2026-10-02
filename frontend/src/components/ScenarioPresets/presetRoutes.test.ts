import { NEW_PRESET_ROUTE, PRESETS_ROUTE, presetEditorRoutePath } from './presetRoutes'

describe('presetEditorRoutePath', () => {
  it('builds an edit route under the preset library', () => {
    expect(presetEditorRoutePath('nightly_probe')).toBe('/scanner/presets/nightly_probe/edit')
  })

  it('escapes a name so it stays inside one path segment', () => {
    expect(presetEditorRoutePath('a/b')).toBe('/scanner/presets/a%2Fb/edit')
  })

  it('does not collide with the create route for a preset named "new"', () => {
    expect(presetEditorRoutePath('new')).not.toBe(NEW_PRESET_ROUTE)
    expect(presetEditorRoutePath('new').startsWith(`${NEW_PRESET_ROUTE}/`)).toBe(true)
  })

  it('nests both routes under the scanner so the scanner nav item stays selected', () => {
    expect(PRESETS_ROUTE.startsWith('/scanner/')).toBe(true)
    expect(NEW_PRESET_ROUTE.startsWith('/scanner/')).toBe(true)
  })
})
