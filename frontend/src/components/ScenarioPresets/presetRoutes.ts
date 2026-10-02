/**
 * Preset routes live under the scanner so presets stay a scanner concept
 * rather than a separate top-level destination.
 *
 * Editing uses a trailing `/edit` segment rather than bare
 * `/scanner/presets/:presetName` so the static create route cannot shadow a
 * preset whose name happens to be `new` — preset names are user-authored and
 * `new` matches the server's name pattern.
 */

export const PRESETS_ROUTE = '/scanner/presets'
export const NEW_PRESET_ROUTE = `${PRESETS_ROUTE}/new`

export function presetEditorRoutePath(name: string): string {
  return `${PRESETS_ROUTE}/${encodeURIComponent(name)}/edit`
}
