import { makeStyles, tokens } from '@fluentui/react-components'

export const useLaunchPresetDialogStyles = makeStyles({
  body: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
  },
  scenarioLine: {
    color: tokens.colorNeutralForeground3,
  },
  numberInput: {
    maxWidth: '10rem',
  },
})
