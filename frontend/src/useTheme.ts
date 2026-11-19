import { useCallback, useEffect, useState } from 'react'
import { loadStored, saveStored } from './utils'

export type Theme = 'light' | 'spotify'

const STORAGE_KEY = 'sain.theme'

/** Тема оформления: светлая (по умолчанию) или Spotify. Живёт в localStorage. */
export function useTheme(): { theme: Theme; toggle: () => void; set: (t: Theme) => void } {
  const [theme, setTheme] = useState<Theme>(() => loadStored<Theme>(STORAGE_KEY, 'light'))

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('theme-spotify', theme === 'spotify')
    saveStored(STORAGE_KEY, theme)
  }, [theme])

  const toggle = useCallback(() => setTheme((t) => (t === 'spotify' ? 'light' : 'spotify')), [])

  return { theme, toggle, set: setTheme }
}
