import type { Theme } from '../useTheme'

interface Props {
  theme: Theme
  onToggle: () => void
}

/** Одна кнопка: переключает светлую тему и стиль Spotify. */
export default function ThemeToggle({ theme, onToggle }: Props) {
  const spotify = theme === 'spotify'
  return (
    <button
      onClick={onToggle}
      aria-pressed={spotify}
      title={spotify ? 'Вернуть светлую тему' : 'Включить стиль Spotify'}
      className="theme-toggle shrink-0 border-slate-200 bg-white text-slate-600 hover:border-brand-300"
    >
      <span aria-hidden>{spotify ? '🎧' : '🎨'}</span>
      <span className="hidden sm:inline">{spotify ? 'Spotify' : 'Стиль'}</span>
    </button>
  )
}
