import type { LlmInfo } from '../types'

interface Props {
  info: LlmInfo | null
  onChange: (provider: string) => void
  disabled?: boolean
  busy?: boolean
}

/**
 * Индикатор LLM: прокси нейросети хакатона (gpt-6-luna).
 * Показывает текущего провайдера и состояние доступности ключа.
 */
export default function LlmSwitch({ info, onChange, disabled, busy }: Props) {
  const options = info?.options ?? []
  const current = info?.provider ?? 'openai'
  const currentOption = options.find((o) => o.id === current)
  const other = options.find((o) => o.id !== current)
  const ok = info?.configured ?? false

  const next = () => {
    if (!other) return
    onChange(other.id)
  }

  return (
    <button
      onClick={next}
      disabled={disabled || busy || !other}
      title={
        other
          ? `Переключить на ${other.label} (${other.hint})`
          : 'Нет альтернативного провайдера'
      }
      className="flex shrink-0 items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] text-slate-600 transition hover:border-brand-300 disabled:cursor-not-allowed disabled:opacity-60"
    >
      <span className={`h-2 w-2 rounded-full ${ok ? 'bg-emerald-500' : 'bg-slate-300'}`} />
      <span className="font-medium">{busy ? '…' : (currentOption?.label ?? current)}</span>
      <span aria-hidden className="text-slate-400">
        ⇄
      </span>
    </button>
  )
}
