import { useEffect, useRef, useState } from 'react'

interface Props {
  onSend: (text: string) => void
  onStop: () => void
  streaming: boolean
  disabled?: boolean
  maxLength?: number
  /** Внешний текст (например, клик по подсказке на пустом экране). */
  draft?: string
  onDraftConsumed?: () => void
}

/**
 * Поле ввода + «Спросить» / «Стоп».
 * Подсказки живут в пустом экране (EmptyState), а «Начать чат заново» — в шапке.
 */
export default function Composer({
  onSend,
  onStop,
  streaming,
  disabled,
  maxLength = 4000,
  draft,
  onDraftConsumed,
}: Props) {
  const [text, setText] = useState('')
  const areaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = areaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [text])

  // Подсказка с пустого экрана: подставляем и ставим курсор в конец.
  useEffect(() => {
    if (!draft) return
    setText(draft)
    onDraftConsumed?.()
    requestAnimationFrame(() => {
      const el = areaRef.current
      if (el) {
        el.focus()
        el.setSelectionRange(draft.length, draft.length)
      }
    })
  }, [draft, onDraftConsumed])

  const canSend = text.trim().length > 0 && !streaming && !disabled

  const submit = () => {
    const value = text.trim()
    if (!value || streaming || disabled) return
    onSend(value)
    setText('')
  }

  return (
    <div className="border-t border-slate-200 bg-white/80 px-4 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur">
      <div className="flex items-end gap-2">
        <textarea
          ref={areaRef}
          value={text}
          onChange={(e) => setText(e.target.value.slice(0, maxLength))}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          rows={1}
          placeholder="Опишите ситуацию или задайте вопрос…"
          aria-label="Ваш вопрос"
          className="min-h-[44px] flex-1 resize-none rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-brand-500 focus:ring-2 focus:ring-brand-100 disabled:bg-slate-50"
          disabled={disabled}
        />

        {streaming ? (
          <button
            onClick={onStop}
            aria-label="Остановить генерацию"
            className="flex h-11 shrink-0 items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
          >
            <span className="h-2.5 w-2.5 rounded-sm bg-slate-700" aria-hidden />
            Стоп
          </button>
        ) : (
          <button
            onClick={submit}
            disabled={!canSend}
            aria-label="Спросить"
            className={`flex h-11 shrink-0 items-center gap-1.5 rounded-xl px-4 text-sm font-semibold transition
              ${canSend ? 'bg-brand-600 text-white shadow-sm hover:bg-brand-700' : 'cursor-not-allowed bg-slate-200 text-slate-400'}`}
          >
            <span className="hidden sm:inline">Спросить</span>
            <span aria-hidden>→</span>
          </button>
        )}
      </div>

      <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-400">
        <span>
          <kbd className="rounded border border-slate-200 bg-slate-50 px-1">Enter</kbd> — отправить,{' '}
          <kbd className="rounded border border-slate-200 bg-slate-50 px-1">Shift+Enter</kbd> — новая строка
        </span>
        <span className={text.length > maxLength * 0.9 ? 'text-amber-600' : undefined}>
          {text.length}/{maxLength}
        </span>
      </div>
    </div>
  )
}
