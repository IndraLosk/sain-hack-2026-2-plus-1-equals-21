import type { Expert } from '../types'
import { initials } from '../utils'

interface Props {
  experts: Expert[]
  activeId: string
  onChange: (id: string) => void
  disabled?: boolean
  /** true — на вопрос отвечают все эксперты: вкладки становятся выбором, чей ответ смотреть. */
  multi?: boolean
  onMultiChange?: (value: boolean) => void
}

/**
 * Компактный переключатель эксперта: одна строка, аватар + имя, роль — в тултипе.
 * На мобиле — горизонтальный скролл со snap. Справа — сегмент «Один / Все эксперты».
 */
export default function ExpertTabs({ experts, activeId, onChange, disabled, multi, onMultiChange }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap">
      <div
        role="tablist"
        aria-label={multi ? 'Чей ответ показать' : 'Выбор эксперта'}
        className="chat-scroll -mx-1 flex min-w-0 flex-1 snap-x snap-mandatory gap-1.5 overflow-x-auto px-1 pb-0.5"
      >
        {experts.map((expert) => {
          const active = expert.id === activeId
          return (
            <button
              key={expert.id}
              role="tab"
              aria-selected={active}
              disabled={disabled}
              onClick={() => onChange(expert.id)}
              title={`${expert.name} — ${expert.role_badge}`}
              className={`flex shrink-0 snap-start items-center gap-1.5 rounded-full border px-2.5 py-1.5 text-xs transition
                ${active ? 'border-brand-600 bg-brand-600 font-semibold text-white shadow-sm' : 'border-slate-200 bg-white text-slate-700 hover:border-brand-300'}
                disabled:cursor-not-allowed disabled:opacity-60`}
            >
              <span
                className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold
                  ${active ? 'bg-white/20 text-white' : 'bg-slate-100 text-slate-600'}`}
              >
                {initials(expert.name)}
              </span>
              <span className="whitespace-nowrap">{expert.name}</span>
            </button>
          )
        })}
      </div>

      {onMultiChange && (
        <div
          role="group"
          aria-label="Сколько экспертов отвечает"
          className="flex shrink-0 rounded-full border border-slate-200 bg-white p-0.5 text-xs"
        >
          <button
            onClick={() => onMultiChange(false)}
            disabled={disabled}
            aria-pressed={!multi}
            className={`rounded-full px-2.5 py-1 transition disabled:cursor-not-allowed disabled:opacity-60
              ${!multi ? 'bg-slate-900 font-semibold text-white' : 'text-slate-600 hover:text-slate-900'}`}
          >
            Один
          </button>
          <button
            onClick={() => onMultiChange(true)}
            disabled={disabled}
            aria-pressed={!!multi}
            className={`rounded-full px-2.5 py-1 transition disabled:cursor-not-allowed disabled:opacity-60
              ${multi ? 'bg-brand-600 font-semibold text-white' : 'text-slate-600 hover:text-slate-900'}`}
          >
            Все эксперты
          </button>
        </div>
      )}
    </div>
  )
}
