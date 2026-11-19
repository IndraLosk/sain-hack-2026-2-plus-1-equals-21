import type { HealthInfo, MemoryItem } from '../types'

interface Props {
  open: boolean
  onClose: () => void
  health: HealthInfo | null
  memory: MemoryItem[]
  onClearMemory: () => void
}

const MEMORY_LABELS: Record<string, string> = {
  sleep_schedule: 'Режим сна',
  sleep_problem: 'Проблема со сном',
  constraints: 'Ограничения',
  goal: 'Цель',
  context: 'Контекст',
  preference: 'Предпочтения',
}

/** Панель «Что я о вас помню» + состояние системы. Только работающие действия. */
export default function MemoryPanel({ open, onClose, health, memory, onClearMemory }: Props) {
  if (!open) return null

  const llmOk = health?.llm.ok
  const chunks = health?.vector_store.chunks ?? 0

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-slate-900/30" onClick={onClose}>
      <aside
        className="h-full w-full max-w-sm overflow-y-auto bg-white p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
        aria-label="Память и состояние"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-slate-900">Что я о вас помню</h2>
          <button onClick={onClose} className="text-xl text-slate-400 hover:text-slate-700" aria-label="Закрыть">
            ×
          </button>
        </div>
        <p className="mt-1 text-xs text-slate-500">
          Ассистент сохраняет только устойчивые факты (режим, ограничения, цель). Дозировки и диагнозы не сохраняются.
        </p>

        {memory.length === 0 ? (
          <p className="mt-4 rounded-xl bg-slate-50 p-3 text-sm text-slate-500">
            Пока пусто — факты появятся после пары сообщений о вашей ситуации.
          </p>
        ) : (
          <ul className="mt-4 space-y-2">
            {memory.map((item) => (
              <li key={item.key} className="rounded-xl border border-slate-200 p-3">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  {MEMORY_LABELS[item.key] ?? item.key}
                </div>
                <div className="text-sm text-slate-800">{item.value}</div>
              </li>
            ))}
          </ul>
        )}

        {memory.length > 0 && (
          <button
            onClick={onClearMemory}
            className="mt-3 w-full rounded-xl border border-rose-200 px-3 py-2 text-sm text-rose-700 hover:bg-rose-50"
          >
            Забыть обо мне
          </button>
        )}

        <h3 className="mt-6 text-sm font-semibold text-slate-900">Состояние системы</h3>
        <dl className="mt-2 space-y-1.5 text-xs">
          <div className="flex justify-between gap-3">
            <dt className="text-slate-500">Модель</dt>
            <dd className="truncate font-medium text-slate-800">{health?.llm.model ?? '—'}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-500">LLM</dt>
            <dd className={llmOk ? 'text-emerald-600' : 'text-amber-600'}>
              {llmOk ? 'доступна' : 'недоступна'}
            </dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-500">Эмбеддинги</dt>
            <dd className="truncate font-medium text-slate-800">
              {health?.vector_store.embedding_model ?? 'bge-m3'} · ONNX
            </dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-500">Фрагментов в базе</dt>
            <dd className="font-medium text-slate-800">{chunks}</dd>
          </div>
          {health?.database?.open_gaps !== undefined && (
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500">Запросов без ответа</dt>
              <dd className="font-medium text-slate-800">{String(health.database.open_gaps)}</dd>
            </div>
          )}
        </dl>
      </aside>
    </div>
  )
}
