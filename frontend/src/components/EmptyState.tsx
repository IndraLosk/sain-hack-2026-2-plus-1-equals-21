interface Props {
  onPick: (text: string) => void
  /** «128 статей · 1 240 фрагментов» — откуда ассистент берёт ответы. */
  knowledgeLabel?: string
}

const SUGGESTIONS: { icon: string; title: string; text: string }[] = [
  { icon: '🌙', title: 'Не могу уснуть', text: 'Как быстрее уснуть, если мысли не выключаются?' },
  { icon: '🍷', title: 'Вино и сон', text: 'Почему после бокала вина сон хуже?' },
  { icon: '☕', title: 'Кофеин днём', text: 'Сколько кофеина можно выпить днём, чтобы не мешать сну?' },
  { icon: '📱', title: 'Экраны вечером', text: 'Насколько вреден экран телефона перед сном и что помогает?' },
]

/** Первый экран: подсказки карточками по центру, а не мелкие чипы внизу. */
export default function EmptyState({ onPick, knowledgeLabel }: Props) {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center px-1 py-6 text-center">
      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-brand-600 text-lg font-semibold text-white shadow-sm">
        С
      </div>
      <h2 className="mt-3 text-lg font-semibold text-slate-900">Здравствуйте, я Сэйн</h2>
      <p className="mt-1 max-w-md text-sm leading-relaxed text-slate-600">
        Отвечаю только по научным статьям из базы, не назначаю дозировки и диагнозы.
      </p>
      {knowledgeLabel && (
        <span className="mt-2 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] text-slate-500">
          {knowledgeLabel}
        </span>
      )}

      <div className="mt-5 grid w-full gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((item) => (
          <button
            key={item.text}
            onClick={() => onPick(item.text)}
            className="group flex items-start gap-2.5 rounded-xl border border-slate-200 bg-white p-3 text-left transition hover:border-brand-300 hover:shadow-sm"
          >
            <span className="text-base leading-none" aria-hidden>
              {item.icon}
            </span>
            <span className="min-w-0">
              <span className="block text-xs font-semibold text-slate-800">{item.title}</span>
              <span className="mt-0.5 block text-[11px] leading-snug text-slate-500 group-hover:text-slate-600">
                {item.text}
              </span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
