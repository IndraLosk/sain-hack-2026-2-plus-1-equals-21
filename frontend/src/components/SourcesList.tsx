import { useState } from 'react'
import type { SourceItem } from '../types'
import { formatDate } from '../utils'

/** Раскрывающийся список источников: в заголовке — превью названий, внутри — всё. */
export default function SourcesList({ items }: { items: SourceItem[] }) {
  const [open, setOpen] = useState(false)
  if (!items.length) return null

  // Превью: 1–2 названия + год, чтобы свёрнутый блок был информативным.
  const preview = items
    .slice(0, 2)
    .map((src) => {
      const year = src.date ? new Date(src.date).getFullYear() : null
      const title = src.title.length > 42 ? `${src.title.slice(0, 42).trim()}…` : src.title
      return year ? `${title} (${year})` : title
    })
    .join(' · ')
  const rest = items.length > 2 ? ` +${items.length - 2}` : ''

  return (
    <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50/70">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-start justify-between gap-2 px-3 py-2 text-left text-xs hover:text-slate-900"
      >
        <span className="min-w-0">
          <span className="font-medium text-slate-600">Источники · {items.length}</span>
          <span className="ml-1.5 text-slate-400">
            {preview}
            {rest}
          </span>
        </span>
        <span className="shrink-0 text-slate-400">{open ? 'Скрыть' : 'Показать'}</span>
      </button>
      {open && (
        <ul className="space-y-2 border-t border-slate-200 px-3 py-2">
          {items.map((src, index) => (
            <li key={`${src.article_id}:${src.section_id ?? ''}`} className="text-xs text-slate-600">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-[10px] text-slate-400">[{index + 1}]</span>
                <span className="font-medium text-slate-800">{src.title}</span>
                {src.is_new && (
                  <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">
                    новое
                  </span>
                )}
                {src.date && <span className="text-slate-400">{formatDate(src.date)}</span>}
              </div>
              {src.snippet && <p className="mt-1 line-clamp-3 text-slate-500">{src.snippet}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
