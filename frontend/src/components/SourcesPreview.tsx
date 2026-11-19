import type { SourceItem } from '../types'

interface Props {
  items: SourceItem[]
  /** true — ответ ещё стримится: показываем только сводку, без списка. */
  pending?: boolean
}

/** Событие `sources` приходит до первого токена — показываем сводку сразу. */
export default function SourcesPreview({ items, pending }: Props) {
  if (!items.length || !pending) return null
  const articles = new Set(items.map((s) => s.article_id)).size
  const label =
    articles > 1
      ? `Нашёл ${items.length} фрагм. из ${articles} статей`
      : `Нашёл ${items.length} фрагм. из 1 статьи`

  return (
    <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">
      {label}
    </span>
  )
}
