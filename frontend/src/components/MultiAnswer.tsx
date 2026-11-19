import { useMemo, useState } from 'react'
import type { ChatMessage, Expert } from '../types'
import { findUnverifiedNumbers, renderMarkdown } from '../utils'
import SourcesList from './SourcesList'
import SourcesPreview from './SourcesPreview'
import TypingIndicator from './TypingIndicator'

interface Props {
  message: ChatMessage
  experts: Expert[]
  activeId: string
  onSelect: (id: string) => void
}

const COLLAPSE_CHARS = 700

/**
 * Ответ группы экспертов: один вопрос — несколько ответов.
 * Список источников общий (он один на весь вопрос), различается только стиль ответа.
 */
export default function MultiAnswer({ message, experts, activeId, onSelect }: Props) {
  const [expanded, setExpanded] = useState(false)
  const answers = message.answers ?? {}
  const answered = experts.filter((e) => answers[e.id])
  const current = answered.find((e) => e.id === activeId) ?? answered[0]
  const answer = current ? answers[current.id] : undefined

  const unverified = useMemo(
    () =>
      answer?.content && message.sources?.length
        ? findUnverifiedNumbers(
            answer.content,
            message.sources.map((s) => s.snippet ?? ''),
          )
        : [],
    [answer?.content, message.sources],
  )

  const collapsible = !answer?.streaming && (answer?.content.length ?? 0) > COLLAPSE_CHARS

  return (
    <div className="msg-in flex gap-3" data-msg-id={message.id}>
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white">
        С
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <span className="font-semibold text-slate-700">Ответ всех специалистов</span>
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-600">
            источники общие
          </span>
          {message.safetyFlag && (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
              режим безопасности
            </span>
          )}
          <SourcesPreview items={message.sources ?? []} pending={message.streaming} />
          {message.noEvidence && (
            <span className="rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-semibold text-slate-700">
              нет данных в базе
            </span>
          )}
        </div>

        {/* Переключатель вкладок внутри ответа; точка — эксперт ещё печатает */}
        <div role="tablist" aria-label="Ответы специалистов" className="mb-2 flex flex-wrap gap-1.5">
          {answered.map((expert) => {
            const active = current?.id === expert.id
            const item = answers[expert.id]
            return (
              <button
                key={expert.id}
                role="tab"
                aria-selected={active}
                onClick={() => {
                  onSelect(expert.id)
                  setExpanded(false)
                }}
                className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs transition
                  ${active ? 'border-brand-600 bg-brand-50 font-semibold text-brand-700' : 'border-slate-200 bg-white text-slate-600 hover:border-brand-300'}`}
              >
                {item?.streaming ? (
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-brand-500" aria-label="печатает" />
                ) : null}
                {expert.name}
                {item?.error && <span className="text-rose-600">!</span>}
                {item?.warning && <span className="text-amber-600" title={item.warning}>⚠</span>}
              </button>
            )
          })}
        </div>

        <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 text-sm leading-relaxed text-slate-800 shadow-sm">
          {!answer ? (
            <TypingIndicator label="Ищу в статьях…" />
          ) : answer.error ? (
            <p className="text-rose-600">Не удалось получить ответ: {answer.error}</p>
          ) : answer.content ? (
            <>
              <div
                className={`break-words [&_p]:mb-1 ${collapsible && !expanded ? 'relative max-h-56 overflow-hidden' : ''}`}
                dangerouslySetInnerHTML={{ __html: renderMarkdown(answer.content) }}
              />
              {collapsible && (
                <button
                  onClick={() => setExpanded((v) => !v)}
                  className="mt-1 text-xs font-semibold text-brand-700 hover:text-brand-800"
                >
                  {expanded ? 'Свернуть' : 'Показать полностью'}
                </button>
              )}
            </>
          ) : (
            <TypingIndicator label={message.sources?.length ? 'Готовлю ответ…' : 'Ищу в статьях…'} />
          )}
          {answer?.streaming && answer.content && <span className="caret ml-0.5 text-slate-400">▍</span>}
          {answer?.stopped && (
            <span className="ml-1 rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-semibold text-rose-700">
              остановлено
            </span>
          )}
          {unverified.length > 0 && (
            <p className="mt-2 rounded-lg bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
              Число не найдено в источнике: {unverified.slice(0, 4).join(', ')}
            </p>
          )}
          {!answer?.streaming && message.sources && <SourcesList items={message.sources} />}
        </div>
      </div>
    </div>
  )
}
