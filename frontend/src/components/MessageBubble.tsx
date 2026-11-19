import { useMemo, useState } from 'react'
import type { ChatMessage, Expert } from '../types'
import { findUnverifiedNumbers, renderMarkdown } from '../utils'
import SourcesList from './SourcesList'
import SourcesPreview from './SourcesPreview'
import TypingIndicator from './TypingIndicator'

interface Props {
  message: ChatMessage
  expert?: Expert
}

/** Порог, после которого длинный ответ сворачивается. */
const COLLAPSE_CHARS = 700

/** Одно сообщение диалога: пользователь (справа) или эксперт (слева). */
export default function MessageBubble({ message, expert }: Props) {
  const [expanded, setExpanded] = useState(false)
  const isUser = message.role === 'user'

  const unverified = useMemo(
    () =>
      message.content && message.sources?.length
        ? findUnverifiedNumbers(
            message.content,
            message.sources.map((s) => s.snippet ?? ''),
          )
        : [],
    [message.content, message.sources],
  )

  if (isUser) {
    return (
      <div className="msg-in flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-sm text-white shadow-sm">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </div>
    )
  }

  const collapsible = !message.streaming && message.content.length > COLLAPSE_CHARS

  return (
    <div className="msg-in flex gap-3" data-msg-id={message.id}>
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white">
        С
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <span className="font-semibold text-slate-700">{expert?.name ?? 'Эксперт'}</span>
          {message.safetyFlag && (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
              режим безопасности
            </span>
          )}
          {message.warning && (
            <span
              className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800"
              title={message.warning}
            >
              ответ без модели
            </span>
          )}
          {message.noEvidence && (
            <span className="rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-semibold text-slate-700">
              нет данных в базе
            </span>
          )}
          <SourcesPreview items={message.sources ?? []} pending={message.streaming} />
          {message.stopped && (
            <span className="rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-semibold text-rose-700">
              остановлено
            </span>
          )}
        </div>

        <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 text-sm leading-relaxed text-slate-800 shadow-sm">
          {message.error ? (
            <p className="text-rose-600">Не удалось получить ответ: {message.error}</p>
          ) : message.content ? (
            <>
              <div
                className={`break-words [&_p]:mb-1 ${collapsible && !expanded ? 'relative max-h-56 overflow-hidden' : ''}`}
                dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
              />
              {collapsible && !expanded && (
                <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-white" />
              )}
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
          {message.streaming && message.content && <span className="caret ml-0.5 text-slate-400">▍</span>}
          {unverified.length > 0 && (
            <p className="mt-2 rounded-lg bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
              Число не найдено в источнике: {unverified.slice(0, 4).join(', ')}
            </p>
          )}
          {!message.streaming && message.sources && <SourcesList items={message.sources} />}
        </div>
      </div>
    </div>
  )
}
