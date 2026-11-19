import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, streamChat, streamChatMulti } from './api'
import ArticleLibrary from './components/ArticleLibrary'
import Composer from './components/Composer'
import EmptyState from './components/EmptyState'
import ExpertTabs from './components/ExpertTabs'
import MemoryPanel from './components/MemoryPanel'
import MessageBubble from './components/MessageBubble'
import LlmSwitch from './components/LlmSwitch'
import MultiAnswer from './components/MultiAnswer'
import ThemeToggle from './components/ThemeToggle'
import type { Catalog, ChatMessage, HealthInfo, LlmInfo, MemoryItem, PublicConfig, SseEvent } from './types'
import { useTheme } from './useTheme'
import { loadStored, saveStored, uid } from './utils'

const DEFAULT_DISCLAIMER = 'Рекомендации ассистента Сэйн не заменяют консультацию врача'

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [config, setConfig] = useState<PublicConfig | null>(null)
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [memory, setMemory] = useState<MemoryItem[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)

  const [programId, setProgramId] = useState<string>('')
  const [expertId, setExpertId] = useState<string>('')
  const [sessionId, setSessionId] = useState<string>(() => loadStored('sain.session', uid('sess')))
  const [userId] = useState<string>(() => loadStored('sain.user', uid('user')))

  const [llm, setLlm] = useState<LlmInfo | null>(null)
  const [llmBusy, setLlmBusy] = useState(false)
  const { theme, toggle: toggleTheme } = useTheme()

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [panelOpen, setPanelOpen] = useState(false)
  /** Экран «Список статей»: статьи базы, рекомендация ссылки, загрузка файла. */
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [libraryTab, setLibraryTab] = useState<'articles' | 'recommend' | 'upload'>('articles')
  /** true — на вопрос отвечают все специалисты (источники общие, различается стиль). */
  const [multiMode, setMultiMode] = useState(true)
  /** Текст, подставленный в поле ввода кликом по подсказке. */
  const [draft, setDraft] = useState('')

  const streamRef = useRef<{ abort: () => void; done: Promise<void> } | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // ---------------------------------------------------------------- загрузка
  useEffect(() => {
    saveStored('sain.session', sessionId)
  }, [sessionId])
  useEffect(() => {
    saveStored('sain.user', userId)
  }, [userId])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [cat, cfg] = await Promise.all([api.catalog(), api.config()])
        if (cancelled) return
        setCatalog(cat)
        setConfig(cfg)
        setProgramId(cat.programs[0]?.id ?? '')
        const savedExpert = loadStored<string>('sain.expert', '')
        const expert = cat.experts.find((e) => e.id === savedExpert) ?? cat.experts[0]
        setExpertId(expert.id)
      } catch (err) {
        if (!cancelled) setLoadError((err as Error).message)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (expertId) saveStored('sain.expert', expertId)
  }, [expertId])

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health())
    } catch {
      /* тихо игнорируем */
    }
  }, [])

  const refreshMemory = useCallback(async () => {
    try {
      const res = await api.memory(userId)
      setMemory(res.items ?? [])
    } catch {
      /* тихо игнорируем */
    }
  }, [userId])

  const refreshLlm = useCallback(async () => {
    try {
      setLlm(await api.llm())
    } catch {
      /* тихо игнорируем */
    }
  }, [])

  const switchLlm = useCallback(
    async (provider: string) => {
      setLlmBusy(true)
      try {
        setLlm(await api.setLlm(provider))
        await refreshHealth()
      } catch (err) {
        setLoadError(`Не удалось переключить модель: ${(err as Error).message}`)
      } finally {
        setLlmBusy(false)
      }
    },
    [refreshHealth],
  )

  useEffect(() => {
    void refreshHealth()
    void refreshMemory()
    void refreshLlm()
  }, [refreshHealth, refreshMemory, refreshLlm])

  // ---------------------------------------------------------------- автоскролл
  const lastMsg = messages[messages.length - 1]
  const lastLen = lastMsg
    ? lastMsg.answers
      ? Object.values(lastMsg.answers).reduce((sum, a) => sum + a.content.length, 0)
      : lastMsg.content.length
    : 0
  const scrolledFor = useRef<string | null>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    // Первый токен нового ответа: прокручиваем к началу пузыря, а не к хвосту.
    if (lastMsg && lastLen > 0 && scrolledFor.current !== lastMsg.id) {
      scrolledFor.current = lastMsg.id
      const bubbles = el.querySelectorAll('[data-msg-id]')
      const target = bubbles[bubbles.length - 1] as HTMLElement | undefined
      if (target) {
        el.scrollTop = Math.max(0, target.offsetTop - 12)
        return
      }
    }
    el.scrollTop = el.scrollHeight
  }, [messages.length, lastLen, lastMsg])

  // ---------------------------------------------------------------- стрим
  const handleEvent = useCallback((event: SseEvent) => {
    setMessages((prev) => {
      const next = [...prev]
      const last = next[next.length - 1]
      if (!last || last.role !== 'assistant') return prev
      switch (event.type) {
        case 'token':
          if (last.answers) {
            // мульти-режим: токен адресован конкретному эксперту
            const id = event.expert_id ?? ''
            const answer = last.answers[id]
            if (answer) answer.content += event.content
          } else {
            last.content += event.content
          }
          break
        case 'expert_start':
          if (last.answers && !last.answers[event.expert_id]) {
            last.answers[event.expert_id] = {
              expertId: event.expert_id,
              content: '',
              streaming: true,
              done: false,
            }
          }
          break
        case 'expert_done': {
          const answer = last.answers?.[event.expert_id]
          if (answer) {
            answer.streaming = false
            answer.done = true
            answer.stopped = event.stopped
          }
          break
        }
        case 'sources':
          last.sources = event.items
          break
        case 'meta':
          last.noEvidence = event.no_evidence
          last.safetyFlag = event.is_new_knowledge
          break
        case 'warning':
          if (last.answers && event.expert_id) {
            const answer = last.answers[event.expert_id]
            if (answer) answer.warning = event.message
          } else {
            last.warning = event.message
          }
          break
        case 'stopped':
          if (last.answers) Object.values(last.answers).forEach((a) => (a.streaming = false))
          else last.stopped = true
          last.streaming = false
          break
        case 'error':
          if (last.answers) {
            const ids = Object.keys(last.answers)
            const id = ids[ids.length - 1]
            if (id) last.answers[id].error = event.message
          } else {
            last.error = event.message
          }
          last.streaming = false
          break
        case 'done':
          if (last.answers) Object.values(last.answers).forEach((a) => (a.streaming = false))
          last.streaming = false
          break
      }
      return next
    })
  }, [])

  const send = useCallback(
    (text: string) => {
      if (!programId || !expertId || streaming) return
      const userMsg: ChatMessage = { id: uid('m'), role: 'user', content: text }

      // Мульти-режим: вопрос уходит всем экспертам, источники у них общие.
      const expertIds = catalog?.experts.map((e) => e.id) ?? []
      const multi = multiMode && expertIds.length > 1
      const botMsg: ChatMessage = multi
        ? {
            id: uid('m'),
            role: 'assistant',
            content: '',
            replyTo: userMsg.id,
            answers: {},
            streaming: true,
            sources: [],
          }
        : {
            id: uid('m'),
            role: 'assistant',
            content: '',
            expertId,
            streaming: true,
            sources: [],
          }
      setMessages((prev) => [...prev, userMsg, botMsg])
      setStreaming(true)

      const onError = (message: string) =>
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === 'assistant') {
            last.streaming = false
            if (last.answers) Object.values(last.answers).forEach((a) => (a.streaming = false))
            else last.error = message
          }
          return next
        })

      const handle = multi
        ? streamChatMulti(
            { session_id: sessionId, program_id: programId, expert_ids: expertIds, message: text, user_id: userId },
            { onEvent: handleEvent, onError },
          )
        : streamChat(
            { session_id: sessionId, program_id: programId, expert_id: expertId, message: text, user_id: userId },
            { onEvent: handleEvent, onError },
          )
      streamRef.current = handle
      void handle.done.finally(() => {
        setStreaming(false)
        streamRef.current = null
        setMessages((prev) =>
          prev.map((m) => {
            if (!m.streaming) return m
            if (m.answers) {
              Object.values(m.answers).forEach((a) => (a.streaming = false))
            }
            return { ...m, streaming: false }
          }),
        )
        void refreshMemory()
        void refreshHealth()
      })
    },
    [catalog, expertId, handleEvent, multiMode, programId, refreshHealth, refreshMemory, sessionId, streaming, userId],
  )

  // ---------------------------------------------------------------- стоп / сброс
  const stop = useCallback(() => {
    streamRef.current?.abort()
  }, [])

  const reset = useCallback(() => {
    streamRef.current?.abort()
    setMessages([])
    setSessionId(uid('sess'))
  }, [])

  const knowledgeLabel = useMemo(() => {
    const chunks = health?.vector_store?.chunks ?? 0
    if (!chunks) return null
    const articles = Number(health?.vector_store?.articles ?? 0)
    const fmt = (n: number) => n.toLocaleString('ru-RU')
    return articles > 0 ? `${fmt(articles)} статей · ${fmt(chunks)} фрагментов` : `${fmt(chunks)} фрагментов`
  }, [health])

  const activeExpert = useMemo(() => catalog?.experts.find((e) => e.id === expertId), [catalog, expertId])

  // ---------------------------------------------------------------- рендер
  return (
    <div className="mx-auto flex h-[100dvh] w-full max-w-[1100px] flex-col bg-slate-50">
      <header className="flex items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3">
        <div className="min-w-0">
          <h1 className="truncate text-base font-semibold text-slate-900">Сэйн · ассистент по научным статьям</h1>
          <p className="truncate text-xs text-slate-500">{config?.disclaimer ?? DEFAULT_DISCLAIMER}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
          <button
            onClick={reset}
            title="Начать чат заново: история очищается"
            className="rounded-xl border border-slate-300 px-2.5 py-1.5 text-xs text-slate-600 hover:bg-slate-100 sm:px-3"
          >
            <span className="hidden sm:inline">Новый чат</span>
            <span aria-hidden className="sm:hidden">
              +
            </span>
          </button>
          <span
            className="hidden items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-[11px] text-slate-500 md:flex"
            title={knowledgeLabel ?? ''}
          >
            <span aria-hidden>📚</span>
            {knowledgeLabel ?? 'база знаний'}
          </span>
          <button
            onClick={() => {
              setLibraryTab('articles')
              setLibraryOpen(true)
            }}
            title="Список статей: что есть в базе и какие куски использовал агент"
            className="rounded-xl border border-slate-300 px-2.5 py-1.5 text-xs text-slate-600 hover:bg-slate-100 sm:px-3"
          >
            <span aria-hidden>📄</span> <span className="hidden sm:inline">Статьи</span>
          </button>
          <button
            onClick={() => {
              setLibraryTab('recommend')
              setLibraryOpen(true)
            }}
            title="Порекомендовать статью по ссылке — она попадёт в базу знаний"
            className="rounded-xl border border-slate-300 px-2.5 py-1.5 text-xs text-slate-600 hover:bg-slate-100 sm:px-3"
          >
            <span aria-hidden>🔗</span> <span className="hidden md:inline">Порекомендовать статью</span>
          </button>
          <button
            onClick={() => {
              setLibraryTab('upload')
              setLibraryOpen(true)
            }}
            title="Загрузить файл или документ в векторную базу"
            className="rounded-xl border border-slate-300 px-2.5 py-1.5 text-xs text-slate-600 hover:bg-slate-100 sm:px-3"
          >
            <span aria-hidden>📎</span> <span className="hidden md:inline">Загрузить файл</span>
          </button>
          <LlmSwitch info={llm} onChange={switchLlm} disabled={streaming} busy={llmBusy} />
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
          <button
            onClick={() => {
              setPanelOpen(true)
              void refreshMemory()
              void refreshHealth()
            }}
            className="rounded-xl border border-slate-300 px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-100"
          >
            Память
          </button>
        </div>
      </header>

      {/* Выбор эксперта: одна строка + сегмент «Один / Все эксперты» */}
      <section className="border-b border-slate-200 bg-white/60 px-4 py-2.5">
        {catalog && (
          <ExpertTabs
            experts={catalog.experts}
            activeId={expertId}
            onChange={setExpertId}
            disabled={streaming}
            multi={multiMode}
            onMultiChange={setMultiMode}
          />
        )}
      </section>

      {/* Лента сообщений */}
      <main ref={scrollRef} className="chat-scroll flex flex-1 flex-col overflow-y-auto px-4 py-4">
        {messages.length === 0 && <EmptyState onPick={setDraft} knowledgeLabel={knowledgeLabel ?? undefined} />}
        <div className="mt-auto space-y-4">
        {messages.map((message) =>
          message.answers ? (
            <MultiAnswer
              key={message.id}
              message={message}
              experts={catalog?.experts ?? []}
              activeId={expertId}
              onSelect={setExpertId}
            />
          ) : (
            <MessageBubble
              key={message.id}
              message={message}
              expert={catalog?.experts.find((e) => e.id === message.expertId) ?? activeExpert}
            />
          ),
        )}
        </div>
      </main>

      {loadError && (
        <div className="border-t border-rose-200 bg-rose-50 px-4 py-2 text-xs text-rose-700">
          Ошибка загрузки каталога: {loadError}
        </div>
      )}

      <Composer
        onSend={send}
        onStop={stop}
        streaming={streaming}
        disabled={!catalog || !expertId}
        draft={draft}
        onDraftConsumed={() => setDraft('')}
      />

      <footer className="border-t border-slate-200 bg-white/80 px-4 py-2 text-center text-[11px] text-slate-500">
        Рекомендация ассистента не заменяет консультацию врача
      </footer>

      {libraryOpen && (
        <ArticleLibrary
          open={libraryOpen}
          onClose={() => setLibraryOpen(false)}
          programId={programId}
          userId={userId}
          initialTab={libraryTab}
          onChanged={() => void refreshHealth()}
        />
      )}

      {panelOpen && (
        <MemoryPanel
          open={panelOpen}
          memory={memory}
          health={health}
          onClose={() => setPanelOpen(false)}
          onClearMemory={() => {
            void api
              .clearMemory(userId)
              .then(() => setMemory([]))
              .catch(() => undefined)
          }}
        />
      )}
    </div>
  )
}
