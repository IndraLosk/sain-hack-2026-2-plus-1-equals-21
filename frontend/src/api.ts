/**
 * API-клиент: каталог, здоровье, память + SSE-стрим чата с поддержкой отмены.
 * Все запросы идут на тот же origin (FastAPI раздаёт фронтенд).
 */
import type {
  Catalog,
  HealthInfo,
  LibraryResponse,
  LlmInfo,
  MemoryItem,
  PublicConfig,
  RecommendedArticle,
  SseEvent,
} from './types'
import type { ChatMessage } from './types'

export type HistoryMessage = Pick<ChatMessage, 'role' | 'content' | 'expertId' | 'safetyFlag' | 'sources'>

const API = '/api/v1'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      detail = body?.detail ?? detail
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  catalog: () => request<Catalog>('/catalog'),
  config: () => request<PublicConfig>('/config'),
  health: () => request<HealthInfo>('/health'),
  history: (sessionId: string) =>
    request<{ session_id: string; messages: HistoryMessage[] }>(`/chat/history?session_id=${encodeURIComponent(sessionId)}`),
  memory: (userId: string) => request<{ user_id: string; items: MemoryItem[] }>(`/memory?user_id=${encodeURIComponent(userId)}`),
  clearMemory: (userId: string) => request<{ removed: number }>(`/memory?user_id=${encodeURIComponent(userId)}`, { method: 'DELETE' }),
  clearHistory: (sessionId: string) =>
    request<{ status: string }>(`/chat/history?session_id=${encodeURIComponent(sessionId)}`, { method: 'DELETE' }),
  llm: () => request<LlmInfo>('/llm'),
  setLlm: (provider: string) =>
    request<LlmInfo>('/llm', { method: 'POST', body: JSON.stringify({ provider }) }),
  stop: (sessionId: string) =>
    request<{ status: string }>(`/chat/stop?session_id=${encodeURIComponent(sessionId)}`, { method: 'POST' }),

  /** «Список статей»: статьи базы + рекомендованные + использованные фрагменты. */
  library: (programId?: string) =>
    request<LibraryResponse>(`/library${programId ? `?program_id=${encodeURIComponent(programId)}` : ''}`),

  /** «Порекомендовать статью»: ссылка сохраняется в SQLite и индексируется в фоне. */
  recommendArticle: (payload: {
    url: string
    title?: string
    program_id: string
    user_id?: string
    comment?: string
  }) => request<RecommendedArticle>('/articles/recommend', { method: 'POST', body: JSON.stringify(payload) }),

  deleteArticle: (articleId: string) =>
    request<{ status: string }>(`/articles/${encodeURIComponent(articleId)}`, { method: 'DELETE' }),

  /** «Загрузить файл/документ в векторную базу»: multipart, без Content-Type вручную. */
  uploadDocument: async (file: File, programId: string, title: string, userId: string) => {
    const form = new FormData()
    form.append('file', file)
    form.append('program_id', programId)
    form.append('title', title)
    form.append('user_id', userId)
    const res = await fetch(`${API}/documents/upload`, { method: 'POST', body: form })
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`
      try {
        const body = await res.json()
        detail = body?.detail ?? detail
      } catch {
        /* ignore */
      }
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    }
    return (await res.json()) as { article_id: string; title: string; message: string }
  },
}

/** Один вопрос — всем экспертам. Бэкенд ищет источники один раз, поэтому они общие. */
export function streamChatMulti(
  payload: { session_id: string; program_id: string; expert_ids: string[]; message: string; user_id: string },
  handlers: StreamHandlers,
): { abort: () => void; done: Promise<void> } {
  return openStream(`${API}/chat/stream-multi`, payload, handlers)
}

export interface StreamHandlers {
  onEvent: (event: SseEvent) => void
  onError?: (message: string) => void
}

/**
 * Отправляет сообщение и читает SSE-поток.
 * Возвращает функцию отмены (AbortController): прерывает HTTP-соединение,
 * бэкенд при этом помечает генерацию остановленной.
 */
export function streamChat(
  payload: { session_id: string; program_id: string; expert_id: string; message: string; user_id: string },
  handlers: StreamHandlers,
): { abort: () => void; done: Promise<void> } {
  return openStream(`${API}/chat/stream`, payload, handlers)
}

/** Общий SSE-ридер: одинаковый разбор кадров для одиночного и мульти-режима. */
function openStream(
  url: string,
  payload: Record<string, unknown>,
  handlers: StreamHandlers,
): { abort: () => void; done: Promise<void> } {
  const controller = new AbortController()

  const done = (async () => {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) {
        throw new Error(`Не удалось начать ответ (${res.status})`)
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done: finished } = await reader.read()
        if (finished) break
        buffer += decoder.decode(value, { stream: true })
        const frames = buffer.split('\n\n')
        buffer = frames.pop() ?? ''
        for (const frame of frames) {
          const line = frame.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue
          const raw = line.slice(5).trim()
          if (!raw) continue
          try {
            handlers.onEvent(JSON.parse(raw) as SseEvent)
          } catch {
            /* пропускаем битый кадр */
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      handlers.onError?.((err as Error).message || 'Ошибка сети')
    }
  })()

  return { abort: () => controller.abort(), done }
}
