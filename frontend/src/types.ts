/** Типы API (совпадают с Pydantic-схемами backend/schemas.py). */

export interface Program {
  id: string
  title: string
  description: string
  cover_color: string
  default_expert_id: string
}

export interface Expert {
  id: string
  name: string
  role_badge: string
  avatar_url: string
  system_style_prompt: string
}

export interface Catalog {
  version: string
  programs: Program[]
  experts: Expert[]
}

export interface SourceItem {
  article_id: string
  section_id?: string | null
  title: string
  date?: string | null
  is_new: boolean
  snippet: string
}

export interface ExpertAnswer {
  expertId: string
  content: string
  streaming: boolean
  done: boolean
  stopped?: boolean
  error?: string
  /** Модель не ответила — показан fallback по фактам. */
  warning?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  /** Для режима «все специалисты»: один вопрос — несколько ответов по экспертам. */
  answers?: Record<string, ExpertAnswer>
  /** id сообщения пользователя, на которое отвечает группа. */
  replyTo?: string
  expertId?: string
  safetyFlag?: boolean
  sources?: SourceItem[]
  noEvidence?: boolean
  streaming?: boolean
  stopped?: boolean
  error?: string
  /** Модель не ответила — показан fallback по фактам. */
  warning?: string
  /** Номера из ответа, которых нет в источниках (проверка без LLM). */
  unverifiedNumbers?: string[]
}

export interface HealthInfo {
  status: string
  llm: { model: string; ok: boolean; provider?: string; error?: string; model_available?: boolean }
  vector_store: { collection: string; chunks: number; articles?: number; embedding_model: string }
  database?: Record<string, number | string>
}

export interface LlmOption {
  id: string
  label: string
  hint: string
  configured: boolean
}

export interface LlmInfo {
  provider: string
  model: string
  configured: boolean
  options: LlmOption[]
}

export interface PublicConfig {
  llm_model: string
  llm_provider?: string
  embedding_model: string
  disclaimer: string
  freshness_days: number
  max_message_length: number
}

export interface MemoryItem {
  key: string
  value: string
  confidence: number
  updated_at?: number
}

export type SseEvent =
  | { type: 'safety_flag'; triggered: boolean }
  | { type: 'sources'; items: SourceItem[] }
  | {
      type: 'meta'
      session_id: string
      expert_id: string
      expert_ids?: string[]
      program_id: string
      is_new_knowledge: boolean
      no_evidence: boolean
      multi?: boolean
    }
  | { type: 'token'; content: string; expert_id?: string }
  | { type: 'expert_start'; expert_id: string; name: string }
  | { type: 'expert_done'; expert_id: string; stopped: boolean }
  | { type: 'warning'; message: string; reason?: string; expert_id?: string }
  | { type: 'done' }
  | { type: 'stopped' }
  | { type: 'error'; message: string }

/** Рекомендованная пользователем статья (SQLite) — кнопка «Порекомендовать статью». */
export interface RecommendedArticle {
  id: string
  url: string
  title: string
  program_id: string
  user_id?: string | null
  comment: string
  status: 'pending' | 'indexed' | 'failed'
  chunks: number
  error: string
  source_kind: string
  created_at: number
  updated_at: number
}

/** Фрагмент статьи, который агент реально подставил в ответ. */
export interface UsedChunk {
  id: string
  article_id: string
  article_title: string
  section_id: string
  section_title: string
  text: string
  query: string
  session_id?: string | null
  expert_id?: string | null
  similarity: number
  used_at: number
}

/** Элемент экрана «Список статей». */
export interface LibraryArticle {
  article_id: string
  title: string
  url: string
  program_id: string
  status: 'indexed' | 'pending' | 'failed' | string
  chunks: number
  source_kind: string
  added_at: number
  used_chunks: UsedChunk[]
}

export interface LibraryResponse {
  articles: LibraryArticle[]
  total: number
  used_chunks_total: number
  vector_chunks: number
}
