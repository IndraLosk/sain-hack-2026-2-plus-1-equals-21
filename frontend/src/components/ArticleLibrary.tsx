import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { LibraryArticle, LibraryResponse, UsedChunk } from '../types'
import { formatDate } from '../utils'

interface Props {
  open: boolean
  onClose: () => void
  programId: string
  userId: string
  /** С какой вкладки открыть: статьи / рекомендация ссылки / загрузка файла. */
  initialTab?: Tab
  /** Колбэк после изменения базы знаний — обновляет счётчики в шапке. */
  onChanged?: () => void
}

type Tab = 'articles' | 'recommend' | 'upload'

const STATUS_LABEL: Record<string, { text: string; className: string }> = {
  indexed: { text: 'в базе', className: 'bg-emerald-100 text-emerald-700' },
  pending: { text: 'индексируется', className: 'bg-amber-100 text-amber-700' },
  failed: { text: 'ошибка', className: 'bg-rose-100 text-rose-700' },
}

function timeAgo(ts: number): string {
  if (!ts) return ''
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'только что'
  if (diff < 3600) return `${Math.floor(diff / 60)} мин назад`
  if (diff < 86400) return `${Math.floor(diff / 3600)} ч назад`
  return formatDate(new Date(ts * 1000).toISOString())
}

/**
 * Экран «Список статей» с тремя вкладками:
 *  • Статьи — что есть в базе и какие куски агент реально использовал в ответах;
 *  • Порекомендовать — ссылка на статью → SQLite → векторная база;
 *  • Загрузить — файл/документ (.pdf/.txt/.md) → векторная база.
 */
export default function ArticleLibrary({ open, onClose, programId, userId, initialTab, onChanged }: Props) {
  const [tab, setTab] = useState<Tab>(initialTab ?? 'articles')
  const [data, setData] = useState<LibraryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)

  const [url, setUrl] = useState('')
  const [title, setTitle] = useState('')
  const [comment, setComment] = useState('')
  const [busy, setBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const [fileName, setFileName] = useState('')

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await api.library(programId || undefined))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [programId])

  useEffect(() => {
    if (open) {
      if (initialTab) setTab(initialTab)
      void refresh()
    }
  }, [open, initialTab, refresh])

  const submitLink = async () => {
    if (!url.trim() || busy) return
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const res = await api.recommendArticle({
        url: url.trim(),
        title: title.trim(),
        program_id: programId || 'sleep_recovery',
        user_id: userId,
        comment: comment.trim(),
      })
      setNotice(`Ссылка сохранена: «${res.title}». Индексация в векторную базу идёт в фоне.`)
      setUrl('')
      setTitle('')
      setComment('')
      setTab('articles')
      await refresh()
      onChanged?.()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const submitFile = async () => {
    const file = fileRef.current?.files?.[0]
    if (!file || busy) return
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const res = await api.uploadDocument(file, programId || 'sleep_recovery', title.trim(), userId)
      setNotice(res.message)
      setTitle('')
      setFileName('')
      if (fileRef.current) fileRef.current.value = ''
      setTab('articles')
      await refresh()
      onChanged?.()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const remove = async (articleId: string) => {
    try {
      await api.deleteArticle(articleId)
      await refresh()
      onChanged?.()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const articles = data?.articles ?? []
  const totalUsed = useMemo(
    () => articles.reduce((sum, a) => sum + (a.used_chunks?.length ?? 0), 0),
    [articles],
  )

  if (!open) return null

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-slate-900/30" onClick={onClose}>
      <aside
        className="flex h-full w-full max-w-2xl flex-col overflow-hidden bg-white shadow-xl"
        onClick={(e) => e.stopPropagation()}
        aria-label="Список статей"
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Список статей</h2>
            <p className="text-[11px] text-slate-500">
              {data ? `${data.total} статей · ${totalUsed} использованных фрагментов` : 'загружаю…'}
            </p>
          </div>
          <button onClick={onClose} className="text-xl text-slate-400 hover:text-slate-700" aria-label="Закрыть">
            ×
          </button>
        </div>

        <div className="flex gap-1 border-b border-slate-200 px-3 py-2">
          {(
            [
              ['articles', 'Статьи'],
              ['recommend', 'Порекомендовать статью'],
              ['upload', 'Загрузить файл'],
            ] as [Tab, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                tab === id ? 'bg-brand-600 text-white' : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {error && (
            <div className="mb-3 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
              {error}
            </div>
          )}
          {notice && (
            <div className="mb-3 rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
              {notice}
            </div>
          )}

          {tab === 'recommend' && (
            <div className="space-y-3">
              <p className="text-xs text-slate-500">
                Вставьте ссылку на научную статью. Она сохранится в базе данных и будет проиндексирована в векторную
                базу — эксперты начнут её учитывать.
              </p>
              <label className="block text-xs font-medium text-slate-700">
                Ссылка на статью
                <input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://pubmed.ncbi.nlm.nih.gov/…"
                  className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm text-slate-900 outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                />
              </label>
              <label className="block text-xs font-medium text-slate-700">
                Название (необязательно)
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Например: Light exposure and sleep onset"
                  className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm text-slate-900 outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                />
              </label>
              <label className="block text-xs font-medium text-slate-700">
                Комментарий (необязательно)
                <textarea
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  rows={3}
                  placeholder="Почему статья полезна"
                  className="mt-1 w-full resize-none rounded-xl border border-slate-300 px-3 py-2 text-sm text-slate-900 outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                />
              </label>
              <button
                onClick={submitLink}
                disabled={busy || !url.trim()}
                className={`w-full rounded-xl px-3 py-2 text-sm font-semibold transition ${
                  busy || !url.trim()
                    ? 'cursor-not-allowed bg-slate-200 text-slate-400'
                    : 'bg-brand-600 text-white hover:bg-brand-700'
                }`}
              >
                {busy ? 'Сохраняю…' : 'Порекомендовать статью'}
              </button>
            </div>
          )}

          {tab === 'upload' && (
            <div className="space-y-3">
              <p className="text-xs text-slate-500">
                Документ будет нарезан на фрагменты и положен в векторную базу. Поддерживаются .pdf, .txt, .md, .csv.
              </p>
              <label className="block text-xs font-medium text-slate-700">
                Файл
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.txt,.md,.markdown,.csv,.json"
                  onChange={(e) => setFileName(e.target.files?.[0]?.name ?? '')}
                  className="mt-1 w-full cursor-pointer rounded-xl border border-dashed border-slate-300 px-3 py-6 text-sm text-slate-600 file:mr-3 file:rounded-lg file:border-0 file:bg-brand-600 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-white hover:border-brand-400"
                />
              </label>
              {fileName && <p className="text-[11px] text-slate-500">Выбран: {fileName}</p>}
              <label className="block text-xs font-medium text-slate-700">
                Название (необязательно)
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Как подписать документ в базе"
                  className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm text-slate-900 outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                />
              </label>
              <button
                onClick={submitFile}
                disabled={busy || !fileName}
                className={`w-full rounded-xl px-3 py-2 text-sm font-semibold transition ${
                  busy || !fileName
                    ? 'cursor-not-allowed bg-slate-200 text-slate-400'
                    : 'bg-brand-600 text-white hover:bg-brand-700'
                }`}
              >
                {busy ? 'Загружаю…' : 'Загрузить в векторную базу'}
              </button>
            </div>
          )}

          {tab === 'articles' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-slate-500">
                  {data ? `фрагментов в векторной базе: ${data.vector_chunks}` : ''}
                </span>
                <button
                  onClick={() => void refresh()}
                  className="rounded-lg border border-slate-300 px-2.5 py-1 text-[11px] text-slate-600 hover:bg-slate-100"
                >
                  {loading ? 'Обновляю…' : 'Обновить'}
                </button>
              </div>
              {!loading && articles.length === 0 && (
                <p className="rounded-xl bg-slate-50 p-3 text-sm text-slate-500">
                  Пока пусто. Добавьте статью по ссылке или загрузите документ.
                </p>
              )}
              {articles.map((article) => (
                <ArticleCard
                  key={article.article_id}
                  article={article}
                  open={expanded === article.article_id}
                  onToggle={() =>
                    setExpanded((prev) => (prev === article.article_id ? null : article.article_id))
                  }
                  onDelete={() => void remove(article.article_id)}
                />
              ))}
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}

function ArticleCard({
  article,
  open,
  onToggle,
  onDelete,
}: {
  article: LibraryArticle
  open: boolean
  onToggle: () => void
  onDelete: () => void
}) {
  const status = STATUS_LABEL[article.status] ?? { text: article.status, className: 'bg-slate-100 text-slate-600' }
  const chunks: UsedChunk[] = article.used_chunks ?? []

  return (
    <div className="rounded-xl border border-slate-200">
      <div className="flex items-start justify-between gap-2 px-3 py-2">
        <button onClick={onToggle} className="min-w-0 flex-1 text-left" aria-expanded={open}>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="truncate text-sm font-medium text-slate-800">{article.title || article.article_id}</span>
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${status.className}`}>
              {status.text}
            </span>
            {article.source_kind === 'file' && (
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-600">файл</span>
            )}
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
            <span>{article.chunks} фрагм. в базе</span>
            <span>·</span>
            <span>{chunks.length ? `${chunks.length} использовано агентом` : 'агент ещё не использовал'}</span>
            {article.added_at > 0 && (
              <>
                <span>·</span>
                <span>{timeAgo(article.added_at)}</span>
              </>
            )}
            {article.url && (
              <>
                <span>·</span>
                <a
                  href={article.url}
                  target="_blank"
                  rel="noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="truncate text-brand-700 hover:underline"
                >
                  ссылка
                </a>
              </>
            )}
          </div>
        </button>
        <div className="flex shrink-0 items-center gap-1">
          <button onClick={onToggle} className="rounded-lg px-2 py-1 text-[11px] text-slate-500 hover:bg-slate-100">
            {open ? 'Свернуть' : 'Куски'}
          </button>
          <button
            onClick={onDelete}
            title="Удалить статью из базы"
            className="rounded-lg px-2 py-1 text-[11px] text-rose-600 hover:bg-rose-50"
          >
            Удалить
          </button>
        </div>
      </div>

      {open && (
        <div className="space-y-2 border-t border-slate-200 bg-slate-50/60 px-3 py-2">
          {chunks.length === 0 ? (
            <p className="text-[11px] text-slate-500">
              Агент пока не использовал фрагменты этой статьи. Они появятся здесь после ответа, в котором статья
              стала источником.
            </p>
          ) : (
            chunks.map((chunk) => (
              <div key={chunk.id} className="rounded-lg border border-slate-200 bg-white p-2">
                <div className="flex flex-wrap items-center gap-2 text-[10px] text-slate-400">
                  {chunk.section_title && <span className="font-medium text-slate-600">{chunk.section_title}</span>}
                  {chunk.section_id && <span className="font-mono">{chunk.section_id}</span>}
                  {chunk.similarity > 0 && <span>релевантность {chunk.similarity.toFixed(2)}</span>}
                  <span>· {timeAgo(chunk.used_at)}</span>
                </div>
                {chunk.query && <p className="mt-1 text-[11px] italic text-slate-500">Вопрос: {chunk.query}</p>}
                <p className="mt-1 whitespace-pre-wrap text-xs leading-relaxed text-slate-700">{chunk.text}</p>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
