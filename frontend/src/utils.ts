/** Утилиты: id, localStorage, форматирование, markdown-lite. */

export function uid(prefix = 's'): string {
  const rnd = crypto.getRandomValues(new Uint32Array(2))
  return `${prefix}_${Date.now().toString(36)}${rnd[0].toString(36)}${rnd[1].toString(36)}`
}

export function loadStored<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

export function saveStored(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* приватный режим — игнорируем */
  }
}

export function formatDate(iso?: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', year: 'numeric' })
}

export function initials(name: string): string {
  return name
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .join('')
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

/**
 * Минимальный markdown: **жирный**, *курсив*, `код`, списки, абзацы.
 * Сначала экранируем HTML — вставка через dangerouslySetInnerHTML безопасна.
 */
export function renderMarkdown(text: string): string {
  const withInline = escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code class="rounded bg-slate-100 px-1 py-0.5 text-[0.85em]">$1</code>')
    // Сноски на источники: [1], [2] → номер-ссылка на список источников.
    .replace(/\[(\d{1,2})\]/g, '<sup class="ml-0.5 font-semibold text-brand-700">[$1]</sup>')

  const out: string[] = []
  let inList = false
  for (const line of withInline.split('\n')) {
    const bullet = /^\s*[-•]\s+(.*)$/.exec(line)
    if (bullet) {
      if (!inList) {
        out.push('<ul class="my-1 list-disc space-y-1 pl-5">')
        inList = true
      }
      out.push(`<li>${bullet[1]}</li>`)
      continue
    }
    if (inList) {
      out.push('</ul>')
      inList = false
    }
    out.push(line.trim() === '' ? '<div class="h-2"></div>' : `<p>${line}</p>`)
  }
  if (inList) out.push('</ul>')
  return out.join('')
}

/**
 * Проверка чисел в ответе (без LLM, см. ideas/09 п. 12).
 * Достаём из ответа числа с единицами измерения и проверяем, что они есть
 * в сниппетах источников. Возвращает числа, которых в источниках нет.
 */
const NUM_RE = /(\d+(?:[.,]\d+)?)\s*(мг|mg|г|g|мл|ml|час\w*|ч|hour\w*|h|мин\w*|min\w*|%|°c|градус\w*)/gi

function normalizeNumber(raw: string): string {
  return raw.replace(',', '.').replace(/^0+(?=\d)/, '')
}

export function findUnverifiedNumbers(answer: string, snippets: string[]): string[] {
  if (!answer || !snippets.length) return []
  const haystack = snippets.join(' ').toLowerCase()
  const missing: string[] = []
  const seen = new Set<string>()
  for (const match of answer.matchAll(NUM_RE)) {
    const value = normalizeNumber(match[1])
    const unit = match[2].toLowerCase()
    const key = `${value}${unit}`
    if (seen.has(key)) continue
    seen.add(key)
    // число считаем подтверждённым, если оно встречается в источниках
    // (в любом написании: с точкой/запятой, с юнитом или без)
    const variants = [value, value.replace('.', ','), match[1]]
    if (!variants.some((v) => haystack.includes(v.toLowerCase()))) {
      missing.push(`${match[1]} ${match[2]}`)
    }
  }
  return missing
}
