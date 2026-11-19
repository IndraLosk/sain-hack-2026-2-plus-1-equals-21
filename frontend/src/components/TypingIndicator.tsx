/** Три пульсирующие точки + подпись: видно, что запрос жив, пока нет первого токена. */
export default function TypingIndicator({ label = 'Ищу в статьях…' }: { label?: string }) {
  return (
    <span className="flex items-center gap-2 text-slate-400" aria-live="polite">
      <span className="flex items-center gap-1" aria-hidden>
        <span className="dot h-1.5 w-1.5 rounded-full bg-slate-400" />
        <span className="dot h-1.5 w-1.5 rounded-full bg-slate-400" style={{ animationDelay: '0.15s' }} />
        <span className="dot h-1.5 w-1.5 rounded-full bg-slate-400" style={{ animationDelay: '0.3s' }} />
      </span>
      <span className="text-xs">{label}</span>
    </span>
  )
}
