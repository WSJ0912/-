export function StatPill({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "neutral" | "good" | "warn" }) {
  return <span className={`stat-pill ${tone}`}><span>{label}</span><strong>{value}</strong></span>;
}
