import type { LucideIcon } from "lucide-react";

export function IconButton({ icon: Icon, label, onClick, disabled = false }: { icon: LucideIcon; label: string; onClick?: () => void; disabled?: boolean }) {
  return <button className="icon-button" aria-label={label} title={label} onClick={onClick} disabled={disabled}><Icon size={17} strokeWidth={1.8} /></button>;
}
