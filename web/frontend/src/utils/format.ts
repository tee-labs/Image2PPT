// Formatters + label tables shared across the UI.

export function fmtETA(s: number): string {
  if (!s || s <= 0) return "—";
  if (s < 60) return `${s} 秒`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return r ? `${m} 分 ${r} 秒` : `${m} 分`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return mm ? `${h} 小时 ${mm} 分` : `${h} 小时`;
}

export function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const opts: Intl.DateTimeFormatOptions = sameDay
    ? { hour: "2-digit", minute: "2-digit", hour12: false }
    : { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false };
  return d.toLocaleString("zh-CN", opts);
}

export const STATUS_LABEL: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  done: "已完成",
  failed: "失败",
  canceled: "已取消",
};

export const MODE_LABEL: Record<string, string> = {
  full: "完整模式",
  "text-only": "仅文字",
};

export type FileKind = "pdf" | "img" | "zip";

export function fileKind(name: string): FileKind {
  const n = name.toLowerCase();
  if (n.endsWith(".pdf")) return "pdf";
  if (n.endsWith(".zip")) return "zip";
  return "img";
}

// Map backend's source_kind onto our visual kind.
export function kindFromSource(source_kind: string): FileKind {
  if (source_kind === "pdf") return "pdf";
  if (source_kind === "dir") return "zip";
  return "img";
}
