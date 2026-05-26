import type { VersionInfo } from "../api/client";

export default function VersionBadge({ info }: { info: VersionInfo | null }) {
  if (!info) return null;
  let cls = "vbadge";
  let text = "已是最新";
  if (info.updating) {
    cls += " updating";
    text = "更新中";
  } else if (info.behind > 0) {
    cls += " behind";
    text = `落后 ${info.behind} 个 commit`;
  }
  return (
    <a className={cls} href={info.remote_url} target="_blank" rel="noreferrer" title={`分支：${info.branch}`}>
      <span className="dot" />
      <span className="sha">{info.short_commit}</span>
      <span style={{ color: "var(--faint)" }}>·</span>
      <span>{text}</span>
    </a>
  );
}
