import type { VersionInfo } from "../api/client";

export default function UpdateBanner({
  version,
  onUpdate,
}: {
  version: VersionInfo | null;
  onUpdate: () => void;
}) {
  if (!version) return null;
  if (version.updating) {
    return (
      <div className="banner">
        <span className="badge">
          <span
            className="dot"
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "var(--accent)",
              animation: "pulse 1.4s ease-in-out infinite",
            }}
          />
          重启中
        </span>
        <div>服务正在重启以应用更新…</div>
      </div>
    );
  }
  if (version.behind > 0) {
    return (
      <div className="banner">
        <span className="badge">可更新</span>
        <div>
          代码已落后 <strong>{version.behind}</strong> 个 commit ·{" "}
          <span className="mono" style={{ color: "var(--muted)", fontSize: 12 }}>
            {version.branch}
          </span>
          <span style={{ color: "var(--muted)" }}>
            {" — "}
            {version.auto_update ? "下次轮询会自动拉取并重启。" : "自动更新已关闭。"}
          </span>
        </div>
        <div className="spacer" />
        <button className="btn primary sm" onClick={onUpdate}>
          立即更新
        </button>
      </div>
    );
  }
  return null;
}
