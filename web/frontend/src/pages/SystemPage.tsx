import { useState } from "react";
import { api, type VersionInfo } from "../api/client";
import VersionBadge from "../components/VersionBadge";
import { Icon } from "../components/icons";
import { fmtTime } from "../utils/format";

export default function SystemPage({
  version,
  onUpdate,
  onCheckUpdate,
  onVersionChange,
}: {
  version: VersionInfo | null;
  onUpdate: () => void;
  onCheckUpdate: () => void;
  onVersionChange?: (v: Partial<VersionInfo>) => void;
}) {
  const [autoBusy, setAutoBusy] = useState(false);
  const toggleAutoUpdate = async () => {
    if (!version || autoBusy) return;
    const next = !version.auto_update;
    setAutoBusy(true);
    // Optimistic UI; revert on error.
    onVersionChange?.({ auto_update: next });
    try {
      await api.setAutoUpdate(next);
    } catch (e) {
      onVersionChange?.({ auto_update: !next });
      alert((e as Error).message);
    } finally {
      setAutoBusy(false);
    }
  };

  if (!version) return null;
  const statusText = version.updating
    ? "更新中"
    : version.behind > 0
    ? `落后 ${version.behind} 个 commit`
    : "已是最新";

  const sb = version.sandbox_backend || "none";
  const sbLabel: Record<string, string> = {
    "sandbox-exec": "macOS sandbox-exec",
    bwrap: "Linux bwrap",
    firejail: "Linux firejail",
    none: "未启用 FS 沙箱",
  };

  return (
    <main className="main">
      <div className="card">
        <div className="section-head" style={{ margin: "-4px 0 14px" }}>
          <h2>版本信息</h2>
          <button className="btn sm ghost" onClick={onCheckUpdate}>
            <Icon.Refresh /> 检查更新
          </button>
        </div>

        {version.behind > 0 && !version.updating && (
          <div className="banner" style={{ marginBottom: 16 }}>
            <span className="badge">可更新</span>
            <div>
              代码已落后 <strong>{version.behind}</strong> 个 commit。
              <span style={{ color: "var(--muted)", marginLeft: 6 }}>
                {version.auto_update ? "下次轮询会自动拉取并重启。" : "自动更新已关闭。"}
              </span>
            </div>
            <div className="spacer" />
            <button className="btn primary sm" onClick={onUpdate}>
              立即更新
            </button>
          </div>
        )}

        <div className="kv-row">
          <div className="k">当前 commit</div>
          <div className="v mono">{version.commit}</div>
        </div>
        <div className="kv-row">
          <div className="k">分支</div>
          <div className="v mono">{version.branch}</div>
        </div>
        <div className="kv-row">
          <div className="k">状态</div>
          <div className="v" style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <VersionBadge info={version} />
            <span style={{ color: "var(--muted)", fontSize: 12 }}>{statusText}</span>
          </div>
        </div>
        <div className="kv-row">
          <div className="k">远端仓库</div>
          <div className="v">
            <a
              href={version.remote_url}
              target="_blank"
              rel="noreferrer"
              style={{
                color: "var(--accent)",
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <Icon.Github /> {version.remote_url.replace(/^https?:\/\//, "")}
            </a>
          </div>
        </div>
        <div className="kv-row">
          <div className="k">上次检查</div>
          <div className="v" style={{ color: "var(--muted)" }}>
            {fmtTime(version.last_check)}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="section-head" style={{ margin: "-4px 0 6px" }}>
          <h2>更新策略</h2>
        </div>
        <div className="kv-row">
          <div className="k">自动更新</div>
          <div className="v" style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div
              className={`toggle ${version.auto_update ? "on" : ""} ${autoBusy ? "busy" : ""}`}
              role="switch"
              aria-checked={version.auto_update}
              tabIndex={0}
              onClick={toggleAutoUpdate}
              onKeyDown={(e) => {
                if (e.key === " " || e.key === "Enter") {
                  e.preventDefault();
                  toggleAutoUpdate();
                }
              }}
              style={{ cursor: autoBusy ? "wait" : "pointer" }}
            />
            <span style={{ color: "var(--muted)", fontSize: 12 }}>
              {version.auto_update
                ? "检测到新版本时自动拉取并热重启服务。"
                : "需手动触发更新。"}
            </span>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="section-head" style={{ margin: "-4px 0 6px" }}>
          <h2>安全沙箱</h2>
        </div>
        <div className="kv-row">
          <div className="k">FS 沙箱</div>
          <div className="v">
            {sbLabel[sb] || sb}
            <span style={{ marginLeft: 8, color: "var(--muted)", fontSize: 12 }}>
              {sb === "none"
                ? "资源限制和环境隔离仍在生效。"
                : version.sandbox_allow_network
                ? "允许联网（首跑下载模型需要）。"
                : "已切断网络。"}
            </span>
          </div>
        </div>
      </div>
    </main>
  );
}
