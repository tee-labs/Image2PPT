import { useState } from "react";
import { api, auth, type Me } from "../api/client";
import { Icon } from "../components/icons";
import TopBar from "../components/TopBar";
import { t, useLocale } from "../i18n";

const REPO_URL = "https://github.com/tee-labs/Image2PPT";

export default function Login({ onLogin }: { onLogin: (me: Me) => void }) {
  useLocale();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await api.login(username, password);
      auth.setToken(res.access_token);
      onLogin({ id: 0, username: res.username, is_admin: res.is_admin });
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-wrap">
      <div className="login-topbar">
        <TopBar />
      </div>

      <div className="login-card-wrap">
        <form className="login-card" onSubmit={submit}>
          <div className="login-brand-row">
            <h1 className="login-wordmark">DeckWeaver</h1>
            <div className="login-sub">{t("brand.tagline")}</div>
          </div>

          <label className="field">
            <span>{t("login.username")}</span>
            <input
              className="input"
              autoFocus
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
          <label className="field">
            <span>{t("login.password")}</span>
            <input
              className="input"
              type="password"
              autoComplete="current-password"
              value={password}
              placeholder="••••••••"
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          {err && <div className="err">{err}</div>}

          <button
            className="btn primary login-btn"
            disabled={busy || !password}
          >
            {busy ? t("login.submitting") : t("login.submit")}
          </button>

          <a className="login-repo" href={REPO_URL} target="_blank" rel="noreferrer">
            <Icon.Github />
            <div className="info">
              <div className="login-repo-name">tee-labs/Image2PPT</div>
              <div className="login-repo-meta">{t("login.see_source")}</div>
            </div>
            <span className="login-repo-arrow">↗</span>
          </a>

          <div className="foot">
            <span>DeckWeaver · web</span>
            <span>v0.1.0</span>
          </div>
        </form>
      </div>
    </div>
  );
}
