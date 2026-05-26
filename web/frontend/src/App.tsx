import { useEffect, useState } from "react";
import { api, auth, type Me } from "./api/client";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";

export default function App() {
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!auth.token) {
      setLoading(false);
      return;
    }
    api
      .me()
      .then(setMe)
      .catch(() => setMe(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ padding: 24, color: "var(--muted)" }}>加载中…</div>
    );
  }

  if (!me) {
    return <Login onLogin={(user) => setMe(user)} />;
  }

  return (
    <Dashboard
      me={me}
      onLogout={() => {
        auth.clear();
        setMe(null);
      }}
    />
  );
}
