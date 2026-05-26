// WebSocket helper — auto-reconnects on close.
import { auth } from "./client";

export type WSMessage =
  | {
      type: "job";
      id: string;
      status: string;
      progress_pct: number;
      current_page: number;
      page_count: number;
      owner_id: number;
    }
  | { type: "log"; id: string; line: string; owner_id: number }
  | {
      type: "system";
      commit: string;
      short_commit: string;
      behind: number;
      ahead: number;
      updating: boolean;
    };

export function connectWS(onMessage: (msg: WSMessage) => void): () => void {
  let closed = false;
  let ws: WebSocket | null = null;
  let timer: number | null = null;

  const open = () => {
    const token = auth.token;
    if (!token) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/jobs?token=${encodeURIComponent(token)}`);
    ws.onmessage = (e) => {
      try {
        onMessage(JSON.parse(e.data));
      } catch {}
    };
    ws.onclose = () => {
      if (closed) return;
      timer = window.setTimeout(open, 2000);
    };
    ws.onerror = () => ws?.close();
  };

  open();
  return () => {
    closed = true;
    if (timer) window.clearTimeout(timer);
    ws?.close();
  };
}
