import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { fileKind, fmtSize } from "../utils/format";
import { Icon } from "./icons";

type Mode = "full" | "text-only";

export default function UploadCard({ onCreated }: { onCreated: () => void }) {
  const [mode, setMode] = useState<Mode>("full");
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-derive a default name from the first file when files change,
  // unless the user has typed their own.
  useEffect(() => {
    if (nameTouched) return;
    if (files.length === 0) {
      setName("");
      return;
    }
    const base = files[0].name.replace(/\.[^.]+$/, "");
    setName(files.length > 1 ? `${base} 等 ${files.length} 个文件` : base);
  }, [files, nameTouched]);

  const previews = useMemo(() => {
    const out: Record<number, string> = {};
    files.forEach((f, i) => {
      if (f.type && f.type.startsWith("image/")) {
        out[i] = URL.createObjectURL(f);
      }
    });
    return out;
  }, [files]);
  useEffect(() => () => Object.values(previews).forEach((u) => URL.revokeObjectURL(u)), [previews]);

  const pickFiles = (list: FileList | null) => {
    if (!list) return;
    setFiles(Array.from(list));
  };
  const removeAt = (i: number) => setFiles((fs) => fs.filter((_, j) => j !== i));

  const submit = async () => {
    if (!files.length) return;
    setBusy(true);
    setError(null);
    try {
      await api.createJob(mode, files, name);
      setFiles([]);
      setName("");
      setNameTouched(false);
      if (inputRef.current) inputRef.current.value = "";
      onCreated();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div className="section-head" style={{ margin: "-4px 0 14px" }}>
        <h2>
          <Icon.Sparkle style={{ verticalAlign: -3, marginRight: 6, color: "var(--accent)" }} />
          新建任务
        </h2>
        <span className="meta">支持 PDF · 图片 · ZIP 压缩包</span>
      </div>

      <label
        className="field"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 6,
          marginBottom: 14,
          fontSize: 12,
          color: "var(--text-2)",
          fontWeight: 500,
        }}
      >
        任务名称
        <input
          className="input"
          value={name}
          placeholder={files.length ? "" : "上传文件后将自动生成名称，也可手动输入"}
          onChange={(e) => {
            setName(e.target.value);
            setNameTouched(true);
          }}
        />
        <span style={{ fontSize: 11, color: "var(--muted)", fontWeight: 400 }}>
          {nameTouched ? "已手动命名" : files.length ? "默认根据文件生成，可修改" : "若留空将使用文件名"}
        </span>
      </label>

      <div className="upload">
        <div
          className={`dropzone ${dragging ? "dragging" : ""}`}
          onClick={() => files.length === 0 && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            pickFiles(e.dataTransfer.files);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.pdf,.zip"
            onChange={(e) => pickFiles(e.target.files)}
          />
          {files.length === 0 ? (
            <>
              <div className="dz-title">
                <Icon.Upload />
                拖拽文件到此处{" "}
                <span style={{ color: "var(--muted)", fontWeight: 400 }}>或点击选择</span>
              </div>
              <div className="dz-hint">单个 PDF、单张/多张图片，或图片 ZIP 压缩包。</div>
              <div className="dz-ext">
                <span>.pdf</span>
                <span>.png</span>
                <span>.jpg</span>
                <span>.webp</span>
                <span>.tif</span>
                <span>.zip</span>
              </div>
            </>
          ) : (
            <>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 12,
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 500 }}>
                  已选中 <strong style={{ color: "var(--accent)" }}>{files.length}</strong> 个文件
                </div>
                <button
                  className="btn sm ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    inputRef.current?.click();
                  }}
                >
                  添加更多
                </button>
              </div>
              <div className="dz-files">
                {files.map((f, i) => (
                  <div key={i} className="dz-file" onClick={(e) => e.stopPropagation()}>
                    <div className="thumb">
                      {previews[i] ? (
                        <img src={previews[i]} alt="" />
                      ) : (
                        <span>{fileKind(f.name).toUpperCase()}</span>
                      )}
                    </div>
                    <div>
                      <div className="name">{f.name}</div>
                      <div className="size">{fmtSize(f.size)}</div>
                    </div>
                    <button
                      className="btn icon ghost"
                      style={{ width: 22, height: 22, padding: 2 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        removeAt(i);
                      }}
                      title="移除"
                    >
                      <Icon.X />
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="upload-side">
          <div className="mode-group">
            <label className={`opt ${mode === "full" ? "active" : ""}`}>
              <input
                type="radio"
                name="mode"
                hidden
                checked={mode === "full"}
                onChange={() => setMode("full")}
              />
              <span className="radio" />
              <div>
                <div className="label">完整模式</div>
                <div className="desc">文字 + 图标分别识别并可编辑。耗时较长。</div>
              </div>
            </label>
            <label className={`opt ${mode === "text-only" ? "active" : ""}`}>
              <input
                type="radio"
                name="mode"
                hidden
                checked={mode === "text-only"}
                onChange={() => setMode("text-only")}
              />
              <span className="radio" />
              <div>
                <div className="label">仅文字模式</div>
                <div className="desc">整图作背景，仅文字可编辑。速度更快。</div>
              </div>
            </label>
          </div>

          {error && <div style={{ color: "var(--bad)", fontSize: 12 }}>{error}</div>}

          <div style={{ display: "flex", gap: 8, marginTop: "auto" }}>
            <button
              className="btn primary"
              style={{ flex: 1, justifyContent: "center", padding: "10px 14px" }}
              disabled={busy || !files.length}
              onClick={submit}
            >
              {busy ? "上传中…" : files.length ? `开始转换 (${files.length})` : "开始转换"}
            </button>
            {files.length > 0 && (
              <button
                className="btn"
                onClick={() => setFiles([])}
                disabled={busy}
                title="清空"
              >
                <Icon.X />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
