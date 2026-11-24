import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { api, type Answer, type Doc } from "./api";

type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; answer: Answer }
  | { role: "error"; text: string };

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [uploading, setUploading] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refresh = () =>
    api
      .listDocuments()
      .then(setDocs)
      .catch((e: Error) => setNotice(`Cannot reach the API: ${e.message}`));

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  async function onFiles(files: FileList | null) {
    if (!files) return;
    setNotice(null);
    for (const file of Array.from(files)) {
      setUploading(file.name);
      try {
        const doc = await api.upload(file);
        setNotice(`Added ${doc.filename}: ${doc.pages} pages, ${doc.chunks} chunks.`);
      } catch (e) {
        setNotice(`${file.name}: ${(e as Error).message}`);
      }
    }
    setUploading(null);
    refresh();
  }

  async function onDelete(doc: Doc) {
    if (!confirm(`Remove ${doc.filename}?`)) return;
    try {
      await api.remove(doc.id);
      setSelected((s) => s.filter((id) => id !== doc.id));
      refresh();
    } catch (e) {
      setNotice((e as Error).message);
    }
  }

  async function onAsk(event?: FormEvent) {
    event?.preventDefault();
    const text = question.trim();
    if (!text || asking) return;
    setQuestion("");
    setMessages((m) => [...m, { role: "user", text }]);
    setAsking(true);
    try {
      const answer = await api.ask(text, selected);
      setMessages((m) => [...m, { role: "assistant", answer }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "error", text: (e as Error).message }]);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>
          Docu<span>Mind</span>
        </h1>
        <p className="tagline">Ask your PDFs. Every answer cites its page.</p>

        <label className={`dropzone ${uploading ? "busy" : ""}`}>
          <input
            type="file"
            accept="application/pdf"
            multiple
            disabled={!!uploading}
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
          {uploading ? `Indexing ${uploading}…` : "+ Upload PDFs"}
        </label>
        {notice && <p className="notice">{notice}</p>}

        <h2>
          Documents{" "}
          <small>{selected.length ? `${selected.length} selected` : "searching all"}</small>
        </h2>
        {docs.length === 0 && <p className="muted">No documents yet.</p>}
        <ul className="docs">
          {docs.map((doc) => (
            <li key={doc.id}>
              <label>
                <input
                  type="checkbox"
                  checked={selected.includes(doc.id)}
                  onChange={(e) =>
                    setSelected((s) =>
                      e.target.checked ? [...s, doc.id] : s.filter((id) => id !== doc.id),
                    )
                  }
                />
                <span className="doc-name" title={doc.filename}>
                  {doc.filename}
                </span>
              </label>
              <span className="muted">{doc.pages}p</span>
              <button
                className="icon"
                aria-label={`Remove ${doc.filename}`}
                onClick={() => onDelete(doc)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <h2>Upload a PDF, then ask a question.</h2>
              <p className="muted">
                DocuMind finds the most relevant passages with hybrid search (vectors + keywords),
                reranks them, and answers only from those passages, with numbered citations.
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageView key={i} message={m} />
          ))}
          {asking && <div className="bubble assistant thinking">Searching your documents…</div>}
          <div ref={bottomRef} />
        </div>

        <form className="composer" onSubmit={onAsk}>
          <textarea
            value={question}
            placeholder="Ask about your documents…"
            rows={2}
            maxLength={1000}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onAsk();
              }
            }}
          />
          <button type="submit" disabled={asking || !question.trim()}>
            Ask
          </button>
        </form>
      </main>
    </div>
  );
}

function MessageView({ message }: { message: Message }) {
  const [active, setActive] = useState<number | null>(null);
  if (message.role === "user") return <div className="bubble user">{message.text}</div>;
  if (message.role === "error") return <div className="bubble error">{message.text}</div>;

  const { answer } = message;
  return (
    <div className="bubble assistant">
      <p className="answer">{withCitations(answer.answer, setActive)}</p>
      {answer.sources.length > 0 && (
        <div className="sources">
          {answer.sources.map((s) => (
            <div
              key={s.n}
              className={`source ${s.cited ? "cited" : ""} ${active === s.n ? "active" : ""}`}
              onMouseEnter={() => setActive(s.n)}
              onMouseLeave={() => setActive(null)}
            >
              <div className="source-head">
                <span className="badge">{s.n}</span>
                <strong title={s.filename}>{s.filename}</strong>
                <span className="muted">p. {s.page}</span>
              </div>
              <p>{s.snippet.length > 260 ? `${s.snippet.slice(0, 260)}…` : s.snippet}</p>
            </div>
          ))}
        </div>
      )}
      <div className="meta">
        {answer.model} · {(answer.latency_ms / 1000).toFixed(1)} s
      </div>
    </div>
  );
}

/** Turn "[1]" and "[1, 2]" into hoverable citation chips. */
function withCitations(text: string, onHover: (n: number | null) => void): ReactNode[] {
  return text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g).map((part, i) => {
    const match = part.match(/^\[(.+)\]$/);
    if (!match) return part;
    return match[1].split(",").map((n) => {
      const num = Number(n.trim());
      return (
        <sup
          key={`${i}-${num}`}
          className="cite"
          onMouseEnter={() => onHover(num)}
          onMouseLeave={() => onHover(null)}
        >
          {num}
        </sup>
      );
    });
  });
}
