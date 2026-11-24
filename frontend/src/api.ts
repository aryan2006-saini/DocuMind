export type Doc = {
  id: string;
  filename: string;
  pages: number;
  chunks: number;
  created_at: string | null;
};

export type Source = {
  n: number;
  document_id: string;
  filename: string;
  page: number;
  snippet: string;
  score: number;
  cited: boolean;
};

export type Answer = {
  answer: string;
  citations: number[];
  sources: Source[];
  model: string;
  latency_ms: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail))
        message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* keep the status text */
    }
    throw new Error(message);
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
}

export const api = {
  listDocuments: () => request<Doc[]>("/api/documents"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Doc>("/api/documents", { method: "POST", body: form });
  },
  remove: (id: string) => request<void>(`/api/documents/${id}`, { method: "DELETE" }),
  ask: (question: string, documentIds: string[]) =>
    request<Answer>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        document_ids: documentIds.length ? documentIds : null,
      }),
    }),
};
