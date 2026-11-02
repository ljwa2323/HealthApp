const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const TOKEN_KEY = "healthapp_token";
const USER_KEY = "healthapp_user";

export type AuthUser = {
  user_id: string;
  email: string;
  display_name: string;
  patient_id: string;
  sex: "female" | "male" | "other" | "unknown";
};

export type AuthResponse = {
  token: string;
  user: AuthUser;
};

export type ExtractedFact = {
  fact_id: string;
  kind: string;
  concept: {
    display: string;
    local_display?: string | null;
    local_code?: string | null;
  };
  value:
    | { type: "quantity"; number: number; unit?: string | null }
    | { type: "text"; text: string };
  reference_range?: {
    low?: number | null;
    high?: number | null;
    unit?: string | null;
    text?: string | null;
  } | null;
  interpretation?: string | null;
  confidence?: number | null;
  status?: string;
  attributes?: Record<string, unknown>;
  time?: { start?: string | null; precision?: string };
};

export type ExtractedEvent = {
  event_id: string;
  event_type: string;
  title: string;
  event_time?: {
    start?: string | null;
    end?: string | null;
    precision?: string;
    source?: string;
  };
  institution?: string | null;
  summary?: string | null;
  narrative?: string | null;
  // Reconstructed multi-column table (row-major cells) for the table view.
  grid?: string[][] | null;
  facts: ExtractedFact[];
  patient_id?: string;
  source_artifact_ids?: string[];
};

export type UploadResult = {
  artifact: {
    artifact_id: string;
    file_name: string;
    document_type?: string | null;
    ocr_engine?: string | null;
    ocr_text?: string | null;
    ocr_blocks?: Array<{
      block_id: string;
      text: string;
      confidence?: number | null;
      bbox?: number[] | null;
    }>;
  };
  status: string;
  pipeline_steps: string[];
  extracted_event: ExtractedEvent | null;
  cross_check?: Array<{
    fact_id: string;
    concept: string;
    status: "verified" | "not_found" | "conflict";
    detail: string;
  }>;
  quality?: {
    ok: boolean;
    issues: string[];
    metrics?: Record<string, number | string>;
    doc_type?: "auto" | "text" | "imaging" | null;
  } | null;
};

export type TimelineEventApi = {
  event_id: string;
  patient_id: string;
  event_type: string;
  title: string;
  event_time: { start?: string | null };
  institution?: string | null;
  summary?: string | null;
  source_artifact_ids?: string[];
  grid?: string[][] | null;
  facts: Array<{
    fact_id: string;
    concept: { display: string };
    value?: { type: string; number?: number; unit?: string; text?: string } | null;
    interpretation?: "low" | "normal" | "high" | "abnormal" | "unknown" | null;
    evidence?: Array<{ quote?: string | null; artifact_id?: string | null }>;
    attributes?: Record<string, unknown>;
  }>;
};

function authHeaders(token?: string | null): HeadersInit {
  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") return payload.detail;
    return JSON.stringify(payload);
  } catch {
    return `Request failed: ${response.status}`;
  }
}

export function loadStoredSession(): AuthResponse | null {
  const token = localStorage.getItem(TOKEN_KEY);
  const rawUser = localStorage.getItem(USER_KEY);
  if (!token || !rawUser) return null;
  try {
    return { token, user: JSON.parse(rawUser) as AuthUser };
  } catch {
    return null;
  }
}

export function saveSession(session: AuthResponse): void {
  localStorage.setItem(TOKEN_KEY, session.token);
  localStorage.setItem(USER_KEY, JSON.stringify(session.user));
}

export function clearSession(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export async function registerUser(input: {
  email: string;
  password: string;
  display_name: string;
  sex?: AuthUser["sex"];
}): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/api/auth/register`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new Error(await readError(response));
  const data = (await response.json()) as AuthResponse;
  saveSession(data);
  return data;
}

export async function loginUser(input: {
  email: string;
  password: string;
}): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new Error(await readError(response));
  const data = (await response.json()) as AuthResponse;
  saveSession(data);
  return data;
}

export async function logoutUser(token: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/auth/logout`, {
      method: "POST",
      headers: authHeaders(token),
    });
  } finally {
    clearSession();
  }
}

export async function changePassword(
  token: string,
  currentPassword: string,
  newPassword: string,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/auth/change-password`, {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
  if (!response.ok) throw new Error(await readError(response));
}

export async function fetchMe(token: string): Promise<AuthUser> {
  const response = await fetch(`${API_BASE}/api/auth/me`, {
    headers: authHeaders(token),
  });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as AuthUser;
}

export async function uploadHealthRecord(
  file: File,
  patientId: string,
  token: string,
): Promise<UploadResult> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/documents`,
    {
      method: "POST",
      headers: authHeaders(token),
      body,
    },
  );

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response.json();
}

export type SplitResult = {
  event: ExtractedEvent;
  facts: ExtractedFact[];
  cross_check: UploadResult["cross_check"];
};

export async function splitDocument(
  patientId: string,
  artifactId: string,
  token: string,
): Promise<SplitResult> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/documents/${artifactId}/split`,
    {
      method: "POST",
      headers: authHeaders(token),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function fetchTimeline(
  patientId: string,
  token: string,
): Promise<TimelineEventApi[]> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/timeline`,
    { headers: authHeaders(token) },
  );
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as TimelineEventApi[];
}

export async function addTimelineEvent(
  patientId: string,
  token: string,
  event: ExtractedEvent,
): Promise<TimelineEventApi> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/timeline`,
    {
      method: "POST",
      headers: { ...authHeaders(token), "Content-Type": "application/json" },
      body: JSON.stringify(event),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as TimelineEventApi;
}

export async function updateTimelineEvent(
  patientId: string,
  eventId: string,
  token: string,
  payload: {
    title?: string;
    summary?: string;
    institution?: string;
    event_time_start?: string;
    facts?: Array<{
      fact_id: string;
      label?: string;
      value_text?: string;
      interpretation?: string;
      status?: "confirmed" | "corrected" | "rejected";
    }>;
  },
): Promise<TimelineEventApi> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/timeline/${eventId}`,
    {
      method: "PUT",
      headers: {
        ...authHeaders(token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as TimelineEventApi;
}

export async function deleteTimelineEvent(
  patientId: string,
  eventId: string,
  token: string,
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/timeline/${eventId}`,
    {
      method: "DELETE",
      headers: authHeaders(token),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
}

export async function fetchArtifactObjectUrl(
  patientId: string,
  artifactId: string,
  token: string,
): Promise<{ url: string; mediaType: string }> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/artifacts/${artifactId}`,
    { headers: authHeaders(token) },
  );
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  const blob = await response.blob();
  return {
    url: URL.createObjectURL(blob),
    mediaType: blob.type || "application/octet-stream",
  };
}

export type AssistantReply = {
  answer: string;
  evidence_refs: string[];
  used_fact_ids: string[];
  context?: {
    observations?: Array<{
      fact_id?: string;
      time?: string | null;
      concept?: string;
      value?: { type?: string; number?: number; unit?: string; text?: string } | null;
      interpretation?: string | null;
    }>;
    events?: Array<{ title?: string; time?: string | null; type?: string }>;
    data_gaps?: string[];
    data_quality_issues?: string[];
    evidence_refs?: string[];
  } | null;
  conversation_id?: string | null;
};

export type ConversationSummary = {
  conversation_id: string;
  title: string;
  preview: string;
  updated_at: string;
};

export type ConversationDetail = {
  conversation_id: string;
  title: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
};

export async function listConversations(
  patientId: string,
  token: string,
  search = "",
): Promise<ConversationSummary[]> {
  const url = new URL(`${API_BASE}/api/patients/${patientId}/conversations`);
  if (search) url.searchParams.set("q", search);
  const res = await fetch(url.toString(), { headers: authHeaders(token) });
  if (!res.ok) throw new Error(await readError(res));
  const data = await res.json();
  return data.conversations ?? [];
}

export async function getConversation(
  patientId: string,
  token: string,
  conversationId: string,
): Promise<ConversationDetail> {
  const res = await fetch(
    `${API_BASE}/api/patients/${patientId}/conversations/${conversationId}`,
    { headers: authHeaders(token) },
  );
  if (!res.ok) throw new Error(await readError(res));
  return await res.json();
}

export async function deleteConversation(
  patientId: string,
  token: string,
  conversationId: string,
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/patients/${patientId}/conversations/${conversationId}`,
    { method: "DELETE", headers: authHeaders(token) },
  );
  if (!res.ok) throw new Error(await readError(res));
}

export async function runHealthAnalysis(
  patientId: string,
  token: string,
): Promise<AssistantReply> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/assistant/analyze`,
    {
      method: "POST",
      headers: {
        ...authHeaders(token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ task: "general_longitudinal_review" }),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as AssistantReply;
}

export async function askHealthAssistant(
  patientId: string,
  token: string,
  question: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
  conversationId?: string | null,
): Promise<AssistantReply> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/assistant/chat`,
    {
      method: "POST",
      headers: {
        ...authHeaders(token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        history,
        task: "general_longitudinal_review",
        conversation_id: conversationId ?? null,
      }),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as AssistantReply;
}

export type StreamEvent = {
  type: string;
  text?: string;
  answer?: string;
  confidence?: number;
  conversation_id?: string | null;
  context?: AssistantReply["context"] | null;
  used_fact_ids?: string[];
  detail?: string;
};

export type StreamResult = {
  answer: string;
  conversation_id: string | null;
  context: AssistantReply["context"] | null;
  used_fact_ids: string[];
  confidence: number | null;
};

export async function askHealthAssistantStream(
  patientId: string,
  token: string,
  question: string,
  history: Array<{ role: "user" | "assistant"; content: string }>,
  conversationId?: string | null,
  onChunk?: (text: string) => void,
): Promise<StreamResult> {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/assistant/chat/stream`,
    {
      method: "POST",
      headers: {
        ...authHeaders(token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        history,
        task: "general_longitudinal_review",
        conversation_id: conversationId ?? null,
      }),
    },
  );
  if (!response.ok || !response.body) throw new Error(await readError(response));

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let conversationIdResult: string | null = null;
  let context: AssistantReply["context"] | null = null;
  let usedFactIds: string[] = [];
  let confidence: number | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      const jsonText = line.slice(5).trim();
      if (!jsonText) continue;
      let event: StreamEvent;
      try {
        event = JSON.parse(jsonText) as StreamEvent;
      } catch {
        continue;
      }
      if (event.type === "meta") {
        conversationIdResult = event.conversation_id ?? conversationIdResult;
        context = event.context ?? null;
        usedFactIds = event.used_fact_ids ?? [];
      } else if (event.type === "chunk") {
        answer += event.text ?? "";
        onChunk?.(event.text ?? "");
      } else if (event.type === "done") {
        answer = event.answer ?? answer;
        confidence = event.confidence ?? null;
        conversationIdResult = event.conversation_id ?? conversationIdResult;
      } else if (event.type === "error") {
        throw new Error(event.detail ?? "流式回答失败");
      }
    }
  }

  return {
    answer,
    conversation_id: conversationIdResult,
    context,
    used_fact_ids: usedFactIds,
    confidence,
  };
}