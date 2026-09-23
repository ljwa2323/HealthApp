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

export type TimelineEventApi = {
  event_id: string;
  patient_id: string;
  event_type: string;
  title: string;
  event_time: { start?: string | null };
  institution?: string | null;
  summary?: string | null;
  source_artifact_ids?: string[];
  facts: Array<{
    fact_id: string;
    concept: { display: string };
    value?: { type: string; number?: number; unit?: string; text?: string } | null;
    interpretation?: "low" | "normal" | "high" | "abnormal" | "unknown" | null;
    evidence?: Array<{ quote?: string | null; artifact_id?: string | null }>;
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
) {
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
    evidence_refs?: string[];
  } | null;
};

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
      }),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as AssistantReply;
}
