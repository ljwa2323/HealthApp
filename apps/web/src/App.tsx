import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import {
  AuthResponse,
  AuthUser,
  askHealthAssistant,
  AssistantReply,
  changePassword,
  clearSession,
  deleteTimelineEvent,
  ExtractedEvent,
  fetchArtifactObjectUrl,
  fetchTimeline,
  loadStoredSession,
  loginUser,
  logoutUser,
  registerUser,
  runHealthAnalysis,
  TimelineEventApi,
  addTimelineEvent,
  updateTimelineEvent,
  UploadResult,
  uploadHealthRecord,
} from "./api";
import type { TimelineEvent } from "./types";

type Tab = "overview" | "timeline" | "upload" | "assistant";
type Filter = "all" | TimelineEvent["type"];
type AuthMode = "login" | "register";
type UploadStatus = "idle" | "processing" | "complete" | "offline";

type UploadState = {
  fileName: string | null;
  status: UploadStatus;
  message: string | null;
  startedAt: number | null;
  result: UploadResult | null;
  qualityWarning: boolean;
};

type ChatTurn = { role: "user" | "assistant"; content: string };

type AssistantState = {
  messages: ChatTurn[];
  question: string;
  busy: boolean;
  error: string | null;
  contextPreview: AssistantReply["context"] | null;
};

const emptyUploadState: UploadState = {
  fileName: null,
  status: "idle",
  message: null,
  startedAt: null,
  result: null,
  qualityWarning: false,
};

const emptyAssistantState: AssistantState = {
  messages: [],
  question: "",
  busy: false,
  error: null,
  contextPreview: null,
};

const eventTypeLabel: Record<string, string> = {
  laboratory: "检验",
  imaging: "影像",
  visit: "门诊",
  medication: "用药",
  procedure: "操作",
  vital: "生命体征",
  hospitalization: "住院",
  symptom: "症状",
  other: "其他",
};

function mapApiEvent(event: TimelineEventApi): TimelineEvent {
  const start = event.event_time?.start ?? "";
  const displayDate = String(start).slice(0, 10);
  const mappedType = (
    ["laboratory", "imaging", "visit", "medication", "procedure", "vital"].includes(
      event.event_type,
    )
      ? event.event_type
      : "visit"
  ) as TimelineEvent["type"];

  return {
    id: event.event_id,
    date: displayDate,
    type: mappedType,
    title: event.title,
    institution: event.institution ?? "",
    displayDate,
    summary: event.summary ?? "",
    source: event.facts?.[0]?.evidence?.[0]?.quote ?? "Uploaded record",
    artifactId:
      event.source_artifact_ids?.[0] ||
      event.facts?.[0]?.evidence?.[0]?.artifact_id ||
      undefined,
    facts: (event.facts ?? []).map((fact) => {
      const value = fact.value;
      let display = "-";
      if (value?.type === "quantity" && value.number != null) {
        display = `${value.number}${value.unit ? ` ${value.unit}` : ""}`;
      } else if (value?.type === "text" && value.text) {
        display = value.text;
      }
      return {
        id: fact.fact_id,
        label: fact.concept.display,
        value: display,
        interpretation:
          fact.interpretation === "low" ||
          fact.interpretation === "normal" ||
          fact.interpretation === "high" ||
          fact.interpretation === "abnormal"
            ? fact.interpretation
            : undefined,
        evidence: fact.evidence?.[0]?.quote ?? "",
      };
    }),
  };
}

function StatusPill({
  value,
}: {
  value?: "normal" | "high" | "low" | "abnormal";
}) {
  if (!value) return null;
  const label =
    value === "normal"
      ? "范围内"
      : value === "high"
        ? "偏高"
        : value === "low"
          ? "偏低"
          : "需关注";
  return <span className={`status-pill status-${value}`}>{label}</span>;
}

function ReportPreview({
  patientId,
  artifactId,
  token,
  quote,
}: {
  patientId: string;
  artifactId?: string;
  token: string;
  quote: string;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState("application/octet-stream");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [lightbox, setLightbox] = useState(false);

  useEffect(() => {
    if (!artifactId) {
      setUrl(null);
      setError("这条记录没有保存原报告图片。请重新上传后再查看。");
      return;
    }

    let active = true;
    let objectUrl: string | null = null;
    setLoading(true);
    setError(null);

    void fetchArtifactObjectUrl(patientId, artifactId, token)
      .then((result) => {
        if (!active) {
          URL.revokeObjectURL(result.url);
          return;
        }
        objectUrl = result.url;
        setUrl(result.url);
        setMediaType(result.mediaType);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "无法加载原报告");
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [patientId, artifactId, token]);

  const isImage = mediaType.startsWith("image/");

  return (
    <div className="source-box report-preview-box">
      <div className="report-preview-copy">
        <span>原始报告</span>
        <strong>{quote || "已保存的上传原件"}</strong>
      </div>

      {loading && <p className="empty-copy">正在加载原报告图片...</p>}
      {error && <p className="auth-error">{error}</p>}

      {!loading && !error && url && isImage && (
          <button
          type="button"
          className="report-thumb-button"
          onClick={() => setLightbox(true)}
          title="点击查看大图"
        >
          <span className="report-thumb-frame">
            <img src={url} alt="Original report" className="report-thumb" />
          </span>
          <span>点击查看大图</span>
        </button>
      )}

      {!loading && !error && url && !isImage && (
        <a className="secondary-button" href={url} download>
          下载原文件
        </a>
      )}

      {lightbox && url && (
        <div
          className="report-lightbox"
          role="dialog"
          aria-modal="true"
          onClick={() => setLightbox(false)}
        >
          <button
            type="button"
            className="report-lightbox-close"
            onClick={() => setLightbox(false)}
          >
            关闭
          </button>
          <img
            src={url}
            alt="Original report full size"
            className="report-lightbox-image"
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}

function EventCard({
  event,
  expanded,
  onToggle,
  patientId,
  token,
  onDelete,
  onSaveEdit,
}: {
  event: TimelineEvent;
  expanded: boolean;
  onToggle: () => void;
  patientId: string;
  token: string;
  onDelete: (event: TimelineEvent) => void;
  onSaveEdit: (
    event: TimelineEvent,
    draft: {
      title: string;
      summary: string;
      institution: string;
      eventDate: string;
      facts: Array<{
        fact_id: string;
        label: string;
        value_text: string;
        interpretation: string;
      }>;
    },
  ) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [title, setTitle] = useState(event.title);
  const [summary, setSummary] = useState(event.summary);
  const [institution, setInstitution] = useState(event.institution);
  const [eventDate, setEventDate] = useState(event.displayDate);
  const [factDrafts, setFactDrafts] = useState(
    event.facts.map((fact) => ({
      fact_id: fact.id,
      label: fact.label,
      value_text: fact.value,
      interpretation: fact.interpretation ?? "unknown",
    })),
  );

  function startEdit() {
    setTitle(event.title);
    setSummary(event.summary);
    setInstitution(event.institution);
    setEventDate(event.displayDate);
    setFactDrafts(
      event.facts.map((fact) => ({
        fact_id: fact.id,
        label: fact.label,
        value_text: fact.value,
        interpretation: fact.interpretation ?? "unknown",
      })),
    );
    setEditing(true);
  }

  return (
    <article className="event-card">
      <button className="event-head" onClick={onToggle}>
        <span className={`event-icon event-${event.type}`}>
          {(eventTypeLabel[event.type] ?? "其").slice(0, 1)}
        </span>
        <span className="event-copy">
          <span className="event-meta">
            {event.displayDate
              ? `${event.displayDate} · ${eventTypeLabel[event.type] ?? event.type}`
              : `时间未知 · ${eventTypeLabel[event.type] ?? event.type}`}
          </span>
          <strong>{event.title}</strong>
          <span>{event.institution}</span>
        </span>
        <span className="event-toggle">{expanded ? "收起" : "查看"}</span>
      </button>

      <p className="event-summary">{event.summary}</p>

      {expanded && (
        <div className="event-detail">
          {!editing ? (
            <>
              <div className="fact-list">
                {event.facts.map((fact) => (
                  <div className="fact-row" key={fact.id}>
                    <div>
                      <span className="fact-label">{fact.label}</span>
                      <strong>{fact.value}</strong>
                    </div>
                    <StatusPill value={fact.interpretation} />
                    <p className="evidence-copy">{fact.evidence}</p>
                  </div>
                ))}
              </div>
              <ReportPreview
                patientId={patientId}
                artifactId={event.artifactId}
                token={token}
                quote={event.source}
              />
              <div className="event-actions">
                <button type="button" className="secondary-button" onClick={startEdit}>
                  修正提取结果
                </button>
                <button
                  type="button"
                  className="danger-button"
                  onClick={() => onDelete(event)}
                >
                  删除记录
                </button>
              </div>
            </>
          ) : (
            <form
              className="event-edit-form"
              onSubmit={(submitEvent) => {
                submitEvent.preventDefault();
                setSaving(true);
                void onSaveEdit(event, {
                  title,
                  summary,
                  institution,
                  eventDate,
                  facts: factDrafts,
                })
                  .then(() => setEditing(false))
                  .finally(() => setSaving(false));
              }}
            >
              <label>
                <span>标题</span>
                <input value={title} onChange={(e) => setTitle(e.target.value)} required />
              </label>
              <label>
                <span>机构</span>
                <input
                  value={institution}
                  onChange={(e) => setInstitution(e.target.value)}
                />
              </label>
              <label>
                <span>事件日期</span>
                <input
                  type="date"
                  value={eventDate}
                  onChange={(e) => setEventDate(e.target.value)}
                />
              </label>
              <label>
                <span>摘要</span>
                <textarea
                  value={summary}
                  onChange={(e) => setSummary(e.target.value)}
                  rows={3}
                />
              </label>

              <div className="edit-facts">
                <strong>结构化事实</strong>
                {factDrafts.map((fact, index) => (
                  <div className="edit-fact-row" key={fact.fact_id}>
                    <input
                      value={fact.label}
                      onChange={(e) => {
                        const next = [...factDrafts];
                        next[index] = { ...fact, label: e.target.value };
                        setFactDrafts(next);
                      }}
                      placeholder="指标名"
                    />
                    <input
                      value={fact.value_text}
                      onChange={(e) => {
                        const next = [...factDrafts];
                        next[index] = { ...fact, value_text: e.target.value };
                        setFactDrafts(next);
                      }}
                      placeholder="数值，如 88 umol/L"
                    />
                    <select
                      value={fact.interpretation}
                      onChange={(e) => {
                        const next = [...factDrafts];
                        next[index] = {
                          ...fact,
                          interpretation: e.target.value,
                        };
                        setFactDrafts(next);
                      }}
                    >
                      <option value="unknown">未知</option>
                      <option value="normal">正常</option>
                      <option value="high">偏高</option>
                      <option value="low">偏低</option>
                      <option value="abnormal">异常</option>
                    </select>
                  </div>
                ))}
              </div>

              <div className="event-actions">
                <button className="primary-button" type="submit" disabled={saving}>
                  {saving ? "保存中..." : "保存修正"}
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => setEditing(false)}
                  disabled={saving}
                >
                  取消
                </button>
              </div>
            </form>
          )}
        </div>
      )}
    </article>
  );
}

function ProfileMenu({
  session,
  onLogout,
}: {
  session: AuthResponse;
  onLogout: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      const target = event.target as HTMLElement | null;
      if (target?.closest(".profile-menu")) return;
      setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open]);

  async function handleChangePassword(event: FormEvent) {
    event.preventDefault();
    if (newPassword.length < 6) {
      setError("新密码至少 6 位");
      return;
    }
    if (newPassword !== confirmPassword) {
      setError("两次输入的新密码不一致");
      return;
    }
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await changePassword(session.token, currentPassword, newPassword);
      setMessage("密码已更新");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "修改失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="profile-menu">
      <button
        className="profile-button"
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span>{session.user.display_name}</span>
        <span className="avatar">
          {session.user.display_name.slice(0, 1).toUpperCase()}
        </span>
      </button>

      {open && (
        <div className="profile-dropdown" role="menu">
          <div className="profile-dropdown-meta">
            <strong>{session.user.display_name}</strong>
            <span>{session.user.email}</span>
          </div>
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              setShowPassword(true);
              setError(null);
              setMessage(null);
            }}
          >
            修改密码
          </button>
          <button
            type="button"
            role="menuitem"
            className="profile-logout"
            onClick={() => {
              setOpen(false);
              onLogout();
            }}
          >
            退出登录
          </button>
        </div>
      )}

      {showPassword && (
        <div
          className="profile-modal-backdrop"
          onClick={() => setShowPassword(false)}
        >
          <form
            className="profile-modal"
            onClick={(event) => event.stopPropagation()}
            onSubmit={(event) => void handleChangePassword(event)}
          >
            <div className="panel-head compact">
              <div>
                <span className="eyebrow">账户安全</span>
                <h2>修改密码</h2>
              </div>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setShowPassword(false)}
              >
                关闭
              </button>
            </div>
            <label>
              当前密码
              <input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
                autoComplete="current-password"
              />
            </label>
            <label>
              新密码
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                minLength={6}
                autoComplete="new-password"
              />
            </label>
            <label>
              确认新密码
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                minLength={6}
                autoComplete="new-password"
              />
            </label>
            {error && <p className="auth-error">{error}</p>}
            {message && <p className="empty-copy">{message}</p>}
            <button className="primary-button" type="submit" disabled={busy}>
              {busy ? "保存中..." : "保存新密码"}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

function AuthScreen({
  onAuthenticated,
}: {
  onAuthenticated: (session: AuthResponse) => void;
}) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [sex, setSex] = useState<AuthUser["sex"]>("unknown");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const session =
        mode === "register"
          ? await registerUser({
              email,
              password,
              display_name: displayName,
              sex,
            })
          : await loginUser({ email, password });
      onAuthenticated(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-panel">
        <div className="auth-brand">
          <span className="brand-mark">H</span>
          <div>
            <strong>HealthApp</strong>
            <p>注册后建立你的个人健康档案</p>
          </div>
        </div>

        <div className="auth-tabs">
          <button
            type="button"
            className={mode === "register" ? "active" : ""}
            onClick={() => setMode("register")}
          >
            注册
          </button>
          <button
            type="button"
            className={mode === "login" ? "active" : ""}
            onClick={() => setMode("login")}
          >
            登录
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === "register" && (
            <>
              <label>
                <span>姓名 / 昵称</span>
                <input
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="例如：张三"
                  required
                />
              </label>
              <label>
                <span>性别（可选）</span>
                <select
                  value={sex}
                  onChange={(e) => setSex(e.target.value as AuthUser["sex"])}
                >
                  <option value="unknown">暂不填写</option>
                  <option value="female">女</option>
                  <option value="male">男</option>
                  <option value="other">其他</option>
                </select>
              </label>
            </>
          )}
          <label>
            <span>邮箱</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
            />
          </label>
          <label>
            <span>密码</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="至少 6 位"
              minLength={6}
              required
            />
          </label>

          {error && <p className="auth-error">{error}</p>}

          <button className="primary-button" type="submit" disabled={busy}>
            {busy ? "处理中..." : mode === "register" ? "创建档案" : "进入 HealthApp"}
          </button>
        </form>
      </div>
    </div>
  );
}

function Overview({
  onOpenTimeline,
  user,
  eventCount,
}: {
  onOpenTimeline: () => void;
  user: AuthUser;
  eventCount: number;
}) {
  const empty = eventCount === 0;
  return (
    <section className="page-stack">
      <div className="hero-card">
        <div>
          <span className="eyebrow">纵向健康概览</span>
          <h1>{user.display_name} 的健康记录</h1>
          <p>
            {empty
              ? "档案已创建。先上传检查报告，系统会按真实临床时间整理进时间轴。"
              : `当前档案已整理 ${eventCount} 条事件。系统优先展示变化趋势，而不是孤立的一次检查。`}
          </p>
        </div>
        <button className="primary-button" onClick={onOpenTimeline}>
          {empty ? "去添加记录" : "查看完整时间轴"}
        </button>
      </div>

      <div className="metric-grid">
        <article className="metric-card">
          <span>已归档记录</span>
          <strong>{eventCount}</strong>
          <small>{empty ? "等待首份报告" : "属于当前登录用户"}</small>
        </article>
        <article className="metric-card">
          <span>账号</span>
          <strong style={{ fontSize: "1.2rem" }}>{user.email}</strong>
          <small>patient_id: {user.patient_id}</small>
        </article>
        <article className="metric-card attention-card">
          <span>下一步</span>
          <strong style={{ fontSize: "1.35rem" }}>
            {empty ? "上传报告" : "持续补充"}
          </strong>
          <small>只展示你本人上传的数据</small>
        </article>
      </div>

      {empty && (
        <article className="panel">
          <h2>还没有可展示的健康事实</h2>
          <p>概览、时间轴和 AI 助手都基于你的真实记录，不会预填示例病历。</p>
        </article>
      )}
    </section>
  );
}

function Timeline({
  events,
  loading,
  onAdd,
  patientId,
  token,
  onDelete,
  onSaveEdit,
}: {
  events: TimelineEvent[];
  loading: boolean;
  onAdd: () => void;
  patientId: string;
  token: string;
  onDelete: (event: TimelineEvent) => void;
  onSaveEdit: (
    event: TimelineEvent,
    draft: {
      title: string;
      summary: string;
      institution: string;
      eventDate: string;
      facts: Array<{
        fact_id: string;
        label: string;
        value_text: string;
        interpretation: string;
      }>;
    },
  ) => Promise<void>;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState("");

  const filtered = useMemo(
    () => events.filter((event) => filter === "all" || event.type === filter),
    [events, filter],
  );

  const filters: Array<{ key: Filter; label: string }> = [
    { key: "all", label: "全部" },
    { key: "laboratory", label: "检验" },
    { key: "imaging", label: "影像" },
    { key: "medication", label: "用药" },
  ];

  return (
    <section className="page-stack">
      <div className="section-title">
        <div>
          <span className="eyebrow">Longitudinal timeline</span>
          <h1>健康时间轴</h1>
          <p>按事件实际发生时间排序，而不是按上传时间排序。</p>
        </div>
        <div className="title-actions">
          <button className="primary-button" type="button" onClick={onAdd}>
            添加记录
          </button>
        </div>
      </div>

      <div className="filter-row">
        {filters.map((item) => (
          <button
            key={item.key}
            className={filter === item.key ? "filter active" : "filter"}
            onClick={() => setFilter(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {loading ? (
        <p className="empty-copy">正在加载时间轴...</p>
      ) : filtered.length === 0 ? (
        <article className="panel">
          <h2>还没有记录</h2>
          <p>上传报告后会出现在这里。</p>
        </article>
      ) : (
        <div className="timeline-shell">
          <aside className="year-rail">
            <strong>时间</strong>
            <span />
            <strong>轴</strong>
          </aside>
          <div className="timeline-list">
            {filtered.map((event) => (
              <EventCard
                key={event.id}
                event={event}
                expanded={expanded === event.id}
                onToggle={() =>
                  setExpanded((current) =>
                    current === event.id ? "" : event.id,
                  )
                }
                patientId={patientId}
                token={token}
                onDelete={onDelete}
                onSaveEdit={onSaveEdit}
              />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function FactReviewPanel({
  event,
  ocrText,
  review,
  setReview,
  editingFactId,
  setEditingFactId,
  editValue,
  setEditValue,
  extraFacts,
  setExtraFacts,
}: {
  event: ExtractedEvent;
  ocrText?: string | null;
  review: Record<
    string,
    { action: "confirmed" | "corrected" | "rejected"; value?: string }
  >;
  setReview: React.Dispatch<
    React.SetStateAction<
      Record<
        string,
        { action: "confirmed" | "corrected" | "rejected"; value?: string }
      >
    >
  >;
  editingFactId: string | null;
  setEditingFactId: (id: string | null) => void;
  editValue: string;
  setEditValue: (v: string) => void;
  extraFacts: ExtractedFact[];
  setExtraFacts: (f: ExtractedFact[]) => void;
}) {
  const allFacts = [...event.facts, ...extraFacts];
  const [addOpen, setAddOpen] = useState(false);
  const [addName, setAddName] = useState("");
  const [addValue, setAddValue] = useState("");
  const [addUnit, setAddUnit] = useState("");
  const [addRef, setAddRef] = useState("");
  const [addInterp, setAddInterp] = useState("");

  const counts = allFacts.reduce(
    (acc, f) => {
      const r = review[f.fact_id];
      if (r?.action === "confirmed") acc.confirmed += 1;
      else if (r?.action === "corrected") acc.corrected += 1;
      else if (r?.action === "rejected") acc.rejected += 1;
      return acc;
    },
    { confirmed: 0, corrected: 0, rejected: 0 },
  );

  function parseRef(text: string) {
    const t = text.trim();
    if (!t) return null;
    const m = t.match(/^([\d.]+)\s*[-~–]\s*([\d.]+)\s*([\d.]+)?/);
    if (m) {
      return {
        low: Number(m[1]),
        high: Number(m[2]),
        text: t,
        unit: m[3] || null,
      };
    }
    return { low: null, high: null, text: t, unit: null };
  }

  function handleAdd() {
    const name = addName.trim();
    if (!name) return;
    const valueText = addValue.trim();
    const num = Number(valueText);
    const isNum = valueText !== "" && !Number.isNaN(num);
    const fact: ExtractedFact = {
      fact_id: `manual_${Date.now()}`,
      kind: "observation",
      concept: { display: name, local_display: name },
      value: isNum
        ? { type: "quantity", number: num, unit: addUnit.trim() || null }
        : { type: "text", text: valueText },
      reference_range: parseRef(addRef),
      interpretation: addInterp || null,
      confidence: 1,
      status: "extracted",
    };
    setExtraFacts([...extraFacts, fact]);
    setAddOpen(false);
    setAddName("");
    setAddValue("");
    setAddUnit("");
    setAddRef("");
    setAddInterp("");
  }

  return (
    <div className="fact-review">
      <div className="fact-review-head">
        <div>
          <span className="eyebrow">字段提取结果</span>
          <h3>{event.title}</h3>
          <p className="fact-review-meta">
            {event.summary ?? `${allFacts.length} 项检验结果`}
            {event.event_time?.start ? ` · ${event.event_time.start}` : ""}
            {event.institution ? ` · ${event.institution}` : ""}
          </p>
        </div>
        <div className="fact-review-badges">
          <span className="review-badge">
            已确认 {counts.confirmed} / 修正 {counts.corrected} / 排除{" "}
            {counts.rejected} / 共 {allFacts.length}
          </span>
          <button
            className="mini-button add-fact-btn"
            onClick={() => setAddOpen((v) => !v)}
          >
            {addOpen ? "收起" : "+ 手动添加一条"}
          </button>
        </div>
      </div>

      {addOpen && (
        <div className="add-fact-form">
          <div className="add-fact-grid">
            <label className="add-fact-field">
              <span>项目名称 *</span>
              <input
                value={addName}
                onChange={(e) => setAddName(e.target.value)}
                placeholder="如：血尿酸"
              />
            </label>
            <label className="add-fact-field">
              <span>数值 / 结果 *</span>
              <input
                value={addValue}
                onChange={(e) => setAddValue(e.target.value)}
                placeholder="如：360 或 阴性"
              />
            </label>
            <label className="add-fact-field">
              <span>单位（选填）</span>
              <input
                value={addUnit}
                onChange={(e) => setAddUnit(e.target.value)}
                placeholder="如：umol/L"
              />
            </label>
            <label className="add-fact-field">
              <span>参考范围（选填）</span>
              <input
                value={addRef}
                onChange={(e) => setAddRef(e.target.value)}
                placeholder="如：155-422"
              />
            </label>
            <label className="add-fact-field">
              <span>判定（选填）</span>
              <select
                value={addInterp}
                onChange={(e) => setAddInterp(e.target.value)}
              >
                <option value="">未知</option>
                <option value="normal">正常</option>
                <option value="high">偏高</option>
                <option value="low">偏低</option>
              </select>
            </label>
          </div>
          <div className="add-fact-actions">
            <button
              className="mini-button"
              disabled={!addName.trim()}
              onClick={handleAdd}
            >
              添加
            </button>
            <button
              className="mini-button ghost"
              onClick={() => setAddOpen(false)}
            >
              取消
            </button>
          </div>
        </div>
      )}

      <div className="fact-list">
        {allFacts.map((fact) => {
          const r = review[fact.fact_id];
          const lowConf = (fact.confidence ?? 1) < 0.6;
          const isEditing = editingFactId === fact.fact_id;
          const cls = [
            "fact-row",
            r?.action === "confirmed" ? "is-confirmed" : "",
            r?.action === "corrected" ? "is-corrected" : "",
            r?.action === "rejected" ? "is-rejected" : "",
            lowConf && !r ? "is-lowconf" : "",
          ]
            .filter(Boolean)
            .join(" ");

          const valueText =
            fact.value.type === "quantity"
              ? `${fact.value.number}${fact.value.unit ? ` ${fact.value.unit}` : ""}`
              : fact.value.text;
          const shownValue = r?.action === "corrected" && r.value ? r.value : valueText;
          const refText = fact.reference_range?.text
            ? fact.reference_range.text
            : fact.reference_range &&
                (fact.reference_range.low != null || fact.reference_range.high != null)
              ? `${fact.reference_range.low ?? "≤"} - ${
                  fact.reference_range.high ?? "≥"
                }${fact.reference_range.unit ? ` ${fact.reference_range.unit}` : ""}`
              : "—";

          return (
            <div className={cls} key={fact.fact_id}>
              <div className="fact-name">
                <span className="fact-display">{fact.concept.display}</span>
                {fact.concept.local_display &&
                  fact.concept.local_display !== fact.concept.display && (
                    <span className="fact-local">{fact.concept.local_display}</span>
                  )}
              </div>

              <div className="fact-value">
                {isEditing ? (
                  <div className="fact-edit">
                    <input
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          setReview((prev) => ({
                            ...prev,
                            [fact.fact_id]: { action: "corrected", value: editValue },
                          }));
                          setEditingFactId(null);
                        }
                      }}
                      autoFocus
                    />
                    <button
                      className="mini-button"
                      onClick={() => {
                        setReview((prev) => ({
                          ...prev,
                          [fact.fact_id]: { action: "corrected", value: editValue },
                        }));
                        setEditingFactId(null);
                      }}
                    >
                      保存
                    </button>
                    <button
                      className="mini-button ghost"
                      onClick={() => setEditingFactId(null)}
                    >
                      取消
                    </button>
                  </div>
                ) : (
                  <>
                    <span className="fact-number">{shownValue}</span>
                    {fact.interpretation === "high" && (
                      <span className="flag-badge is-high">偏高</span>
                    )}
                    {fact.interpretation === "low" && (
                      <span className="flag-badge is-low">偏低</span>
                    )}
                    {fact.interpretation === "normal" && (
                      <span className="flag-badge is-normal">正常</span>
                    )}
                  </>
                )}
              </div>

              <div className="fact-ref">
                <span className="ref-text">参考 {refText}</span>
                {lowConf && !r && (
                  <span className="lowconf-tag">建议核对</span>
                )}
              </div>

              <div className="fact-actions">
                {r?.action === "confirmed" ? (
                  <span className="mini-button done">已确认</span>
                ) : r?.action === "corrected" ? (
                  <span className="mini-button done">已修正</span>
                ) : r?.action === "rejected" ? (
                  <span className="mini-button done">已排除</span>
                ) : (
                  <>
                    <button
                      className="mini-button"
                      onClick={() =>
                        setReview((prev) => ({
                          ...prev,
                          [fact.fact_id]: { action: "confirmed" },
                        }))
                      }
                    >
                      确认
                    </button>
                    <button
                      className="mini-button ghost"
                      onClick={() => {
                        const v =
                          fact.value.type === "quantity"
                            ? String(fact.value.number)
                            : fact.value.text;
                        setEditValue(v);
                        setEditingFactId(fact.fact_id);
                      }}
                    >
                      修改
                    </button>
                    <button
                      className="mini-button ghost danger"
                      onClick={() =>
                        setReview((prev) => ({
                          ...prev,
                          [fact.fact_id]: { action: "rejected" },
                        }))
                      }
                    >
                      排除
                    </button>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <details className="ocr-source">
        <summary>查看 OCR 原始识别文字</summary>
        <pre className="ocr-result-text">{ocrText ?? "（无）"}</pre>
      </details>
      <p className="fact-review-foot">
        审核结果暂存在当前页面（确认/修正/排除），入库与时间轴接入是下一步。
      </p>
    </div>
  );
}

function UploadPage({
  session,
  uploadState,
  setUploadState,
  onUploaded,
}: {
  session: AuthResponse;
  uploadState: UploadState;
  setUploadState: (
    next: UploadState | ((prev: UploadState) => UploadState),
  ) => void;
  onUploaded: () => void;
}) {
  const [elapsedSec, setElapsedSec] = useState(0);
  const [review, setReview] = useState<
    Record<string, { action: "confirmed" | "corrected" | "rejected"; value?: string }>
  >({});
  const [editingFactId, setEditingFactId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [extraFacts, setExtraFacts] = useState<ExtractedFact[]>([]);
  const [retryOpen, setRetryOpen] = useState(false);
  const [added, setAdded] = useState(false);
  const [addError, setAddError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastFileRef = useRef<File | null>(null);

  useEffect(() => {
    if (uploadState.status !== "processing" || !uploadState.startedAt) {
      return;
    }
    const tick = () => {
      setElapsedSec(
        Math.max(0, Math.floor((Date.now() - uploadState.startedAt!) / 1000)),
      );
    };
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [uploadState.status, uploadState.startedAt]);

  async function handleFile(file: File) {
    lastFileRef.current = file;
    setExtraFacts([]);
    setReview({});
    setAdded(false);
    setAddError("");
    setUploadState({
      fileName: file.name,
      status: "processing",
      message: null,
      startedAt: Date.now(),
      result: null,
      qualityWarning: false,
    });
    try {
      const result = await uploadHealthRecord(
        file,
        session.user.patient_id,
        session.token,
      );
      setUploadState({
        fileName: file.name,
        status: "complete",
        message: null,
        startedAt: null,
        result,
        qualityWarning: Boolean(result.quality && !result.quality.ok),
      });
      onUploaded();
    } catch (err) {
      setUploadState({
        fileName: file.name,
        status: "offline",
        message: err instanceof Error ? err.message : "Upload failed",
        startedAt: null,
        result: null,
        qualityWarning: false,
      });
    }
  }

  const { fileName, status, message } = uploadState;
  // Soft asymptotic progress while waiting; jumps to 100% on complete.
  const progressPct =
    status === "complete"
      ? 100
      : status === "offline"
        ? 0
        : Math.min(92, Math.round(100 * (1 - Math.exp(-elapsedSec / 55))));

  async function handleAddToTimeline() {
    const ev = uploadState.result?.extracted_event;
    if (!ev) return;
    setAddError("");
    const facts: ExtractedFact[] = [];
    for (const f of ev.facts) {
      const r = review[f.fact_id];
      if (r?.action === "rejected") continue;
      let fact: ExtractedFact = { ...f, time: { precision: "unknown" } };
      if (r?.action === "corrected" && r.value) {
        const trimmed = r.value.trim();
        const m = trimmed.match(/^(-?\d+(?:\.\d+)?)\s*(.*)$/);
        fact = {
          ...fact,
          value: m
            ? { type: "quantity", number: Number(m[1]), unit: m[2] || null }
            : { type: "text", text: trimmed },
          status: "corrected",
        };
      } else if (r?.action === "confirmed") {
        fact = { ...fact, status: "confirmed" };
      }
      facts.push(fact);
    }
    for (const f of extraFacts) {
      facts.push({ ...f, time: { precision: "unknown" } });
    }
    const payload: ExtractedEvent = {
      ...ev,
      patient_id: session.user.patient_id,
      source_artifact_ids: [uploadState.result!.artifact.artifact_id],
      facts,
    };
    try {
      await addTimelineEvent(session.user.patient_id, session.token, payload);
      setAdded(true);
      onUploaded();
    } catch (err) {
      setAddError(err instanceof Error ? err.message : "加入时间轴失败");
    }
  }

  return (
    <section className="page-stack">
      <div className="section-title">
        <div>
          <span className="eyebrow">新增健康记录</span>
          <h1>把散落的报告放进同一条时间轴</h1>
          <p>推荐上传报告原图或 PDF。上传后即可在时间轴中查看。</p>
        </div>
      </div>

      <div className="upload-layout">
        <article className="upload-card">
          <label className="drop-zone">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.pdf"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleFile(file);
              }}
            />
            <span className="upload-symbol">+</span>
            <strong>{fileName ?? "选择图片或 PDF"}</strong>
            <p>检验单、影像报告、处方、门诊病历、出院记录均可</p>
            <span className="secondary-button">选择文件</span>
          </label>

          {status !== "idle" && (
            <div
              className={
                status === "complete"
                  ? "processing-box is-complete"
                  : status === "offline"
                    ? "processing-box is-error"
                    : "processing-box is-running"
              }
            >
              {status === "processing" ? (
                <div className="parse-progress">
                  <div className="parse-progress-head">
                    <strong>健康数据解析中...</strong>
                    <span className="parse-progress-pct">{progressPct}%</span>
                  </div>
                  <div
                    className="parse-progress-track"
                    role="progressbar"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={progressPct}
                  >
                    <div
                      className="parse-progress-fill"
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                </div>
              ) : (
                <div className="processing-title">
                  <div>
                    <strong>
                      {status === "complete" ? "提取完成" : "处理失败"}
                    </strong>
                    <span>
                      {status === "complete"
                        ? added
                          ? "已加入时间轴，可在「时间轴」中查看"
                          : "提取完成，审核后可点击「加入时间轴」"
                        : (message ?? "请检查 API 与登录状态")}
                    </span>
                  </div>
                  {status === "complete" ? (
                    <div className="retry-actions">
                      <button
                        className="primary-button add-timeline-btn"
                        onClick={() => void handleAddToTimeline()}
                        disabled={added}
                      >
                        {added ? "已加入时间轴 ✓" : "加入时间轴"}
                      </button>
                      <button
                        className="mini-button"
                        onClick={() => setRetryOpen(true)}
                      >
                        重新识别
                      </button>
                      <span className="processing-badge">DONE</span>
                    </div>
                  ) : (
                    <span className="processing-badge">ERROR</span>
                  )}
                  {status === "complete" && addError && (
                    <p className="add-error">{addError}</p>
                  )}
                </div>
              )}
            </div>
          )}
        </article>

        <article className="panel ingest-explainer">
          <span className="eyebrow">OCR 识别结果</span>
          <h2>
            {status === "idle" && "上传后在此查看识别原文"}
            {status === "processing" && "正在识别文字…"}
            {status === "complete" && "识别完成"}
            {status === "offline" && "识别失败"}
          </h2>
          {status === "idle" && (
            <p>
              上传报告图片或 PDF 后，这里会直接展示 OCR 识别出的文字内容，
              方便核对识别质量。
            </p>
          )}
          {status === "processing" && (
            <p>正在调用本地 OCR 引擎识别报告文字，请稍候…</p>
          )}
          {status === "complete" && uploadState.result && (
            uploadState.result.extracted_event?.facts?.length ? (
              <FactReviewPanel
                event={uploadState.result.extracted_event}
                ocrText={uploadState.result.artifact.ocr_text}
                review={review}
                setReview={setReview}
                editingFactId={editingFactId}
                setEditingFactId={setEditingFactId}
                editValue={editValue}
                setEditValue={setEditValue}
                extraFacts={extraFacts}
                setExtraFacts={setExtraFacts}
              />
            ) : (
              <div className="ocr-result">
                <div className="ocr-meta">
                  <span>
                    引擎：{uploadState.result.artifact.ocr_engine ?? "—"}
                  </span>
                  <span>
                    文字块：{uploadState.result.artifact.ocr_blocks?.length ?? 0}
                  </span>
                  <span>状态：{uploadState.result.status}</span>
                </div>
                <p className="no-facts-hint">
                  未能自动识别出结构化字段，下面是原始识别文字，可先人工确认
                </p>
                <pre className="ocr-result-text">
                  {uploadState.result.artifact.ocr_text || "（未识别到文字）"}
                </pre>
              </div>
            )
          )}
          {status === "offline" && <p>{message ?? "处理失败，请重试"}</p>}
        </article>
      </div>

      {uploadState.qualityWarning && uploadState.result?.quality && (
        <div
          className="quality-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="清晰度提示"
        >
          <div className="quality-modal">
            <span className="eyebrow">识别质量提示</span>
            <h2>清晰度过低，建议重新拍摄</h2>
            <p>
              这张图片{uploadState.result.quality.issues.join("、")}，可能影响识别准确率。
            </p>
            <ul className="quality-issues">
              {uploadState.result.quality.issues.map((issue) => (
                <li key={issue}>
                  {issue === "分辨率过低"
                    ? "图片太小，文字细节不足"
                    : issue === "图片模糊"
                      ? "画面发虚，文字边缘不清晰"
                      : "画面偏暗，文字看不清"}
                </li>
              ))}
            </ul>
            <div className="quality-actions">
              <button
                className="primary-button"
                onClick={() => setUploadState(emptyUploadState)}
              >
                重新拍摄
              </button>
              <button
                className="secondary-button"
                onClick={() =>
                  setUploadState((prev) => ({
                    ...prev,
                    qualityWarning: false,
                  }))
                }
              >
                仍要上传
              </button>
            </div>
          </div>
        </div>
      )}

      {retryOpen && (
        <div
          className="quality-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="重新识别"
        >
          <div className="quality-modal">
            <span className="eyebrow">重新识别</span>
            <h2>清晰图片更有助于识别</h2>
            <p>
              当前结果可能不够准确。你可以重新选择一张更清晰的照片，或对同一张图再次识别。
            </p>
            <div className="quality-actions">
              <button
                className="primary-button"
                onClick={() => {
                  setRetryOpen(false);
                  fileInputRef.current?.click();
                }}
              >
                重新选择图片
              </button>
              <button
                className="secondary-button"
                onClick={() => {
                  setRetryOpen(false);
                  if (lastFileRef.current) {
                    void handleFile(lastFileRef.current);
                  }
                }}
              >
                重新识别这张图
              </button>
              <button
                className="mini-button ghost"
                onClick={() => setRetryOpen(false)}
              >
                取消
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function AssistantPage({
  session,
  events,
  onAdd,
  assistantState,
  setAssistantState,
}: {
  session: AuthResponse;
  events: TimelineEvent[];
  onAdd: () => void;
  assistantState: AssistantState;
  setAssistantState: (
    next: AssistantState | ((prev: AssistantState) => AssistantState),
  ) => void;
}) {
  const { messages, question, busy, error, contextPreview } = assistantState;

  const observations = events.flatMap((event) =>
    event.facts.map((fact) => ({
      date: event.displayDate,
      label: fact.label,
      value: fact.value,
      id: fact.id,
    })),
  );

  async function handleAnalyze() {
    setAssistantState((prev) => ({
      ...prev,
      busy: true,
      error: null,
    }));
    try {
      const reply = await runHealthAnalysis(
        session.user.patient_id,
        session.token,
      );
      setAssistantState((prev) => ({
        ...prev,
        busy: false,
        contextPreview: reply.context ?? null,
        messages: [
          ...prev.messages,
          {
            role: "assistant",
            content: reply.answer,
          },
        ],
      }));
    } catch (err) {
      setAssistantState((prev) => ({
        ...prev,
        busy: false,
        error: err instanceof Error ? err.message : "分析失败",
      }));
    }
  }

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    const history = messages;
    setAssistantState((prev) => ({
      ...prev,
      question: "",
      busy: true,
      error: null,
      messages: [...prev.messages, { role: "user", content: text }],
    }));
    try {
      const reply = await askHealthAssistant(
        session.user.patient_id,
        session.token,
        text,
        history,
      );
      setAssistantState((prev) => ({
        ...prev,
        busy: false,
        contextPreview: reply.context ?? null,
        messages: [
          ...prev.messages,
          { role: "assistant", content: reply.answer },
        ],
      }));
    } catch (err) {
      setAssistantState((prev) => ({
        ...prev,
        busy: false,
        error: err instanceof Error ? err.message : "提问失败",
      }));
    }
  }

  if (events.length === 0) {
    return (
      <section className="page-stack">
        <div className="section-title">
          <div>
            <span className="eyebrow">AI 助手</span>
            <h1>健康分析</h1>
            <p>没有你的健康记录时，这里不会显示任何示例病历。</p>
          </div>
        </div>
        <article className="panel">
          <h2>暂无可用上下文</h2>
          <p>请先上传检查报告。AI 只会基于你本人时间轴里的结构化事实作答。</p>
          <button className="primary-button" type="button" onClick={onAdd}>
            去添加记录
          </button>
        </article>
      </section>
    );
  }

  const previewObservations =
    contextPreview?.observations?.slice(0, 8) ??
    observations.slice(0, 8).map((item) => ({
      time: item.date,
      concept: item.label,
      value: { type: "text", text: item.value },
    }));

  return (
    <section className="page-stack">
      <div className="section-title">
        <div>
          <span className="eyebrow">AI 助手</span>
          <h1>基于结构化时间轴做健康分析</h1>
          <p>
            点击「健康分析」读取后台 Agent Context；也可以继续追问交互。
          </p>
        </div>
        <button
          className="primary-button"
          type="button"
          onClick={() => void handleAnalyze()}
          disabled={busy}
        >
          {busy ? "分析中..." : "健康分析"}
        </button>
      </div>

      <div className="assistant-grid">
        <article className="panel context-card">
          <div className="panel-head compact">
            <div>
              <span className="eyebrow">Agent Context</span>
              <h2>高效结构化上下文</h2>
            </div>
            <span className="context-range">{events.length} 条事件</span>
          </div>

          <div className="context-section">
            <span className="context-label">OBSERVATIONS</span>
            {previewObservations.length === 0 ? (
              <p className="empty-copy">暂无结构化指标。</p>
            ) : (
              previewObservations.map((fact, index) => {
                const value = fact.value as
                  | {
                      type?: string;
                      number?: number;
                      unit?: string;
                      text?: string;
                    }
                  | null
                  | undefined;
                let display = "-";
                if (value?.type === "quantity" && value.number != null) {
                  display = `${value.number}${value.unit ? ` ${value.unit}` : ""}`;
                } else if (value?.text) {
                  display = value.text;
                }
                return (
                  <div className="context-row" key={`${fact.concept}-${index}`}>
                    <span>{String(fact.time ?? "").slice(0, 10)}</span>
                    <strong>{fact.concept}</strong>
                    <span>{display}</span>
                  </div>
                );
              })
            )}
          </div>

          <div className="context-section">
            <span className="context-label">DATA GAPS</span>
            {(contextPreview?.data_gaps ?? ["点击健康分析后显示缺口"]).map(
              (gap) => (
                <div className="gap-row" key={gap}>
                  {gap}
                </div>
              ),
            )}
          </div>
        </article>

        <article className="assistant-answer assistant-chat">
          <span className="assistant-badge">交互分析</span>
          <div className="chat-log">
            {messages.length === 0 ? (
              <p>
                {busy
                  ? "正在分析你的健康数据，切换页面也不会中断，稍后再回来即可查看结果。"
                  : "先点右上角「健康分析」，系统会读取你的结构化时间轴并给出总结。之后可以继续提问，例如：“肌酐最近有什么变化？”"}
              </p>
            ) : (
              messages.map((turn, index) => (
                <div
                  className={
                    turn.role === "user" ? "chat-bubble user" : "chat-bubble ai"
                  }
                  key={`${turn.role}-${index}`}
                >
                  <strong>{turn.role === "user" ? "你" : "HealthApp AI"}</strong>
                  {turn.role === "assistant" ? (
                    <div className="chat-content markdown-body">
                      <Markdown>{turn.content}</Markdown>
                    </div>
                  ) : (
                    <div className="chat-content">{turn.content}</div>
                  )}
                </div>
              ))
            )}
            {busy && messages.length > 0 && (
              <p className="empty-copy">正在生成回复...</p>
            )}
          </div>

          {error && <p className="auth-error">{error}</p>}

          <form className="chat-form" onSubmit={handleAsk}>
            <input
              value={question}
              onChange={(e) =>
                setAssistantState((prev) => ({
                  ...prev,
                  question: e.target.value,
                }))
              }
              placeholder="继续提问，例如：有哪些数据缺口？"
              disabled={busy}
            />
            <button className="primary-button" type="submit" disabled={busy}>
              发送
            </button>
          </form>
          <div className="citation-box">
            <strong>回答只基于你的结构化事实</strong>
            <span>不是医疗诊断，重要结论请与医生确认</span>
          </div>
        </article>
      </div>
    </section>
  );
}

export default function App() {
  const [session, setSession] = useState<AuthResponse | null>(() =>
    loadStoredSession(),
  );
  const [tab, setTab] = useState<Tab>("overview");
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [loadingTimeline, setLoadingTimeline] = useState(false);
  const [uploadState, setUploadState] = useState<UploadState>(emptyUploadState);
  const [assistantState, setAssistantState] =
    useState<AssistantState>(emptyAssistantState);

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: "overview", label: "概览" },
    { key: "timeline", label: "时间轴" },
    { key: "upload", label: "添加记录" },
    { key: "assistant", label: "AI 助手" },
  ];

  async function refreshTimeline(active: AuthResponse) {
    setLoadingTimeline(true);
    try {
      const rows = await fetchTimeline(active.user.patient_id, active.token);
      setTimeline(rows.map(mapApiEvent));
    } catch {
      setTimeline([]);
    } finally {
      setLoadingTimeline(false);
    }
  }

  useEffect(() => {
    if (!session) return;
    void refreshTimeline(session);
  }, [session?.token]);

  if (!session) {
    return <AuthScreen onAuthenticated={setSession} />;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setTab("overview")}>
          <span className="brand-mark">H</span>
          <span>
            <strong>HealthApp</strong>
            <small>Personal health timeline</small>
          </span>
        </button>

        <nav className="desktop-nav" aria-label="主导航">
          {tabs.map((item) => (
            <button
              key={item.key}
              className={tab === item.key ? "nav-button active" : "nav-button"}
              onClick={() => setTab(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <ProfileMenu
          session={session}
          onLogout={() => {
            void logoutUser(session.token).then(() => {
              clearSession();
              setSession(null);
              setTimeline([]);
              setUploadState(emptyUploadState);
              setAssistantState(emptyAssistantState);
            });
          }}
        />
      </header>

      <main>
        {tab === "overview" && (
          <Overview
            user={session.user}
            eventCount={timeline.length}
            onOpenTimeline={() =>
              setTab(timeline.length === 0 ? "upload" : "timeline")
            }
          />
        )}
        {tab === "timeline" && (
          <Timeline
            events={timeline}
            loading={loadingTimeline}
            onAdd={() => setTab("upload")}
            patientId={session.user.patient_id}
            token={session.token}
            onDelete={(event) => {
              const ok = window.confirm(
                `确认删除「${event.title}」吗？原报告文件也会一并删除。`,
              );
              if (!ok) return;
              void deleteTimelineEvent(
                session.user.patient_id,
                event.id,
                session.token,
              )
                .then(() => refreshTimeline(session))
                .catch((err: unknown) => {
                  window.alert(
                    err instanceof Error ? err.message : "删除失败",
                  );
                });
            }}
            onSaveEdit={async (event, draft) => {
              try {
                await updateTimelineEvent(
                  session.user.patient_id,
                  event.id,
                  session.token,
                  {
                    title: draft.title,
                    summary: draft.summary,
                    institution: draft.institution,
                    event_time_start: draft.eventDate,
                    facts: draft.facts.map((fact) => ({
                      fact_id: fact.fact_id,
                      label: fact.label,
                      value_text: fact.value_text,
                      interpretation: fact.interpretation,
                      status: "corrected",
                    })),
                  },
                );
                await refreshTimeline(session);
              } catch (err) {
                window.alert(err instanceof Error ? err.message : "保存失败");
                throw err;
              }
            }}
          />
        )}
        {tab === "upload" && (
          <UploadPage
            session={session}
            uploadState={uploadState}
            setUploadState={setUploadState}
            onUploaded={() => void refreshTimeline(session)}
          />
        )}
        {tab === "assistant" && (
          <AssistantPage
            session={session}
            events={timeline}
            onAdd={() => setTab("upload")}
            assistantState={assistantState}
            setAssistantState={setAssistantState}
          />
        )}
      </main>

      <nav className="mobile-nav" aria-label="移动端主导航">
        {tabs.map((item) => (
          <button
            key={item.key}
            className={
              tab === item.key
                ? "mobile-nav-button active"
                : "mobile-nav-button"
            }
            onClick={() => setTab(item.key)}
          >
            <span className="mobile-dot" />
            {item.label}
          </button>
        ))}
      </nav>
    </div>
  );
}
