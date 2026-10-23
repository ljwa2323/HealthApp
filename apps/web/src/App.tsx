import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";
import Markdown from "react-markdown";
import {
  AuthResponse,
  AuthUser,
  askHealthAssistant,
  AssistantReply,
  changePassword,
  clearSession,
  ConversationSummary,
  deleteConversation,
  deleteTimelineEvent,
  ExtractedEvent,
  ExtractedFact,
  fetchArtifactObjectUrl,
  fetchTimeline,
  getConversation,
  listConversations,
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
  splitDocument,
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
  conversationId: string | null;
  conversations: ConversationSummary[];
  searchQuery: string;
  loadingConversations: boolean;
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
  conversationId: null,
  conversations: [],
  searchQuery: "",
  loadingConversations: false,
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

// Per-document-type extra fields (backend fills fact.attributes with these
// keys). Rendered as small chips under each fact row.
const ATTR_LABELS: Record<string, string> = {
  dose: "剂量",
  frequency: "频次",
  route: "用法",
  quantity: "数量",
  body_part: "部位",
  conclusion: "结论",
  department: "科室",
  chief_complaint: "主诉",
  admission_date: "入院日期",
  discharge_date: "出院日期",
  anesthesia: "麻醉方式",
  surgeon: "术者",
  pre_diagnosis: "术前诊断",
  post_diagnosis: "术后诊断",
  duration: "持续时间",
  detail: "备注",
};

// Which attribute keys are shown for each document type (fallback: all).
const EVENT_ATTR_KEYS: Record<string, string[]> = {
  medication: ["dose", "frequency", "route", "quantity", "detail"],
  imaging: ["body_part", "conclusion", "detail"],
  visit: ["department", "chief_complaint", "detail"],
  hospitalization: ["department", "admission_date", "discharge_date", "detail"],
  procedure: ["anesthesia", "surgeon", "pre_diagnosis", "post_diagnosis", "detail"],
  symptom: ["duration", "detail"],
  laboratory: [],
  vital: [],
  other: [],
};

function factAttrChips(
  eventType: string | undefined,
  attrs: Record<string, unknown> | undefined,
): Array<{ label: string; value: string }> {
  if (!attrs) return [];
  const keys =
    EVENT_ATTR_KEYS[eventType ?? ""] ??
    Object.keys(attrs).filter((k) => k !== "detail");
  const chips: Array<{ label: string; value: string }> = [];
  for (const key of keys) {
    const val = attrs[key];
    if (val == null || String(val) === "") continue;
    chips.push({
      label: ATTR_LABELS[key] ?? key,
      value: String(val),
    });
  }
  return chips;
}

// Quality-warning modal copy, split by the backend's quality branch
// ("text" = white-background reports, "imaging" = film photo shots).
const QUALITY_ISSUE_TEXT: Record<string, Record<string, string>> = {
  text: {
    分辨率过低: "图片太小，文字细节不足",
    图片模糊: "画面发虚，文字边缘不清晰",
    光线太暗: "画面偏暗，文字看不清",
  },
  imaging: {
    分辨率过低: "影像分辨率不足，细节可能缺失",
    画面模糊: "影像发虚，结构边缘不清晰",
    画面过暗: "影像整体过暗，结构难以辨认",
    画面过曝: "画面过曝或有反光，细节被白光淹没",
  },
};

function qualityModalTitle(docType?: string | null): string {
  return docType === "imaging"
    ? "影像清晰度不足，建议重新拍摄"
    : "清晰度过低，建议重新拍摄";
}

function qualityImpactCopy(docType?: string | null): string {
  return docType === "imaging"
    ? "可能影响后续判读准确率"
    : "可能影响识别准确率";
}

function qualityIssueText(issue: string, docType?: string | null): string {
  return QUALITY_ISSUE_TEXT[docType ?? ""]?.[issue] ?? issue;
}

type UploadQuality = NonNullable<UploadResult["quality"]>;

// Shared quality-warning modal for both branches: "text" (documents) and
// "imaging" (film photos). Only warns — the user can still keep the photo.
function QualityWarningModal({
  quality,
  onRetake,
  onKeep,
}: {
  quality: UploadQuality;
  onRetake: () => void;
  onKeep: () => void;
}) {
  return (
    <div
      className="quality-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="清晰度提示"
    >
      <div className="quality-modal">
        <span className="eyebrow">识别质量提示</span>
        <h2>{qualityModalTitle(quality.doc_type)}</h2>
        <p>
          这张图片{quality.issues.join("、")}，{qualityImpactCopy(quality.doc_type)}。
        </p>
        <ul className="quality-issues">
          {quality.issues.map((issue) => (
            <li key={issue}>{qualityIssueText(issue, quality.doc_type)}</li>
          ))}
        </ul>
        <div className="quality-actions">
          <button className="primary-button" type="button" onClick={onRetake}>
            重新拍摄
          </button>
          <button className="secondary-button" type="button" onClick={onKeep}>
            仍要上传
          </button>
        </div>
      </div>
    </div>
  );
}

/** Normalize a date string into a YYYY-MM-DD sort key ("" when absent). */
function dateSortKey(value: string): string {
  const v = String(value ?? "").trim();
  if (!v) return "";
  const m = v.match(/^(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})/);
  if (m) {
    return `${m[1]}-${m[2].padStart(2, "0")}-${m[3].padStart(2, "0")}`;
  }
  return v.slice(0, 10);
}

function mapApiEvent(event: TimelineEventApi): TimelineEvent {
  const start = event.event_time?.start ?? "";
  const displayDate = String(start).slice(0, 10);
  const mappedType = (
    [
      "laboratory", "imaging", "visit", "medication", "procedure", "vital",
      "hospitalization", "symptom", "other",
    ].includes(event.event_type)
      ? event.event_type
      : "other"
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
        attributes: fact.attributes,
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
        <span className="event-head-actions">
          <span className="event-toggle">{expanded ? "收起" : "查看"}</span>
          <span
            className="event-delete"
            onClick={(clickEvent) => {
              // The head row is one button; stop the click from also
              // toggling the card open.
              clickEvent.stopPropagation();
              onDelete(event);
            }}
          >
            删除记录
          </span>
        </span>
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
                      {factAttrChips(event.type, fact.attributes).length > 0 && (
                        <span className="fact-attrs">
                          {factAttrChips(event.type, fact.attributes).map((chip) => (
                            <span className="attr-chip" key={chip.label}>
                              {chip.label}：{chip.value}
                            </span>
                          ))}
                        </span>
                      )}
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

  const filtered = useMemo(() => {
    const list = events.filter(
      (event) => filter === "all" || event.type === filter,
    );
    // Sort by the document date, newest at the top (further down = older).
    // Events without a date go to the bottom.
    return [...list].sort((a, b) => {
      const da = dateSortKey(a.date);
      const db = dateSortKey(b.date);
      if (!da && !db) return 0;
      if (!da) return 1;
      if (!db) return -1;
      return db.localeCompare(da);
    });
  }, [events, filter]);

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
          <div className="timeline-list">
            {filtered.map((event) => (
              <div
                key={event.id}
                className="timeline-item"
                data-type={event.type}
              >
                <span className="timeline-date">
                  {event.displayDate || "日期未填"}
                </span>
                <span className="timeline-dot" aria-hidden="true" />
                <EventCard
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
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

type CustomField = {
  id: string;
  name: string;
  type: "input" | "select";
  options: string[];
};

function FactReviewPanel({
  event,
  ocrText,
  crossCheck,
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
  crossCheck?: Array<{
    fact_id: string;
    concept: string;
    status: "verified" | "not_found" | "conflict";
    detail: string;
  }>;
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
  // Custom structure: user designs the record's fields (name + input/select),
  // then the add form renders from it and every field becomes one fact.
  const [customOpen, setCustomOpen] = useState(false);
  const [structDraft, setStructDraft] = useState<CustomField[]>([]);
  const [structNameDraft, setStructNameDraft] = useState("");
  const [structure, setStructure] = useState<CustomField[]>([]);
  const [recordName, setRecordName] = useState("");
  const [formValues, setFormValues] = useState<Record<string, string>>({});

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

  function factValueText(fact: ExtractedFact): string {
    const v = fact.value;
    if (!v) return "";
    if (v.type === "quantity") {
      return v.unit ? `${v.number} ${v.unit}` : String(v.number);
    }
    return v.text;
  }

  function openStructEditor() {
    if (structure.length) {
      setStructDraft(structure.map((f) => ({ ...f, options: [...f.options] })));
    } else {
      // Start with several blank rows so multi-field records (e.g. four
      // temperature readings a day) can be designed without adding rows.
      const now = Date.now();
      setStructDraft(
        Array.from({ length: 4 }, (_, i) => ({
          id: `f_${now}_${i}`,
          name: "",
          type: "input",
          options: [],
        })),
      );
    }
    setStructNameDraft(recordName);
    setCustomOpen(true);
  }

  function applyStructure() {
    const valid = structDraft
      .map((f) => ({ ...f, name: f.name.trim() }))
      .filter((f) => f.name !== "");
    if (!valid.length) return;
    setStructure(valid);
    setRecordName(structNameDraft.trim() || "自定义记录");
    // Never auto-fill on apply; the user decides with the "自动填入" button.
    setFormValues({});
    setCustomOpen(false);
    setAddOpen(true);
  }

  function autoFillValues() {
    setFormValues((current) => {
      const next: Record<string, string> = { ...current };
      for (const f of structure) {
        if (next[f.id]) continue; // keep what the user already typed
        const hit = allFacts.find((x) => x.concept.display === f.name);
        if (hit) next[f.id] = factValueText(hit);
      }
      return next;
    });
  }

  function handleAdd() {
    if (structure.length > 0) {
      // A custom structure describes ONE record: all designed fields belong
      // to this single entry. Merge the filled values into one fact instead
      // of creating one fact per field.
      const parts: string[] = [];
      const attrs: Record<string, unknown> = {};
      for (const f of structure) {
        const raw = (formValues[f.id] ?? "").trim();
        if (!raw) continue;
        parts.push(`${f.name}：${raw}`);
        attrs[f.name] = raw;
      }
      if (!parts.length) return;
      setExtraFacts([
        ...extraFacts,
        {
          fact_id: `manual_${Date.now()}`,
          kind: "observation",
          concept: {
            display: recordName.trim() || "自定义记录",
            local_display: recordName.trim() || "自定义记录",
          },
          value: { type: "text", text: parts.join(" / ") },
          reference_range: null,
          interpretation: null,
          confidence: 1,
          status: "extracted",
          attributes: attrs,
        },
      ]);
      setAddOpen(false);
      setFormValues({});
      return;
    }

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
          <button
            className="mini-button struct-btn"
            onClick={openStructEditor}
          >
            {structure.length ? `调整结构（${structure.length} 项）` : "自定义结构"}
          </button>
        </div>
      </div>

      {customOpen && (
        <div className="struct-editor">
          <div className="struct-editor-head">
            <span className="eyebrow">自定义结构</span>
            <span className="struct-hint">
              一条记录一个结构：字段就是这一条记录要包含的数据项；确定后按此录入，可用"自动填入"放入已识别数据
            </span>
          </div>
          <label className="struct-name-field">
            <span>记录名称</span>
            <input
              value={structNameDraft}
              onChange={(e) => setStructNameDraft(e.target.value)}
              placeholder="如：体温记录"
            />
          </label>
          {structDraft.map((f, index) => (
            <div className="struct-row" key={f.id}>
              <input
                className="struct-name"
                value={f.name}
                onChange={(e) =>
                  setStructDraft((d) =>
                    d.map((x, i) =>
                      i === index ? { ...x, name: e.target.value } : x,
                    ),
                  )
                }
                placeholder="字段名称，如：体温"
              />
              <select
                className="struct-type"
                value={f.type}
                onChange={(e) =>
                  setStructDraft((d) =>
                    d.map((x, i) =>
                      i === index
                        ? { ...x, type: e.target.value as CustomField["type"] }
                        : x,
                    ),
                  )
                }
              >
                <option value="input">填写</option>
                <option value="select">选择</option>
              </select>
              {f.type === "select" && (
                <input
                  className="struct-options"
                  value={f.options.join("，")}
                  onChange={(e) =>
                    setStructDraft((d) =>
                      d.map((x, i) =>
                        i === index
                          ? {
                              ...x,
                              options: e.target.value
                                .split(/[,，]/)
                                .map((s) => s.trim())
                                .filter(Boolean),
                            }
                          : x,
                      ),
                    )
                  }
                  placeholder="选项，逗号分隔，如：正常，偏高，偏低"
                />
              )}
              <button
                className="mini-button ghost"
                onClick={() =>
                  setStructDraft((d) => d.filter((_, i) => i !== index))
                }
              >
                删除
              </button>
            </div>
          ))}
          <div className="struct-actions">
            <button
              className="mini-button"
              onClick={() =>
                setStructDraft((d) => [
                  ...d,
                  {
                    id: `f_${Date.now()}_${d.length}`,
                    name: "",
                    type: "input",
                    options: [],
                  },
                ])
              }
            >
              + 添加一项
            </button>
            <button
              className="mini-button struct-apply"
              onClick={applyStructure}
            >
              确定
            </button>
            <button
              className="mini-button ghost"
              onClick={() => setCustomOpen(false)}
            >
              取消
            </button>
          </div>
        </div>
      )}

      {addOpen && (
        <div className="add-fact-form">
          <div className="add-fact-grid">
            {structure.length > 0 ? (
              <>
                <label className="add-fact-field struct-record-name">
                  <span>记录名称</span>
                  <input
                    value={recordName}
                    onChange={(e) => setRecordName(e.target.value)}
                    placeholder="如：体温记录"
                  />
                </label>
                {structure.map((f) => (
                  <label className="add-fact-field" key={f.id}>
                    <span>{f.name}</span>
                    {f.type === "select" ? (
                      <select
                        value={formValues[f.id] ?? ""}
                        onChange={(e) =>
                          setFormValues((v) => ({ ...v, [f.id]: e.target.value }))
                        }
                      >
                        <option value="">请选择</option>
                        {f.options.map((o) => (
                          <option key={o} value={o}>
                            {o}
                          </option>
                        ))}
                      </select>
                    ) : (
                    <input
                      value={formValues[f.id] ?? ""}
                      onChange={(e) =>
                        setFormValues((v) => ({ ...v, [f.id]: e.target.value }))
                      }
                    />
                  )}
                </label>
              ))}
              </>
            ) : (
              <>
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
              </>
            )}
          </div>
          <div className="add-fact-actions">
            {structure.length > 0 && (
              <button
                className="mini-button autofill-btn"
                onClick={autoFillValues}
                title="按字段名称匹配，把已识别出的数据填入空项"
              >
                自动填入
              </button>
            )}
            <button
              className="mini-button"
              disabled={structure.length === 0 && !addName.trim()}
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

      {crossCheck && crossCheck.length > 0 && (
        <div className="cross-check-summary">
          {(() => {
            const verified = crossCheck.filter((c) => c.status === "verified").length;
            const conflict = crossCheck.filter((c) => c.status === "conflict").length;
            const notFound = crossCheck.filter(
              (c) => c.status === "not_found",
            ).length;
            return (
              <>
                <strong>OCR × AI 比对</strong>
                <span className="cc-ok">{verified} 项一致</span>
                {conflict > 0 && (
                  <span className="cc-bad">{conflict} 项与 OCR 不一致</span>
                )}
                {notFound > 0 && (
                  <span className="cc-warn">{notFound} 项 OCR 未读到</span>
                )}
                <span className="cc-hint">
                  （不一致 / 未读到仅提示，不代表 AI 错误，请人工核对）
                </span>
              </>
            );
          })()}
        </div>
      )}

      <div className="fact-list">
        {allFacts.map((fact) => {
          const r = review[fact.fact_id];
          const cc = crossCheck?.find((c) => c.fact_id === fact.fact_id);
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
                {cc && cc.status !== "verified" && (
                  <span
                    className={`cc-badge cc-badge-${cc.status}`}
                    title={cc.detail}
                  >
                    {cc.status === "conflict" ? "与OCR不一致" : "OCR未读到"}
                  </span>
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

              {factAttrChips(event.event_type, fact.attributes).length > 0 && (
                <div className="fact-attrs">
                  {factAttrChips(event.event_type, fact.attributes).map((chip) => (
                    <span className="attr-chip" key={chip.label}>
                      {chip.label}：{chip.value}
                    </span>
                  ))}
                </div>
              )}

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
  const dragDepthRef = useRef(0);
  const [dragActive, setDragActive] = useState(false);
  // The document date shown in the add-to-timeline step. Prefilled from
  // OCR when the report date was recognized; otherwise the user picks it
  // (year-month-day). The timeline sorts by this date, not upload time.
  const [eventDate, setEventDate] = useState("");
  // Full-text view: for narrative documents it is the AI-extracted raw
  // text; for structured ones it falls back to the OCR text. Editable so
  // the user can fix it before adding to the timeline.
  const [fullText, setFullText] = useState("");
  // Which review view is shown. The AI decides by default; the user can
  // override when the AI picked the wrong mode.
  const [viewMode, setViewMode] = useState<"structured" | "narrative">(
    "structured",
  );
  // On-demand split: narrative documents are NOT pre-split at upload
  // (keeps the initial extraction fast). When the user taps 分条展示, we
  // call the /split endpoint once and cache the facts here.
  const [splitFacts, setSplitFacts] = useState<ExtractedFact[] | null>(null);
  const [splitCrossCheck, setSplitCrossCheck] = useState<
    UploadResult["cross_check"]
  >([]);
  const [splitting, setSplitting] = useState(false);
  const [splitError, setSplitError] = useState("");

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

  function handleDragEnter(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    dragDepthRef.current += 1;
    setDragActive(true);
  }

  function handleDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
  }

  function handleDragLeave(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    dragDepthRef.current -= 1;
    if (dragDepthRef.current <= 0) {
      dragDepthRef.current = 0;
      setDragActive(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    dragDepthRef.current = 0;
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    const okType =
      file.type.startsWith("image/") || file.type === "application/pdf";
    if (!okType) {
      window.alert("请选择图片或 PDF 文件");
      return;
    }
    void handleFile(file);
  }

  async function handleFile(file: File) {
    lastFileRef.current = file;
    setExtraFacts([]);    setReview({});
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
      setEventDate(
        result.extracted_event?.event_time?.start?.slice(0, 10) ?? "",
      );
      setFullText(
        result.extracted_event?.narrative ?? result.artifact.ocr_text ?? "",
      );
      // Default to the AI's pick; the toggle lets the user override it.
      setViewMode(
        result.extracted_event?.narrative ? "narrative" : "structured",
      );
      setSplitFacts(null);
      setSplitCrossCheck([]);
      setSplitting(false);
      setSplitError("");
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
  // Pure imaging path: the quality gate classified this as a film shot and
  // OCR was skipped — the right panel must not present itself as an OCR
  // text result.
  const isPureImaging =
    status === "complete" &&
    uploadState.result?.artifact.document_type === "imaging" &&
    !uploadState.result?.artifact.ocr_text;
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
    // Facts source: pre-split facts, or the on-demand /split result.
    const srcFacts = ev.facts?.length ? ev.facts : (splitFacts ?? []);
    const facts: ExtractedFact[] = [];
    for (const f of srcFacts) {
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
      // The user picks the view: split items (facts) or full text.
      facts: viewMode === "narrative" ? [] : facts,
      narrative: viewMode === "narrative" ? fullText : undefined,
      // Timeline sorts by the document date, not upload time: keep the
      // OCR-recognized date unless the user changed/cleared it, and let
      // the user fill in the date when the image had no recognizable one.
      event_time: {
        start: eventDate || null,
        end: null,
        precision: eventDate ? "day" : "unknown",
        source: "patient",
      },
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
          <label
            className={dragActive ? "drop-zone is-dragging" : "drop-zone"}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
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
            <strong>{fileName ?? "选择或拖入图片 / PDF"}</strong>
            <p>
              把报告文件直接拖到这里，或点击选择；检验单、影像报告、处方、门诊病历均可
            </p>
            <span className="secondary-button">
              {dragActive ? "松开即可上传" : "选择文件"}
            </span>
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
                    {status === "complete" && !added && (
                      <label className="event-date-field">
                        <span className="event-date-hint">
                          {eventDate
                            ? "单据日期（已从图上识别，请核对）"
                            : "单据日期（图上未识别到，请选择）"}
                        </span>
                        <input
                          type="date"
                          value={eventDate}
                          onChange={(event) => setEventDate(event.target.value)}
                        />
                      </label>
                    )}
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
          <span className="eyebrow">
            {isPureImaging ? "影像处理结果" : "OCR 识别结果"}
          </span>
          <h2>
            {status === "idle" && "上传后在此查看识别原文"}
            {status === "processing" && "正在识别…"}
            {status === "complete" && (isPureImaging ? "影像原图已保存" : "识别完成")}
            {status === "offline" && "识别失败"}
          </h2>
          {status === "idle" && (
            <p>
              上传报告图片或 PDF 后，这里会直接展示 OCR 识别出的文字内容，
              方便核对识别质量。
            </p>
          )}
          {status === "processing" && (
            <p>正在分析图片，请稍候…</p>
          )}
          {status === "complete" && uploadState.result && (() => {
            const ev = uploadState.result.extracted_event;
            const hasFacts = !!ev?.facts?.length;
            const hasNarrative = !!ev?.narrative;
            if (hasFacts || hasNarrative) {
              // Facts shown in the split view: pre-split for structured
              // documents, or the on-demand /split result for narratives.
              const displayFacts = hasFacts
                ? ev!.facts
                : (splitFacts ?? []);
              const displayCrossCheck = hasFacts
                ? (uploadState.result.cross_check ?? [])
                : splitCrossCheck;
              return (
                <div className="review-views">
                  <div className="view-toggle">
                    <button
                      type="button"
                      className={viewMode === "structured" ? "active" : ""}
                      onClick={() => {
                        setSplitError("");
                        setViewMode("structured");
                        // Narrative docs are not pre-split (keeps upload
                        // fast); split on demand the first time the user
                        // asks for the split view.
                        const cur = uploadState.result?.extracted_event;
                        if (
                          cur &&
                          !cur.facts?.length &&
                          splitFacts === null &&
                          !splitting
                        ) {
                          setSplitting(true);
                          void splitDocument(
                            session.user.patient_id,
                            uploadState.result!.artifact.artifact_id,
                            session.token,
                          )
                            .then((r) => {
                              setSplitFacts(r.facts);
                              setSplitCrossCheck(r.cross_check ?? []);
                            })
                            .catch((err) =>
                              setSplitError(
                                err instanceof Error ? err.message : "分条失败",
                              ),
                            )
                            .finally(() => setSplitting(false));
                        }
                      }}
                    >
                      {splitting ? "正在分条…" : "分条展示"}
                    </button>
                    <button
                      type="button"
                      className={viewMode === "narrative" ? "active" : ""}
                      onClick={() => setViewMode("narrative")}
                    >
                      全部展示
                    </button>
                    <span className="view-toggle-hint">
                      AI 自动判断，可手动切换（如识别不准）
                    </span>
                  </div>
                  {viewMode === "structured" ? (
                    displayFacts.length > 0 ? (
                      <FactReviewPanel
                        event={{ ...ev!, facts: displayFacts }}
                        ocrText={uploadState.result.artifact.ocr_text}
                        crossCheck={displayCrossCheck}
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
                        {splitting ? (
                          <p className="no-facts-hint">
                            正在按你的要求分条提取，请稍候…
                          </p>
                        ) : splitError ? (
                          <p className="add-error">分条失败：{splitError}</p>
                        ) : (
                          <p className="no-facts-hint">
                            该文档未提取到可分条的项目，可在「全部展示」中查看并编辑原文
                          </p>
                        )}
                      </div>
                    )
                  ) : (
                    <div className="ocr-result">
                      <div className="ocr-meta">
                        <span>
                          引擎：{uploadState.result.artifact.ocr_engine ?? "—"}
                        </span>
                        <span>状态：{uploadState.result.status}</span>
                      </div>
                      <p className="no-facts-hint">
                        以下是按原文顺序提取的完整文字，可直接编辑，核对后加入时间轴
                      </p>
                      <textarea
                        className="narrative-edit"
                        rows={14}
                        value={fullText}
                        onChange={(e) => setFullText(e.target.value)}
                        placeholder="（原文）"
                      />
                    </div>
                  )}
                </div>
              );
            }
            if (
              uploadState.result.artifact.document_type === "imaging" &&
              !uploadState.result.artifact.ocr_text
            ) {
              return (
                <div className="ocr-result">
                  <p className="no-facts-hint">
                    检测为影像原图：未识别到文字，原图已保存。确认后点击「加入时间轴」，会同时出现在时间轴的「全部」和「影像」板块。
                  </p>
                </div>
              );
            }
            return (
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
            );
          })()}
          {status === "offline" && <p>{message ?? "处理失败，请重试"}</p>}
        </article>
      </div>

      {uploadState.qualityWarning && uploadState.result?.quality && (
        <QualityWarningModal
          quality={uploadState.result.quality}
          onRetake={() => setUploadState(emptyUploadState)}
          onKeep={() =>
            setUploadState((prev) => ({
              ...prev,
              qualityWarning: false,
            }))
          }
        />
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
  const { messages, question, busy, error, contextPreview, conversationId, conversations, searchQuery, loadingConversations } = assistantState;
  const chatEndRef = useRef<HTMLDivElement>(null);

  // 加载对话列表
  async function refreshConversations() {
    setAssistantState((prev) => ({ ...prev, loadingConversations: true }));
    try {
      const list = await listConversations(session.user.patient_id, session.token, searchQuery);
      setAssistantState((prev) => ({ ...prev, conversations: list, loadingConversations: false }));
    } catch {
      setAssistantState((prev) => ({ ...prev, loadingConversations: false }));
    }
  }

  useEffect(() => {
    void refreshConversations();
  }, [searchQuery]);

  // 自动滚动到底部
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function selectConversation(convId: string) {
    if (busy) return;
    try {
      const detail = await getConversation(session.user.patient_id, session.token, convId);
      setAssistantState((prev) => ({
        ...prev,
        conversationId: convId,
        messages: detail.messages,
        question: "",
        error: null,
        contextPreview: null,
      }));
    } catch (err) {
      setAssistantState((prev) => ({ ...prev, error: err instanceof Error ? err.message : "加载对话失败" }));
    }
  }

  function newChat() {
    if (busy) return;
    setAssistantState((prev) => ({
      ...prev,
      conversationId: null,
      messages: [],
      question: "",
      error: null,
      contextPreview: null,
    }));
  }

  async function handleDeleteConversation(convId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!window.confirm("确定删除这个对话？")) return;
    try {
      await deleteConversation(session.user.patient_id, session.token, convId);
      if (conversationId === convId) {
        newChat();
      }
      void refreshConversations();
    } catch (err) {
      setAssistantState((prev) => ({ ...prev, error: err instanceof Error ? err.message : "删除失败" }));
    }
  }

  const observations = events.flatMap((event) =>
    event.facts.map((fact) => ({
      date: event.displayDate,
      label: fact.label,
      value: fact.value,
      id: fact.id,
    })),
  );

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
        conversationId,
      );
      setAssistantState((prev) => ({
        ...prev,
        busy: false,
        contextPreview: reply.context ?? null,
        conversationId: reply.conversation_id ?? prev.conversationId,
        messages: [
          ...prev.messages,
          { role: "assistant", content: reply.answer },
        ],
      }));
      // 新对话发完后刷新列表
      if (!conversationId && reply.conversation_id) {
        void refreshConversations();
      }
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

  return (
    <section className="page-stack">
      <div className="section-title">
        <div>
          <span className="eyebrow">AI 助手</span>
          <h1>健康咨询</h1>
        </div>
      </div>

      <div className="assistant-layout">
        {/* 左侧对话列表 */}
        <aside className="conv-sidebar">
          <div className="conv-sidebar-head">
            <input
              className="conv-search"
              type="text"
              placeholder="搜索对话..."
              value={searchQuery}
              onChange={(e) =>
                setAssistantState((prev) => ({ ...prev, searchQuery: e.target.value }))
              }
            />
            <button className="primary-button conv-new-btn" type="button" onClick={newChat} disabled={busy}>
              + 新对话
            </button>
          </div>
          <div className="conv-list">
            {loadingConversations ? (
              <p className="empty-copy">加载中...</p>
            ) : conversations.length === 0 ? (
              <p className="empty-copy">暂无历史对话</p>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.conversation_id}
                  className={`conv-item ${conversationId === conv.conversation_id ? "active" : ""}`}
                  onClick={() => void selectConversation(conv.conversation_id)}
                >
                  <div className="conv-item-title">{conv.title}</div>
                  <div className="conv-item-preview">{conv.preview || " "}</div>
                  <button
                    className="conv-delete"
                    onClick={(e) => void handleDeleteConversation(conv.conversation_id, e)}
                    title="删除对话"
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </aside>

        {/* 右侧聊天区 */}
        <article className="assistant-answer assistant-chat">
          <div className="chat-log">
            {messages.length === 0 ? (
              <p className="empty-copy" style={{ padding: 40, textAlign: "center" }}>
                {busy
                  ? "正在生成回复..."
                  : "输入问题开始对话，例如：我的肝功能指标怎么样？"}
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
            {busy && <p className="empty-copy">正在生成回复...</p>}
            <div ref={chatEndRef} />
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
              placeholder="输入你的问题..."
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
