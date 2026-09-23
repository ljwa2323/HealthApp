import { useMemo, useState } from "react";
import { uploadHealthRecord } from "./api";
import { creatinineTrend, events, kidneyContext } from "./data/demo";
import type { TimelineEvent, TrendPoint } from "./types";

type Tab = "overview" | "timeline" | "upload" | "assistant";
type Filter = "all" | TimelineEvent["type"];

const eventTypeLabel: Record<TimelineEvent["type"], string> = {
  laboratory: "检验",
  imaging: "影像",
  visit: "门诊",
  medication: "用药",
  procedure: "操作",
  vital: "生命体征",
};

function TrendChart({ points }: { points: TrendPoint[] }) {
  const width = 520;
  const height = 170;
  const paddingX = 28;
  const paddingY = 20;
  const values = points.map((point) => point.value);
  const min = Math.min(...values) - 5;
  const max = Math.max(...values) + 5;
  const range = Math.max(max - min, 1);

  const coords = points.map((point, index) => {
    const x =
      paddingX +
      (index / Math.max(points.length - 1, 1)) * (width - paddingX * 2);
    const y =
      paddingY +
      ((max - point.value) / range) * (height - paddingY * 2);
    return { ...point, x, y };
  });

  const path = coords
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  return (
    <div className="trend-wrap">
      <svg
        className="trend-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="肌酐纵向变化图"
      >
        <line x1="28" x2="492" y1="140" y2="140" className="grid-line" />
        <line x1="28" x2="492" y1="88" y2="88" className="grid-line" />
        <line x1="28" x2="492" y1="36" y2="36" className="grid-line" />
        <path d={path} className="trend-path" />
        {coords.map((point) => (
          <g key={point.date}>
            <circle cx={point.x} cy={point.y} r="5.5" className="trend-dot" />
            <text x={point.x} y={point.y - 12} className="trend-value">
              {point.value}
            </text>
          </g>
        ))}
      </svg>
      <div className="trend-labels">
        {points.map((point) => (
          <span key={point.date}>{point.date}</span>
        ))}
      </div>
    </div>
  );
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

function EventCard({
  event,
  expanded,
  onToggle,
}: {
  event: TimelineEvent;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <article className="event-card">
      <button className="event-head" onClick={onToggle}>
        <span className={`event-icon event-${event.type}`}>
          {eventTypeLabel[event.type].slice(0, 1)}
        </span>
        <span className="event-copy">
          <span className="event-meta">
            {event.displayDate} · {eventTypeLabel[event.type]}
          </span>
          <strong>{event.title}</strong>
          <span>{event.institution}</span>
        </span>
        <span className="event-toggle">{expanded ? "收起" : "查看"}</span>
      </button>

      <p className="event-summary">{event.summary}</p>

      {expanded && (
        <div className="event-detail">
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
          <div className="source-box">
            <span>原始证据</span>
            <strong>{event.source}</strong>
            <button type="button">查看原报告</button>
          </div>
        </div>
      )}
    </article>
  );
}

function Overview({ onOpenTimeline }: { onOpenTimeline: () => void }) {
  return (
    <section className="page-stack">
      <div className="hero-card">
        <div>
          <span className="eyebrow">纵向健康概览</span>
          <h1>你的健康记录，按时间连续起来</h1>
          <p>
            已整理 2 年 4 个月记录。系统优先展示变化趋势，而不是孤立的一次检查。
          </p>
        </div>
        <button className="primary-button" onClick={onOpenTimeline}>
          查看完整时间轴
        </button>
      </div>

      <div className="metric-grid">
        <article className="metric-card">
          <span>已归档记录</span>
          <strong>28</strong>
          <small>来自 4 家医疗机构</small>
        </article>
        <article className="metric-card">
          <span>可追溯事实</span>
          <strong>164</strong>
          <small>均可回到原始报告</small>
        </article>
        <article className="metric-card attention-card">
          <span>需要继续观察</span>
          <strong>2</strong>
          <small>1 项趋势 + 1 项影像发现</small>
        </article>
      </div>

      <article className="panel trend-panel">
        <div className="panel-head">
          <div>
            <span className="eyebrow">纵向趋势</span>
            <h2>肌酐</h2>
          </div>
          <div className="current-value">
            <strong>88</strong>
            <span>μmol/L</span>
          </div>
        </div>
        <TrendChart points={creatinineTrend} />
        <div className="insight-note">
          <span className="note-mark">i</span>
          <p>
            2024年6月至2026年3月从 74 升至 88 μmol/L。单个结果不足以说明疾病，
            纵向变化值得在后续检查中继续观察。
          </p>
        </div>
      </article>

      <div className="two-column">
        <article className="panel">
          <div className="panel-head compact">
            <div>
              <span className="eyebrow">最近记录</span>
              <h2>新增到时间轴</h2>
            </div>
            <button className="text-button" onClick={onOpenTimeline}>
              全部记录
            </button>
          </div>
          <div className="recent-list">
            {events.slice(0, 3).map((event) => (
              <div className="recent-row" key={event.id}>
                <span className={`mini-dot dot-${event.type}`} />
                <div>
                  <strong>{event.title}</strong>
                  <span>
                    {event.displayDate} · {event.institution}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-head compact">
            <div>
              <span className="eyebrow">数据完整性</span>
              <h2>下一步可补充</h2>
            </div>
          </div>
          <div className="gap-list">
            <div>
              <span>01</span>
              <p>当前没有尿白蛋白/肌酐比记录</p>
            </div>
            <div>
              <span>02</span>
              <p>2026年3月后尚无复查肾功能</p>
            </div>
            <div>
              <span>03</span>
              <p>部分旧处方缺少明确停药时间</p>
            </div>
          </div>
        </article>
      </div>
    </section>
  );
}

function Timeline() {
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState(events[0].id);

  const filtered = useMemo(
    () => events.filter((event) => filter === "all" || event.type === filter),
    [filter],
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
        <button className="primary-button">添加记录</button>
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

      <div className="timeline-shell">
        <aside className="year-rail">
          <strong>2026</strong>
          <span />
          <strong>2025</strong>
          <span />
          <strong>2024</strong>
        </aside>
        <div className="timeline-list">
          {filtered.map((event) => (
            <EventCard
              key={event.id}
              event={event}
              expanded={expanded === event.id}
              onToggle={() =>
                setExpanded((current) => (current === event.id ? "" : event.id))
              }
            />
          ))}
        </div>
      </div>
    </section>
  );
}

function UploadPage() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [status, setStatus] = useState<
    "idle" | "processing" | "complete" | "offline"
  >("idle");

  async function handleFile(file: File) {
    setFileName(file.name);
    setStatus("processing");
    try {
      await uploadHealthRecord(file);
      setStatus("complete");
    } catch {
      window.setTimeout(() => setStatus("offline"), 700);
    }
  }

  return (
    <section className="page-stack">
      <div className="section-title">
        <div>
          <span className="eyebrow">新增健康记录</span>
          <h1>把散落的报告放进同一条时间轴</h1>
          <p>推荐上传报告原图或 PDF。系统先 OCR，再由 LLM 进行结构化。</p>
        </div>
      </div>

      <div className="upload-layout">
        <article className="upload-card">
          <label className="drop-zone">
            <input
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
            <div className="processing-box">
              <div className="processing-title">
                <strong>
                  {status === "complete"
                    ? "处理完成"
                    : status === "offline"
                      ? "界面演示模式"
                      : "正在整理记录"}
                </strong>
                <span>
                  {status === "complete"
                    ? "已进入待确认阶段"
                    : status === "offline"
                      ? "启动 API 后可执行真实上传"
                      : "请保持当前页面打开"}
                </span>
              </div>
              <div className="pipeline">
                <div className="done">文件读取</div>
                <div className={status === "processing" ? "current" : "done"}>
                  OCR
                </div>
                <div className={status === "processing" ? "" : "done"}>
                  文档识别
                </div>
                <div className={status === "complete" ? "done" : ""}>
                  LLM 规整
                </div>
                <div>确认并入库</div>
              </div>
            </div>
          )}
        </article>

        <article className="panel ingest-explainer">
          <span className="eyebrow">为什么不是直接让 VL 诊断</span>
          <h2>图像负责提供原文，模型负责整理语义</h2>
          <p>
            对绝大多数检验和影像报告，真正需要长期保存的是正式报告中的文字结论和结构化指标。
            OCR 后保留原文，再用 LLM 做标准化，可以同时获得可追溯性和较好的结构化质量。
          </p>
          <div className="flow-list">
            <div><span>1</span><p>保存原始图片或 PDF</p></div>
            <div><span>2</span><p>OCR 输出文字与版面坐标</p></div>
            <div><span>3</span><p>LLM 转成标准化 Clinical Facts</p></div>
            <div><span>4</span><p>低置信度结果由用户确认</p></div>
          </div>
        </article>
      </div>
    </section>
  );
}

function AssistantPage() {
  return (
    <section className="page-stack">
      <div className="section-title">
        <div>
          <span className="eyebrow">Agent context</span>
          <h1>AI 不需要每次重读全部病历</h1>
          <p>
            系统根据问题从时间轴中组装最小必要上下文，并保留证据引用。
          </p>
        </div>
        <button className="primary-button">开始提问</button>
      </div>

      <div className="assistant-grid">
        <article className="panel context-card">
          <div className="panel-head compact">
            <div>
              <span className="eyebrow">示例任务</span>
              <h2>{kidneyContext.task}</h2>
            </div>
            <span className="context-range">{kidneyContext.timeRange}</span>
          </div>

          <div className="context-section">
            <span className="context-label">OBSERVATIONS</span>
            {kidneyContext.facts.map((fact) => (
              <div className="context-row" key={fact.join("-")}>
                <span>{fact[0]}</span>
                <strong>{fact[1]}</strong>
                <span>{fact[2]}</span>
              </div>
            ))}
          </div>

          <div className="context-section">
            <span className="context-label">DATA GAPS</span>
            {kidneyContext.gaps.map((gap) => (
              <div className="gap-row" key={gap}>{gap}</div>
            ))}
          </div>

          <div className="context-section">
            <span className="context-label">EVIDENCE REFS</span>
            <div className="tag-row">
              {kidneyContext.evidence.map((item) => (
                <span className="code-tag" key={item}>{item}</span>
              ))}
            </div>
          </div>
        </article>

        <article className="assistant-answer">
          <span className="assistant-badge">AI 辅助总结</span>
          <h2>最近两年的肾功能记录显示什么？</h2>
          <p>
            当前记录中，肌酐从 2025年1月的 76 μmol/L 增至
            2026年3月的 88 μmol/L。最近一次 eGFR 为
            82 mL/min/1.73m²。
          </p>
          <p>
            这些数据可以描述为目前记录中存在上升趋势，但只有少量时间点，
            不能仅凭这组数据判断持续性肾功能异常。后续解释还需要结合复查结果、
            尿检以及临床背景。
          </p>
          <div className="citation-box">
            <strong>本次回答使用了 3 条事实</strong>
            <span>覆盖 2025-01-08 至 2026-03-18</span>
            <button>查看证据</button>
          </div>
        </article>
      </div>
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: "overview", label: "概览" },
    { key: "timeline", label: "时间轴" },
    { key: "upload", label: "添加记录" },
    { key: "assistant", label: "AI 助手" },
  ];

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

        <button className="profile-button">
          <span>示例用户</span>
          <span className="avatar">D</span>
        </button>
      </header>

      <main>
        {tab === "overview" && (
          <Overview onOpenTimeline={() => setTab("timeline")} />
        )}
        {tab === "timeline" && <Timeline />}
        {tab === "upload" && <UploadPage />}
        {tab === "assistant" && <AssistantPage />}
      </main>

      <nav className="mobile-nav" aria-label="移动端主导航">
        {tabs.map((item) => (
          <button
            key={item.key}
            className={tab === item.key ? "mobile-nav-button active" : "mobile-nav-button"}
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
