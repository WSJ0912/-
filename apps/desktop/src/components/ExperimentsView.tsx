import { useState } from "react";
import { FileText, FlaskConical, FolderOpen, Sparkles } from "lucide-react";
import { type CurvePoint, type ExperimentBundle, validateExperimentBundle } from "../contracts";
import { labels, request } from "../api";

type Props = {
  onError: (reason: unknown) => void;
  onNotice: (message: string) => void;
};

const labelNames = new Map<string, string>(labels.map(([key, value]) => [key, value]));

function formatMetric(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(4);
  if (value === null || typeof value !== "object" || Array.isArray(value)) return "--";

  const interval = value as Record<string, unknown>;
  const estimate = typeof interval.estimate === "number" && Number.isFinite(interval.estimate)
    ? interval.estimate.toFixed(4)
    : "--";
  const lower = typeof interval.lower === "number" && Number.isFinite(interval.lower)
    ? interval.lower.toFixed(4)
    : "--";
  const upper = typeof interval.upper === "number" && Number.isFinite(interval.upper)
    ? interval.upper.toFixed(4)
    : "--";
  const sampleSize = Number.isInteger(interval.n) && Number(interval.n) >= 0
    ? `, n=${interval.n}`
    : "";
  return `${estimate} [${lower}, ${upper}]${sampleSize}`;
}

function formatCardValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "--";
  if (Array.isArray(value)) return value.map(String).join("；") || "--";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function CurveChart({ title, points }: { title: string; points: CurvePoint[] }) {
  if (!points.length) return <div className="curve-empty"><strong>{title}</strong><span>未提供曲线点</span></div>;
  const width = 280;
  const height = 190;
  const padding = 30;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;
  const coordinates = points.map((point) => `${padding + point.x * chartWidth},${height - padding - point.y * chartHeight}`).join(" ");
  return <figure className="curve-chart">
    <figcaption>{title}<small>仅绘制实验包内 {points.length} 个原始点</small></figcaption>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} 折线图`}>
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="curve-axis" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="curve-axis" />
      <text x={padding - 4} y={height - padding + 16}>0</text>
      <text x={width - padding - 5} y={height - padding + 16}>1</text>
      <text x={padding - 18} y={padding + 4}>1</text>
      <polyline points={coordinates} className="curve-line" />
      {points.map((point, index) => <circle key={`${point.x}-${point.y}-${index}`} cx={padding + point.x * chartWidth} cy={height - padding - point.y * chartHeight} r="2.6" className="curve-point" />)}
    </svg>
  </figure>;
}

export function ExperimentsView({ onError, onNotice }: Props) {
  const [bundle, setBundle] = useState<ExperimentBundle | null>(null);
  const [assistant, setAssistant] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  async function select() {
    try {
      setBusy(true);
      const imported: unknown = await window.medicalApi.importExperiment();
      if (!imported) return;
      if (!validateExperimentBundle(imported)) throw new Error("实验包未通过共享 JSON Schema 校验");
      setBundle(imported);
      setAssistant(null);
      onNotice("实验包已校验并导入；仅展示包内真实聚合结果");
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  async function summarize() {
    if (!bundle) return;
    try {
      setBusy(true);
      setAssistant(await request("/api/assistant/experiment", "POST", {
        aggregateMetrics: bundle.aggregateMetrics,
        perClassMetrics: bundle.perClassMetrics,
        experimentNotes: "",
      }));
      onNotice("实验摘要已生成，结论仍需研究者核对");
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  const perClassRows = bundle ? Object.entries(bundle.perClassMetrics) : [];
  const metricNames = Array.from(new Set(perClassRows.flatMap(([, metrics]) => Object.keys(metrics))));
  const curves = bundle ? Object.entries(bundle.curves) : [];
  const modelCard = bundle ? Object.entries(bundle.modelCard) : [];

  return <div className="view-stack">
    <div className="section-header">
      <div><p className="eyebrow">RESEARCH / AGGREGATE ONLY</p><h2>实验结果</h2><p className="muted">`.medexperiment` 通过共享契约后才显示；未知指标保持 --，曲线不补点。</p></div>
      <div className="experiment-actions">{bundle && <button className="quiet-button" onClick={() => void summarize()} disabled={busy}><Sparkles size={16} />助手摘要</button>}<button className="primary-button" onClick={() => void select()} disabled={busy}><FolderOpen size={17} />导入实验包</button></div>
    </div>

    {bundle ? <>
      <div className="metric-grid">{Object.entries(bundle.aggregateMetrics).map(([key, value]) => <div className="metric-card" key={key}><span>{key}</span><strong>{formatMetric(value)}</strong><small>聚合结果</small></div>)}</div>

      <section className="panel experiment-panel">
        <div className="panel-title"><div><h3>逐类指标</h3><p>展示实验包原值；不可计算项显示 --。</p></div><span className="tag neutral">{perClassRows.length} 类</span></div>
        {perClassRows.length === 0 ? <div className="table-empty">未提供逐类指标</div> : <div className="experiment-table-wrap"><table className="experiment-table"><thead><tr><th>观察</th>{metricNames.map((metric) => <th key={metric}>{metric}</th>)}</tr></thead><tbody>{perClassRows.map(([label, metrics]) => <tr key={label}><th><span>{labelNames.get(label) || label}</span><small>{label}</small></th>{metricNames.map((metric) => <td key={metric}>{formatMetric(metrics[metric])}</td>)}</tr>)}</tbody></table></div>}
      </section>

      <section className="panel experiment-panel">
        <div className="panel-title"><div><h3>ROC / PR 曲线</h3><p>折线只连接包内真实点，不生成或平滑缺失数据。</p></div><span className="tag neutral">{curves.length} 组</span></div>
        {curves.length === 0 ? <div className="table-empty">未提供曲线点</div> : <div className="curve-groups">{curves.map(([key, curve]) => <article className="curve-group" key={key}><header><strong>{labelNames.get(curve.label) || curve.label}</strong><span>{curve.method} · seed {curve.seed}</span></header><div className="curve-grid"><CurveChart title="ROC" points={curve.roc} /><CurveChart title="PR" points={curve.pr} /></div></article>)}</div>}
      </section>

      <section className="panel experiment-panel">
        <div className="panel-title"><div><h3>模型卡</h3><p>用途、限制和数据声明来自实验包，不由界面推断。</p></div><FileText size={18} /></div>
        {modelCard.length === 0 ? <div className="table-empty">未提供模型卡</div> : <dl className="model-card-grid">{modelCard.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{formatCardValue(value)}</dd></div>)}</dl>}
      </section>

      {assistant && <section className="panel assistant-summary"><div className="panel-title"><div><h3>实验助手摘要</h3><p>仅基于聚合指标，不含患者级数据。</p></div><Sparkles size={18} /></div><p>{assistant.summary}</p>{assistant.limitations?.length > 0 && <ul>{assistant.limitations.map((item: string) => <li key={item}>{item}</li>)}</ul>}</section>}
    </> : <div className="empty-panel tall"><div className="empty-icon"><FlaskConical size={21} /></div><h3>尚无实验包</h3><p>导入后可查看聚合指标、逐类指标、真实曲线点和模型卡。</p></div>}
  </div>;
}
