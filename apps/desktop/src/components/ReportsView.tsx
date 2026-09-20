import { useEffect, useState } from "react";
import { FileBarChart2, FileText, LockKeyhole, Plus, RefreshCw, Sparkles } from "lucide-react";
import { request } from "../api";
import { type ReportRevision, validateReportRevision } from "../contracts";
import { IconButton } from "./IconButton";

type Props = {
  onError: (reason: unknown) => void;
  onNotice: (message: string) => void;
};

const REPORT_TEMPLATE = "检查所见：\n\n印象：\n";

function parseRevision(value: unknown): ReportRevision {
  if (!validateReportRevision(value)) {
    throw new Error("报告版本未通过共享 JSON Schema 校验");
  }
  return value;
}

function parseRevisionList(value: unknown): ReportRevision[] {
  if (!Array.isArray(value) || !value.every(validateReportRevision)) {
    throw new Error("报告列表未通过共享 JSON Schema 校验");
  }
  return value;
}

function displayTime(value: string): string {
  if (!value) return "时间未记录";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN");
}

export function ReportsView({ onError, onNotice }: Props) {
  const [reports, setReports] = useState<ReportRevision[]>([]);
  const [activeReportId, setActiveReportId] = useState<string | null>(null);
  const [revisions, setRevisions] = useState<ReportRevision[]>([]);
  const [selectedRevision, setSelectedRevision] = useState<ReportRevision | null>(null);
  const [studyId, setStudyId] = useState("");
  const [body, setBody] = useState(REPORT_TEMPLATE);
  const [assistantCautions, setAssistantCautions] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  async function loadReports() {
    const value: unknown = await request("/api/reports");
    setReports(parseRevisionList(value));
  }

  useEffect(() => {
    loadReports().catch(onError);
  }, []);

  async function loadHistory(reportId: string, fallback?: ReportRevision, preferredRevision?: number) {
    const value: unknown = await request(`/api/reports/${reportId}/revisions`);
    const normalized = parseRevisionList(value);
    if (fallback && !normalized.some((item) => item.revision === fallback.revision)) normalized.push(fallback);
    normalized.sort((left, right) => right.revision - left.revision);
    setRevisions(normalized);
    const selected = normalized.find((item) => item.revision === preferredRevision) ?? normalized[0] ?? fallback ?? null;
    setSelectedRevision(selected);
    setStudyId(selected?.studyId ?? "");
    setBody(selected?.body ?? REPORT_TEMPLATE);
    setAssistantCautions([]);
  }

  async function openReport(summary: ReportRevision) {
    try {
      setBusy(true);
      setActiveReportId(summary.reportId);
      await loadHistory(summary.reportId, summary);
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  function startNewReport() {
    setActiveReportId(null);
    setRevisions([]);
    setSelectedRevision(null);
    setStudyId("");
    setBody(REPORT_TEMPLATE);
    setAssistantCautions([]);
  }

  const latestRevision = revisions[0] ?? null;
  const selectedIsLatest = Boolean(selectedRevision && latestRevision && selectedRevision.revision === latestRevision.revision);
  const editable = activeReportId === null || Boolean(selectedRevision?.status === "draft" && selectedIsLatest);

  async function save() {
    if (!studyId) return onError(new Error("请先填写检查编号"));
    if (!editable) return onError(new Error("历史版本不可编辑"));
    try {
      setBusy(true);
      const payload: Record<string, unknown> = { studyId, body };
      if (activeReportId) payload.reportId = activeReportId;
      if (selectedRevision?.reviewId) payload.reviewId = selectedRevision.reviewId;
      const next = parseRevision(await request("/api/reports/draft", "POST", payload));
      setActiveReportId(next.reportId);
      await Promise.all([loadReports(), loadHistory(next.reportId, next, next.revision)]);
      onNotice(`已保存第 ${next.revision} 版草稿`);
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  async function confirm() {
    if (!selectedRevision || !selectedIsLatest || selectedRevision.status !== "draft") return;
    try {
      setBusy(true);
      const confirmed = parseRevision(await request("/api/reports/confirm", "POST", {
        reportId: selectedRevision.reportId,
        revision: selectedRevision.revision,
      }));
      await Promise.all([loadReports(), loadHistory(confirmed.reportId, confirmed, confirmed.revision)]);
      onNotice("报告已锁定");
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  async function exportPdf() {
    if (!selectedRevision || selectedRevision.status !== "confirmed") return;
    try {
      setBusy(true);
      const result = await window.medicalApi.exportReport(
        selectedRevision.reportId,
        selectedRevision.revision,
        `${selectedRevision.studyId}-阅片报告-v${selectedRevision.revision}.pdf`,
      );
      if (result.exported) onNotice("PDF 已导出到所选位置");
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  async function organizeWithAssistant() {
    if (!studyId || !editable) return;
    try {
      setBusy(true);
      const payload: Record<string, unknown> = { studyId, clinicianText: body };
      if (selectedRevision?.reviewId) payload.reviewId = selectedRevision.reviewId;
      const result = await request("/api/assistant/report", "POST", payload);
      setBody(String(result.draft ?? body));
      setAssistantCautions(Array.isArray(result.cautions) ? result.cautions.map(String) : []);
      onNotice("助手草稿已返回，仍需医生编辑确认");
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  return <div className="view-stack">
    <div className="section-header">
      <div><p className="eyebrow">REPORTING / VERSION CONTROL</p><h2>报告中心</h2><p className="muted">只显示当前医生拥有的报告；每次保存都会保留不可覆盖的完整版本历史。</p></div>
      <div className="report-header-actions"><button className="quiet-button" onClick={startNewReport}><Plus size={16} />新建报告</button><span className="tag neutral">医生确认必需</span></div>
    </div>

    <div className="report-workspace">
      <aside className="panel report-list-panel">
        <div className="panel-title"><div><h3>我的报告</h3><p>选择报告继续编辑最新草稿或查看历史。</p></div><IconButton icon={RefreshCw} label="刷新报告列表" onClick={() => void loadReports().catch(onError)} /></div>
        {reports.length === 0 ? <div className="table-empty">尚无报告</div> : <div className="report-list">
          {reports.map((latest) => <button type="button" key={latest.reportId} className={`report-list-item ${activeReportId === latest.reportId ? "active" : ""}`} onClick={() => void openReport(latest)} disabled={busy}>
            <span><strong>{latest.studyId}</strong><small className="mono">{latest.reportId}</small></span>
            <span><b>v{latest.revision}</b><small>{latest.status === "confirmed" ? "已确认" : "草稿"}</small></span>
          </button>)}
        </div>}
      </aside>

      <section className="panel report-editor">
        <div className="panel-title"><div><h3>{activeReportId ? "报告正文" : "新建报告"}</h3><p>{selectedRevision && !selectedIsLatest ? "当前查看历史版本，内容只读。" : "助手只接收服务端按检查与复核编号解析的真实上下文。"}</p></div>{selectedRevision && <span className={`tag ${selectedRevision.status === "confirmed" ? "good" : "warn"}`}>{selectedRevision.status === "confirmed" ? `已确认 v${selectedRevision.revision}` : `草稿 v${selectedRevision.revision}`}</span>}</div>
        <label>检查编号<input value={studyId} onChange={(event) => setStudyId(event.target.value)} placeholder="例如 ST-..." disabled={Boolean(activeReportId)} /></label>
        <textarea className="report-textarea" value={body} onChange={(event) => setBody(event.target.value)} disabled={!editable} />
        {assistantCautions.length > 0 && <div className="assistant-cautions">{assistantCautions.map((item) => <span key={item}>{item}</span>)}</div>}
        <div className="editor-actions">
          <button className="quiet-button" onClick={() => void organizeWithAssistant()} disabled={busy || !editable || !studyId}><Sparkles size={16} />助手整理</button>
          <button className="quiet-button" onClick={() => void save()} disabled={busy || !editable || !studyId}><FileText size={16} />保存新版本</button>
          {selectedRevision?.status === "confirmed"
            ? <button className="primary-button" onClick={() => void exportPdf()} disabled={busy}><FileText size={16} />导出此版本 PDF</button>
            : <button className="primary-button" onClick={() => void confirm()} disabled={!selectedRevision || !selectedIsLatest || busy}><LockKeyhole size={16} />确认并锁定</button>}
        </div>
      </section>

      <aside className="panel report-aside">
        <div className="panel-title"><div><h3>完整版本历史</h3><p>选择任一版本查看；历史正文不可修改。</p></div><FileBarChart2 size={19} /></div>
        {revisions.length === 0 ? <div className="table-empty">选择或保存报告后显示版本</div> : <div className="revision-list">
          {revisions.map((revision) => <button type="button" key={revision.revision} className={`revision-item ${selectedRevision?.revision === revision.revision ? "active" : ""}`} onClick={() => { setSelectedRevision(revision); setStudyId(revision.studyId); setBody(revision.body); setAssistantCautions([]); }}>
            <span className="revision-number">v{revision.revision}</span>
            <span><strong>{revision.status === "confirmed" ? "医生已确认" : revision.revision === latestRevision?.revision ? "当前草稿" : "历史草稿"}</strong><small>{displayTime(revision.createdAt)}</small></span>
          </button>)}
        </div>}
        <div className="pdf-note"><FileText size={16} /><span>{selectedRevision?.status === "confirmed" ? "所选版本已锁定，可导出 PDF" : "PDF 导出仅对已确认版本开放"}<br /><small>正文与 AI 溯源附录一并生成</small></span></div>
      </aside>
    </div>
  </div>;
}
