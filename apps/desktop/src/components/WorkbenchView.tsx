import { useEffect, useState } from "react";
import { Activity, ClipboardCheck, Eye } from "lucide-react";
import { formatProbability, labels, request } from "../api";
import {
  type Prediction,
  type PredictionCam,
  type Study,
  validatePrediction,
  validatePredictionCam,
  validateReview,
  validateStudy,
} from "../contracts";
import { DicomViewport } from "./DicomViewport";

type ReviewMap = Record<string, "confirmed" | "denied" | "uncertain">;

type Props = {
  model: any;
  onError: (reason: unknown) => void;
  onNotice: (message: string) => void;
};

export function WorkbenchView({ model, onError, onNotice }: Props) {
  const [studies, setStudies] = useState<Study[]>([]);
  const [selected, setSelected] = useState<Study | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [review, setReview] = useState<ReviewMap>({});
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [camIndex, setCamIndex] = useState(0);
  const [camEnabled, setCamEnabled] = useState(false);
  const [camOpacity, setCamOpacity] = useState(0.45);
  const [cam, setCam] = useState<PredictionCam | null>(null);
  const [camMessage, setCamMessage] = useState("");
  const [camRefresh, setCamRefresh] = useState(0);

  useEffect(() => {
    request<unknown>("/api/studies")
      .then((value) => {
        if (!Array.isArray(value) || !value.every(validateStudy)) {
          throw new Error("检查列表未通过共享 JSON Schema 校验");
        }
        setStudies(value);
      })
      .catch(onError);
  }, []);

  useEffect(() => {
    let active = true;
    setCam(null);
    if (!prediction || !camEnabled) {
      setCamMessage("");
      return () => { active = false; };
    }
    if (!prediction.camAvailable) {
      setCamMessage("当前模型未提供经过分类器权重计算的 CAM");
      return () => { active = false; };
    }
    setCamMessage("正在读取 CAM…");
    request<unknown>(`/api/predictions/${prediction.predictionId}/cams/${camIndex}`)
      .then((value) => {
        if (!active) return;
        if (!validatePredictionCam(value)) {
          throw new Error("CAM 未通过共享 JSON Schema 校验");
        }
        if (value.label !== labels[camIndex]?.[0]) {
          throw new Error("CAM 类别与请求不一致");
        }
        setCam(value);
        setCamMessage("");
      })
      .catch((reason) => {
        if (!active) return;
        setCam(null);
        setCamMessage(reason instanceof Error ? reason.message : "该类别 CAM 不可用");
      });
    return () => { active = false; };
  }, [prediction?.predictionId, prediction?.camAvailable, camEnabled, camIndex, camRefresh]);

  async function runPrediction() {
    if (!selected) return;
    try {
      setBusy(true);
      const next: unknown = await request(`/api/studies/${selected.studyId}/predict`, "POST");
      if (!validatePrediction(next)) {
        throw new Error("预测结果未通过共享 JSON Schema 校验");
      }
      setPrediction(next);
      setReview({});
      setNotes("");
      setCamIndex(0);
      setCamEnabled(Boolean(next.camAvailable));
      setCam(null);
      setCamRefresh((value) => value + 1);
      onNotice("推理完成，原始概率已锁定");
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  async function saveReview() {
    if (!selected || !prediction) return;
    try {
      setBusy(true);
      const saved: unknown = await request("/api/reviews", "POST", {
        studyId: selected.studyId,
        predictionId: prediction.predictionId,
        decisions: review,
        notes,
      });
      if (!validateReview(saved)) {
        throw new Error("复核结果未通过共享 JSON Schema 校验");
      }
      onNotice("复核已保存");
    } catch (reason) {
      onError(reason);
    } finally {
      setBusy(false);
    }
  }

  const selectedLabel = labels[camIndex];

  return <div className="view-stack">
    <div className="section-header">
      <div>
        <p className="eyebrow">REVIEW / MULTI-LABEL OUTPUT</p>
        <h2>阅片工作区</h2>
        <p className="muted">AI 原始概率不可修改；医生复核单独记录为确认、否认或不确定。</p>
      </div>
      <div className="workbench-actions">
        <select value={selected?.studyId || ""} onChange={(event) => {
          const value = studies.find((item) => item.studyId === event.target.value) || null;
          setSelected(value);
          setPrediction(null);
          setReview({});
          setNotes("");
          setCam(null);
          setCamEnabled(false);
          setCamMessage("");
        }}>
          <option value="">选择检查</option>
          {studies.map((study) => <option key={study.studyId} value={study.studyId}>{study.studyId} · {study.viewPosition}</option>)}
        </select>
        <button className="primary-button" onClick={() => void runPrediction()} disabled={!selected || !model?.installed || busy}>
          <Activity size={17} />{busy ? "推理中…" : "运行推理"}
        </button>
      </div>
    </div>

    <div className="workbench-grid">
      <DicomViewport
        studyId={selected?.studyId}
        cam={cam}
        camVisible={camEnabled}
        camOpacity={camOpacity}
        camLabel={selectedLabel ? `${selectedLabel[1]} / ${selectedLabel[0]}` : undefined}
        camMessage={camMessage}
      />
      <aside className="result-panel">
        <div className="result-header">
          <div><p className="eyebrow">AI SCREENING</p><h3>{prediction ? `${prediction.modelId} / ${prediction.modelVersion}` : "尚未运行"}</h3></div>
          {prediction ? <span className="tag good">ONNX 原始结果</span> : <span className="tag neutral">--</span>}
        </div>
        {!model?.installed && <div className="inline-warning"><Activity size={16} /><span>尚未安装模型。请先导入已验证的 `.medmodel`，系统不会生成模拟结果。</span></div>}
        {prediction && <div className="cam-controls">
          <div className="cam-controls-title"><Eye size={15} /><span>分类器 CAM</span></div>
          <label className="cam-switch">
            <input
              type="checkbox"
              role="switch"
              checked={camEnabled}
              disabled={!prediction.camAvailable}
              onChange={(event) => setCamEnabled(event.target.checked)}
            />
            显示热图
          </label>
          <select aria-label="CAM 类别" value={camIndex} disabled={!prediction.camAvailable} onChange={(event) => setCamIndex(Number(event.target.value))}>
            {labels.map(([key, zh], index) => <option key={key} value={index}>{zh} / {key}</option>)}
          </select>
          <label className="cam-opacity">
            <span>透明度</span>
            <input
              type="range"
              min="10"
              max="90"
              step="5"
              value={Math.round(camOpacity * 100)}
              disabled={!camEnabled || !prediction.camAvailable}
              onChange={(event) => setCamOpacity(Number(event.target.value) / 100)}
            />
            <output>{Math.round(camOpacity * 100)}%</output>
          </label>
          {!prediction.camAvailable && <span className="cam-unavailable">模型未提供 CAM</span>}
        </div>}
        {prediction ? <div className="observation-list">
          {labels.map(([key, zh]) => <div className="observation" key={key}>
            <div className="observation-label"><span>{zh}</span><small>{key}</small></div>
            <strong>{formatProbability(prediction.probabilities[key])}</strong>
            <div className="probability-track"><i style={{ width: `${Math.max(0, Math.min(100, Number(prediction.probabilities[key] || 0) * 100))}%` }} /></div>
            <div className="review-controls">
              {(["confirmed", "denied", "uncertain"] as const).map((choice) => <button type="button" className={review[key] === choice ? "selected" : ""} key={choice} onClick={() => setReview((current) => ({ ...current, [key]: choice }))}>{choice === "confirmed" ? "确认" : choice === "denied" ? "否认" : "不确定"}</button>)}
            </div>
          </div>)}
        </div> : <div className="result-empty"><ClipboardCheck size={22} /><p>选择检查并运行已安装模型</p><small>所有未知概率显示为 --</small></div>}
        {prediction && <div className="review-footer"><textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="复核备注（可选）" /><button className="quiet-button" onClick={() => void saveReview()} disabled={busy}><ClipboardCheck size={16} />保存复核</button></div>}
      </aside>
    </div>
  </div>;
}
