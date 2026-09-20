import { useEffect, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { Activity, ArrowUpRight, ClipboardCheck, FileImage, FileText, FlaskConical, FolderOpen, KeyRound, LogIn, LockKeyhole, RefreshCw, ScanLine, Settings2, ShieldCheck, Upload, UserPlus, X } from "lucide-react";
import { ExperimentsView } from "./components/ExperimentsView";
import { IconButton } from "./components/IconButton";
import { RedactionEditor } from "./components/RedactionEditor";
import type { Rectangle } from "./components/RedactionEditor";
import { ReportsView } from "./components/ReportsView";
import { StatPill } from "./components/StatPill";
import { WorkbenchView } from "./components/WorkbenchView";
import { type Study, validateStudy } from "./contracts";
import { NavKey, request, Role } from "./api";

type BootState = { connected: boolean; requiresSetup: boolean; model: any; message?: string };

type NavItem = { id: NavKey; label: string; icon: typeof ScanLine; hint: string };

const navCatalog: Record<NavKey, NavItem> = {
  queue: { id: "queue", label: "检查队列", icon: FolderOpen, hint: "导入与准入" },
  workbench: { id: "workbench", label: "阅片工作区", icon: ScanLine, hint: "AI 结果与复核" },
  reports: { id: "reports", label: "报告中心", icon: FileText, hint: "版本与导出" },
  experiments: { id: "experiments", label: "实验结果", icon: FlaskConical, hint: "聚合指标" },
  models: { id: "models", label: "模型管理", icon: Activity, hint: "清单与校验" },
  admin: { id: "admin", label: "管理员设置", icon: Settings2, hint: "账号与助手" },
};

const roleNavigation: Record<Role, readonly NavKey[]> = {
  admin: ["queue", "experiments", "models", "admin"],
  doctor: ["queue", "workbench", "reports", "experiments"],
};

function App() {
  const [boot, setBoot] = useState<BootState>({ connected: false, requiresSetup: false, model: null });
  const [role, setRole] = useState<Role | null>(null);
  const [userId, setUserId] = useState("");
  const [nav, setNav] = useState<NavKey>("queue");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        if (!window.medicalApi) {
          throw new Error("当前是浏览器预览；请从 Electron 桌面应用启动完整本地服务");
        }
        let status = await window.medicalApi.appStatus();
        for (let attempt = 0; !status.service && !status.serviceError && attempt < 40; attempt += 1) {
          await new Promise((resolve) => window.setTimeout(resolve, 250));
          status = await window.medicalApi.appStatus();
        }
        if (!status.service) throw new Error(status.serviceError || "本地服务尚未启动");
        const remote = await request("/api/status");
        if (mounted) setBoot({ connected: true, requiresSetup: remote.requiresSetup, model: remote.model });
      } catch (reason) {
        if (mounted) setBoot({ connected: false, requiresSetup: false, model: { installed: false, ready: false, message: "尚未安装模型" }, message: reason instanceof Error ? reason.message : "本地服务未连接" });
      }
    })();
    return () => { mounted = false; };
  }, [refresh]);

  const flash = (message: string) => { setNotice(message); window.setTimeout(() => setNotice(""), 3200); };
  const fail = (reason: unknown) => setError(reason instanceof Error ? reason.message : "操作失败");

  if (!boot.connected) return <ConnectionScreen message={boot.message} onRetry={() => setRefresh((value) => value + 1)} />;
  if (!role) return <AuthScreen requiresSetup={boot.requiresSetup} onAuthenticated={(nextRole, nextUser) => { setRole(nextRole); setUserId(nextUser); setNav("queue"); setBoot((current) => ({ ...current, requiresSetup: false })); }} onError={fail} />;

  const navItems = roleNavigation[role].map((key) => navCatalog[key]);

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><ScanLine size={19} /></div><div><strong>影像识别</strong><span>RESEARCH CONSOLE</span></div></div>
      <div className="scope-label">工作区</div>
      <nav aria-label="主导航">{navItems.map(({ id, label, icon: Icon, hint }) => <button key={id} className={`nav-item ${nav === id ? "active" : ""}`} onClick={() => setNav(id)}><Icon size={18} /><span><b>{label}</b><small>{hint}</small></span>{id === "queue" && <i className="nav-count">0</i>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="privacy-note"><ShieldCheck size={16} /><span>本地优先<br /><small>影像不会离开此设备</small></span></div><div className="user-chip"><div className="avatar">{role === "admin" ? "A" : "D"}</div><div><b>{role === "admin" ? "管理员" : "医生"}</b><small>{userId || "当前会话"}</small></div><IconButton icon={LockKeyhole} label="结束会话" onClick={() => { void window.medicalApi.logout(); setRole(null); setUserId(""); setNav("queue"); }} /></div></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><div><p className="eyebrow">CHEST RADIOGRAPHY / V0.1</p><h1>{navItems.find((item) => item.id === nav)?.label}</h1></div><div className="top-actions"><StatPill label="本地服务" value="已连接" tone="good" /><StatPill label="模型" value={boot.model?.installed ? `${boot.model.version || "已安装"}` : "尚未安装"} tone={boot.model?.installed ? "good" : "warn"} /><IconButton icon={RefreshCw} label="刷新状态" onClick={() => setRefresh((value) => value + 1)} /></div></header>
      {error && <div className="alert error"><X size={17} /><span>{error}</span><button onClick={() => setError("")} aria-label="关闭错误"><X size={15} /></button></div>}
      {notice && <div className="alert success"><ClipboardCheck size={17} /><span>{notice}</span></div>}
      <div className="content-area">{nav === "queue" && <QueueView onError={fail} onNotice={flash} onOpenWorkbench={role === "doctor" ? () => setNav("workbench") : undefined} />}{nav === "workbench" && <WorkbenchView model={boot.model} onError={fail} onNotice={flash} />}{nav === "reports" && <ReportsView onError={fail} onNotice={flash} />}{nav === "experiments" && <ExperimentsView onError={fail} onNotice={flash} />}{nav === "models" && <ModelsView model={boot.model} role={role} onError={fail} onNotice={flash} onInstalled={() => setRefresh((value) => value + 1)} />}{nav === "admin" && <AdminView onError={fail} onNotice={flash} />}</div>
    </main>
  </div>;
}

function ConnectionScreen({ message, onRetry }: { message?: string; onRetry: () => void }) {
  return <div className="center-screen"><div className="connection-panel"><div className="brand-mark large"><Activity size={25} /></div><p className="eyebrow">LOCAL SERVICE</p><h1>等待本地推理服务</h1><p className="muted">桌面端只通过受限 IPC 连接到 127.0.0.1。服务未启动时不会展示任何模拟影像或预测。</p><div className="status-line"><span className="status-dot warn" />{message || "正在检查服务状态"}</div><button className="primary-button" onClick={onRetry}><RefreshCw size={17} />重新连接</button></div></div>;
}

function AuthScreen({ requiresSetup, onAuthenticated, onError }: { requiresSetup: boolean; onAuthenticated: (role: Role, userId: string) => void; onError: (reason: unknown) => void }) {
  const [username, setUsername] = useState(""); const [password, setPassword] = useState(""); const [busy, setBusy] = useState(false);
  async function submit(event: React.FormEvent) { event.preventDefault(); setBusy(true); try { const result = await request(requiresSetup ? "/api/setup" : "/api/login", "POST", { username, password }); onAuthenticated(result.role, result.userId); } catch (reason) { onError(reason); } finally { setBusy(false); } }
  return <div className="center-screen"><div className="auth-panel"><div className="auth-header"><div className="brand-mark"><ScanLine size={19} /></div><div><p className="eyebrow">MEDICAL IMAGING PLATFORM</p><h1>{requiresSetup ? "创建首个管理员" : "进入本地工作区"}</h1></div></div><p className="muted">帐号与数据库保存在本机。密码使用 Argon2id 处理，管理员不能代替医生确认报告。</p><form onSubmit={submit} className="form-stack"><label>用户名<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" required minLength={3} /></label><label>密码<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={requiresSetup ? "new-password" : "current-password"} required minLength={12} /></label><button className="primary-button wide" disabled={busy}>{requiresSetup ? <UserPlus size={17} /> : <LogIn size={17} />}{busy ? "处理中…" : requiresSetup ? "创建管理员并继续" : "登录"}</button></form><div className="auth-foot"><ShieldCheck size={15} />离线核心功能可用 · 无模型时不会生成结果</div></div></div>;
}

function SectionHeader({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) { return <div className="section-header"><div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2><p className="muted">{description}</p></div>{action}</div>; }

function QueueView({ onError, onNotice, onOpenWorkbench }: { onError: (reason: unknown) => void; onNotice: (message: string) => void; onOpenWorkbench?: () => void }) {
  const [staged, setStaged] = useState<any[]>([]);
  const [studies, setStudies] = useState<Study[]>([]);
  const [redactions, setRedactions] = useState<Record<string, Rectangle[]>>({});
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const reload = async () => {
    try {
      const value: unknown = await request("/api/studies");
      if (!Array.isArray(value) || !value.every(validateStudy)) {
        throw new Error("检查列表未通过共享 JSON Schema 校验");
      }
      setStudies(value);
    } catch (reason) {
      onError(reason);
    }
  };
  useEffect(() => { void reload(); }, []);
  async function importFiles() { try { setBusy(true); const imported = await window.medicalApi.stageSelectedImages(); if (!imported.length) return; setStaged(imported); onNotice("文件已进入临时区，请完成准入确认"); } catch (reason) { onError(reason); } finally { setBusy(false); } }
  async function commit(item: any, event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    try {
      setBusy(true);
      await request(`/api/import/${item.stagingId}/commit`, "POST", {
        adultConfirmed: data.get("adult") === "on",
        chestConfirmed: data.get("chest") === "on",
        viewConfirmed: data.get("view") === "on",
        viewPosition: data.get("position") || null,
        burnedInReviewed: data.get("burned") === "on",
        rectangles: redactions[item.stagingId] || [],
      });
      setStaged((current) => current.filter((entry) => entry.stagingId !== item.stagingId));
      setRedactions((current) => { const next = { ...current }; delete next[item.stagingId]; return next; });
      await reload();
      onNotice("检查已匿名化并入库");
    } catch (reason) { onError(reason); } finally { setBusy(false); }
  }
  async function cancel(stagingId: string) {
    try {
      await request(`/api/import/${stagingId}`, "DELETE");
      setStaged((current) => current.filter((entry) => entry.stagingId !== stagingId));
      onNotice("临时原件已删除");
    } catch (reason) { onError(reason); }
  }
  return <div className="view-stack">
    <SectionHeader eyebrow="INTAKE / DE-IDENTIFICATION" title="检查队列" description="影像先进入临时区，完成胸片、成人、正位和烧录文字确认后才会写入本地库。" action={<button className="primary-button" onClick={importFiles} disabled={busy}><Upload size={17} />批量导入</button>} />
    <div className="queue-summary"><div><span className="summary-number">{studies.length}</span><span>已入库检查</span></div><div><span className="summary-number">{staged.length}</span><span>待确认文件</span></div><div><span className="summary-number">--</span><span>今日推理</span></div>{onOpenWorkbench && <button className="quiet-button" onClick={onOpenWorkbench}><ScanLine size={16} />打开阅片工作区<ArrowUpRight size={15} /></button>}</div>
    {staged.length > 0 && <section className="panel"><div className="panel-title"><div><h3>临时区准入</h3><p>未通过检查的文件不会生成检查编号。</p></div><span className="tag warn">需要人工确认</span></div><div className="staged-list">{staged.map((item, index) => item.stagingId ? <form className="staged-row" key={item.stagingId} onSubmit={(event) => void commit(item, event)}><div className="file-icon"><FileImage size={18} /></div><div className="file-name"><strong>{item.sourceName}</strong><small>{item.validation?.file_type || "未知格式"} · {item.stagingId}</small></div><div className="checks"><label><input name="chest" type="checkbox" />胸片</label><label><input name="adult" type="checkbox" />成人</label><label><input name="view" type="checkbox" />AP/PA</label><label><input name="burned" type="checkbox" />已检查烧录文字</label><select name="position" defaultValue=""><option value="">体位</option><option value="AP">AP</option><option value="PA">PA</option></select><button className="redaction-button" type="button" onClick={() => setEditing(item.stagingId)}>遮挡 {redactions[item.stagingId]?.length || 0}</button></div><div className="row-actions"><button className="text-button danger" type="button" onClick={() => void cancel(item.stagingId)}>取消</button><button className="small-button" type="submit" disabled={busy}>提交匿名化</button></div></form> : <div className="staged-row rejected" key={`${item.sourceName}-${index}`}><div className="file-icon"><X size={18} /></div><div className="file-name"><strong>{item.sourceName}</strong><small>{(item.reasons || ["无法暂存"]).join("、")}</small></div></div>)}</div></section>}
    {staged.length === 0 && <div className="empty-panel"><div className="empty-icon"><Upload size={21} /></div><h3>暂无待处理文件</h3><p>支持 DICOM、PNG、JPEG；PNG/JPEG 需要人工确认胸片、成人和 AP/PA 体位。</p><button className="quiet-button" onClick={importFiles}><FolderOpen size={16} />选择影像文件</button></div>}
    <section className="panel"><div className="panel-title"><div><h3>最近检查</h3><p>默认只显示随机检查编号，不保存姓名和病历号。</p></div><IconButton icon={RefreshCw} label="刷新检查列表" onClick={() => void reload()} /></div>{studies.length === 0 ? <div className="table-empty">尚无已通过匿名化准入的检查</div> : <div className="table"><div className="table-head"><span>检查编号</span><span>体位</span><span>状态</span><span>时间</span></div>{studies.map((study) => <div className="table-row" key={study.studyId}><span className="mono">{study.studyId}</span><span>{study.viewPosition}</span><span><span className="tag good">{study.status === "ready" ? "待阅片" : study.status}</span></span><span className="muted">{new Date(study.createdAt).toLocaleString("zh-CN")}</span></div>)}</div>}</section>
    {editing && <RedactionEditor stagingId={editing} rectangles={redactions[editing] || []} onChange={(items) => setRedactions((current) => ({ ...current, [editing]: items }))} onClose={() => setEditing(null)} />}
  </div>;
}

function ModelsView({ model, role, onError, onNotice, onInstalled }: { model: any; role: Role; onError: (reason: unknown) => void; onNotice: (message: string) => void; onInstalled: () => void }) { const [selection, setSelection] = useState<{ selectionId: string; name: string } | null>(null); async function select() { try { setSelection(await window.medicalApi.selectModelPackage()); } catch (reason) { onError(reason); } } async function install() { if (!selection) return; try { await window.medicalApi.installSelectedModel(selection.selectionId); setSelection(null); onInstalled(); onNotice("模型包已通过清单与 SHA-256 校验并激活"); } catch (reason) { onError(reason); } } return <div className="view-stack"><SectionHeader eyebrow="MODEL GOVERNANCE" title="模型管理" description="只接受包含完整 14 项输出、预处理定义、阈值、许可和文件哈希的 `.medmodel`。" action={role === "admin" && <button className="primary-button" onClick={select}><FolderOpen size={17} />选择模型包</button>} /><div className="model-hero"><div className="model-symbol"><Activity size={25} /></div><div className="model-copy"><p className="eyebrow">ACTIVE DEPLOYMENT</p><h2>{model?.installed ? model.modelId : "尚未安装模型"}</h2><p>{model?.installed ? `版本 ${model.version} · ${model.ready ? "CPU 推理就绪" : "运行时不可用"}` : "无模型时禁止生成任何预测；请由管理员手动导入许可明确的 .medmodel。"}</p>{selection && <div className="selected-file"><FileText size={15} />{selection.name}<button onClick={() => setSelection(null)} aria-label="清除选择"><X size={15} /></button></div>}{selection && role === "admin" && <button className="quiet-button" onClick={install}>验证并激活</button>}</div><div className={`model-state ${model?.installed ? "ready" : "missing"}`}><span className="status-dot" />{model?.installed ? "已安装" : "缺少模型"}</div></div><div className="panel"><div className="panel-title"><div><h3>部署约束</h3><p>不会自动下载未确认许可的权重。</p></div><ShieldCheck size={19} /></div><div className="constraint-grid"><div><b>输入</b><span>320 × 320 / 单通道</span></div><div><b>输出</b><span>CheXpert 14 项 logits</span></div><div><b>数据范围</b><span>成人正位 AP / PA</span></div><div><b>解释</b><span>分类器 CAM（模型提供时）</span></div></div></div></div>; }

function AdminView({ onError, onNotice }: { onError: (reason: unknown) => void; onNotice: (message: string) => void }) { const [username, setUsername] = useState(""); const [password, setPassword] = useState(""); const [users, setUsers] = useState<any[]>([]); const [key, setKey] = useState(""); async function reload() { try { setUsers(await request("/api/users")); const status = await window.medicalApi.assistantKeyStatus(); setKey(status.configured ? "已配置（已加密）" : "未配置"); } catch (reason) { onError(reason); } } useEffect(() => { void reload(); }, []); async function createDoctor(event: React.FormEvent) { event.preventDefault(); try { await request("/api/users/doctors", "POST", { username, password, role: "doctor" }); setUsername(""); setPassword(""); await reload(); onNotice("医生账号已创建"); } catch (reason) { onError(reason); } } async function saveKey() { const value = window.prompt("输入 OpenAI API 密钥；将由系统 safeStorage 加密，不会进入渲染进程或 Git"); if (!value) return; try { await window.medicalApi.saveAssistantKey(value); setKey("已配置（已加密）"); onNotice("密钥已由系统安全存储加密"); } catch (reason) { onError(reason); } } return <div className="view-stack"><SectionHeader eyebrow="ADMINISTRATION / LOCAL ONLY" title="管理员设置" description="管理员管理账号、模型和助手配置，但不能代替医生确认报告。" /><div className="admin-grid"><section className="panel"><div className="panel-title"><div><h3>创建医生账号</h3><p>密码至少 12 位，服务端使用 Argon2id。</p></div><UserPlus size={19} /></div><form className="form-stack" onSubmit={createDoctor}><label>用户名<input value={username} onChange={(event) => setUsername(event.target.value)} required minLength={3} /></label><label>初始密码<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} required minLength={12} /></label><button className="quiet-button"><UserPlus size={16} />创建账号</button></form><div className="user-list">{users.map((user) => <div className="user-row" key={user.user_id}><div className="avatar small">{user.role === "admin" ? "A" : "D"}</div><span>{user.username}</span><span className="tag neutral">{user.role}</span></div>)}</div></section><section className="panel"><div className="panel-title"><div><h3>阅片助手</h3><p>只接收去标识化结构化结果和医生文字。</p></div><KeyRound size={19} /></div><div className="assistant-status"><span className="status-dot" />{key || "检查中…"}</div><button className="quiet-button" onClick={saveKey}><KeyRound size={16} />配置加密密钥</button><ul className="guard-list"><li>默认模型：gpt-4o-mini</li><li>Responses API：store=false</li><li>无工具调用、不能修改系统状态</li><li>无网络时使用离线模板，不阻塞报告流程</li></ul></section></div></div>; }

export default App;
