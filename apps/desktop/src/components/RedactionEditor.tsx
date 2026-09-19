import { useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { RotateCcw, ShieldCheck, X } from "lucide-react";
import { request } from "../api";
import { IconButton } from "./IconButton";

export type Rectangle = { left: number; top: number; right: number; bottom: number };
type PixelPayload = { width: number; height: number; pixelsBase64: string };

function decode(value: string): Uint8ClampedArray {
  const binary = atob(value);
  const pixels = new Uint8ClampedArray(binary.length);
  for (let index = 0; index < binary.length; index += 1) pixels[index] = binary.charCodeAt(index);
  return pixels;
}

export function RedactionEditor({ stagingId, rectangles, onChange, onClose }: { stagingId: string; rectangles: Rectangle[]; onChange: (rectangles: Rectangle[]) => void; onClose: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [payload, setPayload] = useState<PixelPayload | null>(null);
  const [start, setStart] = useState<{ x: number; y: number } | null>(null);
  const [message, setMessage] = useState("正在载入本地预览…");

  useEffect(() => {
    request(`/api/import/${stagingId}/preview`).then((value) => { setPayload(value); setMessage(""); }).catch((reason) => setMessage(reason instanceof Error ? reason.message : "预览失败"));
  }, [stagingId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !payload) return;
    canvas.width = payload.width;
    canvas.height = payload.height;
    const context = canvas.getContext("2d");
    if (!context) return;
    const gray = decode(payload.pixelsBase64);
    const image = context.createImageData(payload.width, payload.height);
    for (let index = 0; index < gray.length; index += 1) {
      const offset = index * 4;
      image.data[offset] = gray[index]; image.data[offset + 1] = gray[index]; image.data[offset + 2] = gray[index]; image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    context.fillStyle = "rgba(4, 12, 15, 0.92)";
    context.strokeStyle = "#54c8b4";
    context.lineWidth = Math.max(2, payload.width / 400);
    for (const rectangle of rectangles) {
      context.fillRect(rectangle.left, rectangle.top, rectangle.right - rectangle.left, rectangle.bottom - rectangle.top);
      context.strokeRect(rectangle.left, rectangle.top, rectangle.right - rectangle.left, rectangle.bottom - rectangle.top);
    }
  }, [payload, rectangles]);

  function point(event: PointerEvent<HTMLCanvasElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return { x: Math.round((event.clientX - bounds.left) * event.currentTarget.width / bounds.width), y: Math.round((event.clientY - bounds.top) * event.currentTarget.height / bounds.height) };
  }

  return <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="烧录文字遮挡">
    <section className="redaction-modal">
      <header><div><p className="eyebrow">BURNED-IN TEXT REDACTION</p><h2>绘制不可逆遮挡区域</h2><p>在姓名、编号或其他烧录身份信息上拖拽矩形。提交后只保留派生副本。</p></div><IconButton icon={X} label="关闭" onClick={onClose} /></header>
      <div className="redaction-stage">{message && <span className="viewport-hint">{message}</span>}<canvas ref={canvasRef} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); setStart(point(event)); }} onPointerUp={(event) => { if (!start) return; const end = point(event); const rectangle = { left: Math.min(start.x, end.x), top: Math.min(start.y, end.y), right: Math.max(start.x, end.x), bottom: Math.max(start.y, end.y) }; setStart(null); if (rectangle.right - rectangle.left >= 4 && rectangle.bottom - rectangle.top >= 4) onChange([...rectangles, rectangle]); }} /></div>
      <footer><span><ShieldCheck size={15} />已定义 {rectangles.length} 个遮挡区域</span><button className="quiet-button" onClick={() => onChange([])} disabled={!rectangles.length}><RotateCcw size={15} />清除全部</button><button className="primary-button" onClick={onClose}>完成</button></footer>
    </section>
  </div>;
}
