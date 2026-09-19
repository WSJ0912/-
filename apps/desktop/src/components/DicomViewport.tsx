import { useEffect, useRef, useState } from "react";
import { RotateCcw, SunMedium, ZoomIn, ZoomOut } from "lucide-react";
import { request } from "../api";
import { IconButton } from "./IconButton";

type PixelPayload = { width: number; height: number; pixelsBase64: string; min: number; max: number };

const localImages = new Map<string, PixelPayload>();
let cornerstoneReady: Promise<typeof import("@cornerstonejs/core")> | null = null;

function bytesFromBase64(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

async function cornerstone() {
  if (!cornerstoneReady) {
    cornerstoneReady = (async () => {
      const core = await import("@cornerstonejs/core");
      await core.init();
      core.imageLoader.registerImageLoader("medlocal", ((imageId: string) => {
        const payload = localImages.get(imageId);
        if (!payload) return { promise: Promise.reject(new Error("local image payload is unavailable")) };
        const pixels = bytesFromBase64(payload.pixelsBase64);
        const image = {
          imageId,
          minPixelValue: payload.min,
          maxPixelValue: payload.max,
          slope: 1,
          intercept: 0,
          windowCenter: 127.5,
          windowWidth: 255,
          getPixelData: () => pixels,
          rows: payload.height,
          columns: payload.width,
          height: payload.height,
          width: payload.width,
          color: false,
          rgba: false,
          columnPixelSpacing: 1,
          rowPixelSpacing: 1,
          invert: false,
          sizeInBytes: pixels.byteLength,
        };
        return { promise: Promise.resolve(image) };
      }) as any);
      return core;
    })();
  }
  return cornerstoneReady;
}

export function DicomViewport({ studyId }: { studyId?: string }) {
  const elementRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<any>(null);
  const engineRef = useRef<any>(null);
  const idRef = useRef(`viewport-${crypto.randomUUID()}`);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let active = true;
    if (!studyId || !elementRef.current) return;
    setMessage("正在载入去标识化影像…");
    (async () => {
      try {
        const payload = await request(`/api/studies/${studyId}/preview`) as PixelPayload;
        const core = await cornerstone();
        if (!active || !elementRef.current) return;
        const imageId = `medlocal:${studyId}`;
        localImages.set(imageId, payload);
        const renderingEngine = new core.RenderingEngine(`engine-${idRef.current}`);
        renderingEngine.enableElement({ viewportId: idRef.current, element: elementRef.current, type: core.Enums.ViewportType.STACK });
        const viewport = renderingEngine.getViewport(idRef.current) as any;
        await viewport.setStack([imageId]);
        viewport.setProperties({ voiRange: { lower: 0, upper: 255 } });
        viewport.render();
        engineRef.current = renderingEngine;
        viewportRef.current = viewport;
        setMessage("");
      } catch (reason) {
        setMessage(reason instanceof Error ? reason.message : "影像加载失败");
      }
    })();
    return () => {
      active = false;
      viewportRef.current = null;
      if (engineRef.current) {
        engineRef.current.destroy();
        engineRef.current = null;
      }
      localImages.delete(`medlocal:${studyId}`);
    };
  }, [studyId]);

  function zoom(factor: number) {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.setZoom(Math.max(0.2, Math.min(8, viewport.getZoom() * factor)));
    viewport.render();
  }

  function reset() {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.resetCamera();
    viewport.setProperties({ voiRange: { lower: 0, upper: 255 } });
    viewport.render();
  }

  function adjustWindow() {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.setProperties({ voiRange: { lower: 30, upper: 225 } });
    viewport.render();
  }

  return <div className="viewport-shell">
    <div className="viewport-toolbar" aria-label="阅片工具">
      <IconButton icon={ZoomIn} label="放大" onClick={() => zoom(1.2)} disabled={!studyId} />
      <IconButton icon={ZoomOut} label="缩小" onClick={() => zoom(1 / 1.2)} disabled={!studyId} />
      <IconButton icon={SunMedium} label="调整窗宽窗位" onClick={adjustWindow} disabled={!studyId} />
      <IconButton icon={RotateCcw} label="重置视图" onClick={reset} disabled={!studyId} />
    </div>
    <div className="dicom-viewport" ref={elementRef}>{!studyId && <div className="empty-view"><span className="crosshair" /><p>选择一项已匿名化的检查</p><small>仅支持成人正位 AP/PA 胸片</small></div>}{message && <span className="viewport-hint">{message}</span>}</div>
  </div>;
}
