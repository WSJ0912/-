import { useCallback, useEffect, useRef, useState } from "react";
import { RotateCcw, SunMedium, ZoomIn, ZoomOut } from "lucide-react";
import { request } from "../api";
import { IconButton } from "./IconButton";

type PixelPayload = { width: number; height: number; pixelsBase64: string; min: number; max: number };
type CamPayload = { width: number; height: number; pixelsBase64: string };

type Props = {
  studyId?: string;
  cam?: CamPayload | null;
  camVisible?: boolean;
  camOpacity?: number;
  camLabel?: string;
  camMessage?: string;
};

const localImages = new Map<string, PixelPayload>();
let cornerstoneReady: Promise<typeof import("@cornerstonejs/core")> | null = null;

function bytesFromBase64(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function heatColor(value: number): [number, number, number] {
  const normalized = value / 255;
  const red = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * normalized - 3)));
  const green = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * normalized - 2)));
  const blue = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * normalized - 1)));
  return [Math.round(red * 255), Math.round(green * 255), Math.round(blue * 255)];
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

export function DicomViewport({ studyId, cam = null, camVisible = false, camOpacity = 0.45, camLabel, camMessage }: Props) {
  const elementRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const viewportRef = useRef<any>(null);
  const engineRef = useRef<any>(null);
  const idRef = useRef(`viewport-${crypto.randomUUID()}`);
  const drawOverlayRef = useRef<() => void>(() => undefined);
  const [message, setMessage] = useState("");
  const [camRenderMessage, setCamRenderMessage] = useState("");

  const drawCamOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    const element = elementRef.current;
    if (!canvas || !element) return;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, element.clientWidth);
    const height = Math.max(1, element.clientHeight);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!camVisible || !cam || !viewportRef.current) {
      setCamRenderMessage("");
      return;
    }
    try {
      const values = bytesFromBase64(cam.pixelsBase64);
      if (values.length !== cam.width * cam.height) throw new Error("CAM 像素数量与尺寸不一致");
      const viewport = viewportRef.current;
      const viewportImageData = viewport.getImageData();
      const imageData = viewportImageData?.imageData;
      const dimensions = viewportImageData?.dimensions;
      if (!imageData?.indexToWorld || !dimensions) throw new Error("影像坐标尚未就绪");

      const origin = viewport.worldToCanvas(imageData.indexToWorld([0, 0, 0]));
      const xEdge = viewport.worldToCanvas(imageData.indexToWorld([dimensions[0], 0, 0]));
      const yEdge = viewport.worldToCanvas(imageData.indexToWorld([0, dimensions[1], 0]));
      const source = document.createElement("canvas");
      source.width = cam.width;
      source.height = cam.height;
      const sourceContext = source.getContext("2d");
      if (!sourceContext) throw new Error("无法创建 CAM 图层");
      const colored = sourceContext.createImageData(cam.width, cam.height);
      for (let index = 0; index < values.length; index += 1) {
        const [red, green, blue] = heatColor(values[index]);
        const offset = index * 4;
        colored.data[offset] = red;
        colored.data[offset + 1] = green;
        colored.data[offset + 2] = blue;
        colored.data[offset + 3] = 255;
      }
      sourceContext.putImageData(colored, 0, 0);

      context.globalAlpha = Math.max(0.1, Math.min(0.9, camOpacity));
      context.imageSmoothingEnabled = true;
      context.setTransform(
        ratio * (xEdge[0] - origin[0]) / cam.width,
        ratio * (xEdge[1] - origin[1]) / cam.width,
        ratio * (yEdge[0] - origin[0]) / cam.height,
        ratio * (yEdge[1] - origin[1]) / cam.height,
        ratio * origin[0],
        ratio * origin[1],
      );
      context.drawImage(source, 0, 0);
      context.globalAlpha = 1;
      setCamRenderMessage("");
    } catch (reason) {
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);
      setCamRenderMessage(reason instanceof Error ? reason.message : "CAM 无法显示");
    }
  }, [cam, camOpacity, camVisible]);

  useEffect(() => {
    drawOverlayRef.current = drawCamOverlay;
    drawCamOverlay();
  }, [drawCamOverlay]);

  useEffect(() => {
    let active = true;
    if (!studyId || !elementRef.current) return;
    const element = elementRef.current;
    const redraw = () => window.requestAnimationFrame(() => drawOverlayRef.current());
    const observer = new ResizeObserver(redraw);
    observer.observe(element);
    element.addEventListener("CORNERSTONE_IMAGE_RENDERED", redraw);
    element.addEventListener("CORNERSTONE_CAMERA_MODIFIED", redraw);
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
        redraw();
      } catch (reason) {
        setMessage(reason instanceof Error ? reason.message : "影像加载失败");
      }
    })();
    return () => {
      active = false;
      observer.disconnect();
      element.removeEventListener("CORNERSTONE_IMAGE_RENDERED", redraw);
      element.removeEventListener("CORNERSTONE_CAMERA_MODIFIED", redraw);
      viewportRef.current = null;
      if (engineRef.current) {
        engineRef.current.destroy();
        engineRef.current = null;
      }
      localImages.delete(`medlocal:${studyId}`);
      const overlay = overlayRef.current;
      overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
    };
  }, [studyId]);

  function zoom(factor: number) {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.setZoom(Math.max(0.2, Math.min(8, viewport.getZoom() * factor)));
    viewport.render();
    window.requestAnimationFrame(() => drawOverlayRef.current());
  }

  function reset() {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.resetCamera();
    viewport.setProperties({ voiRange: { lower: 0, upper: 255 } });
    viewport.render();
    window.requestAnimationFrame(() => drawOverlayRef.current());
  }

  function adjustWindow() {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.setProperties({ voiRange: { lower: 30, upper: 225 } });
    viewport.render();
    window.requestAnimationFrame(() => drawOverlayRef.current());
  }

  return <div className="viewport-shell">
    <div className="viewport-toolbar" aria-label="阅片工具">
      <IconButton icon={ZoomIn} label="放大" onClick={() => zoom(1.2)} disabled={!studyId} />
      <IconButton icon={ZoomOut} label="缩小" onClick={() => zoom(1 / 1.2)} disabled={!studyId} />
      <IconButton icon={SunMedium} label="调整窗宽窗位" onClick={adjustWindow} disabled={!studyId} />
      <IconButton icon={RotateCcw} label="重置视图" onClick={reset} disabled={!studyId} />
    </div>
    <div className="dicom-viewport" ref={elementRef} />
    <canvas className="cam-overlay" ref={overlayRef} aria-hidden="true" />
    {!studyId && <div className="empty-view viewport-empty"><span className="crosshair" /><p>选择一项已匿名化的检查</p><small>仅支持成人正位 AP/PA 胸片</small></div>}
    {message && <span className="viewport-hint viewport-message">{message}</span>}
    {camVisible && (camMessage || camRenderMessage) && <div className="cam-status">{camMessage || camRenderMessage}</div>}
    {camVisible && cam && !camMessage && !camRenderMessage && <div className="cam-legend"><span>{camLabel || "CAM"}</span><i /><small>低</small><b>高</b></div>}
  </div>;
}
