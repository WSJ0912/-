import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const desktopRoot = path.join(projectRoot, "apps", "desktop");
const viteEntry = path.join(desktopRoot, "node_modules", "vite", "bin", "vite.js");
const electronExecutable = path.join(
  desktopRoot,
  "node_modules",
  "electron",
  "dist",
  process.platform === "win32" ? "electron.exe" : "electron",
);
const devUrl = "http://127.0.0.1:5173";
const children = new Set();
let stopping = false;

function launch(executable, args, options = {}) {
  const child = spawn(executable, args, {
    cwd: desktopRoot,
    stdio: "inherit",
    windowsHide: false,
    ...options,
  });
  children.add(child);
  child.once("error", (error) => {
    console.error(`Failed to start ${executable}: ${error.message}`);
    stop(1);
  });
  child.once("exit", () => children.delete(child));
  return child;
}

function waitForPort(port, host, timeoutMs = 30_000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = net.createConnection({ port, host });
      socket.once("connect", () => {
        socket.destroy();
        resolve();
      });
      socket.once("error", () => {
        socket.destroy();
        if (Date.now() - started >= timeoutMs) {
          reject(new Error(`Vite did not open ${host}:${port} within ${timeoutMs} ms`));
        } else {
          setTimeout(tryConnect, 150);
        }
      });
    };
    tryConnect();
  });
}

function stop(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) child.kill();
  setTimeout(() => process.exit(exitCode), 250).unref();
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => stop(0));
}

const vite = launch(process.execPath, [viteEntry, "--host", "127.0.0.1", "--port", "5173"]);
vite.once("exit", (code) => stop(code ?? 1));

try {
  await waitForPort(5173, "127.0.0.1");
  if (!fs.existsSync(electronExecutable)) {
    throw new Error("Electron binary is missing; run `npm install` in apps/desktop without --ignore-scripts");
  }
  const electron = launch(electronExecutable, [desktopRoot], {
    env: { ...process.env, VITE_DEV_SERVER_URL: devUrl },
  });
  electron.once("exit", (code) => stop(code ?? 0));
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  stop(1);
}
