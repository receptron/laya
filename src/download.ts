/**
 * Fetch the ONNX bundle (graph, weights, calibration config, tokenizer) from a Hugging Face repo into a
 * local cache. Files are only downloaded when missing or when their size differs from the remote one,
 * and each one is written to a temp file and renamed, so an interrupted download never leaves a
 * half-written weight file behind.
 */
import { createWriteStream } from "node:fs";
import { mkdir, rename, stat, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

/** Files that make up one exported checkpoint, relative to the bundle directory. */
export const BUNDLE_FILES = ["laya.onnx", "laya.onnx.data", "laya_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] as const;

/** Where the exported ONNX bundle is published. */
export const DEFAULT_REPO = "receptron/laya-onnx";

export interface DownloadOptions {
  /** Hugging Face repo id (default: receptron/laya-onnx) */
  repo?: string;
  /** git revision in that repo (default: main) */
  revision?: string;
  /** subfolder inside the repo, e.g. "multilingual" (default: repo root = English checkpoint) */
  subfolder?: string;
  /** local cache root (default: $LAYA_CACHE or ~/.cache/receptron-laya) */
  cacheDir?: string;
  /** Hugging Face token, for private repos (default: $HF_TOKEN) */
  token?: string;
  /** called as bytes arrive, per file */
  onProgress?: (info: { file: string; received: number; total: number | null }) => void;
}

export function defaultCacheDir(): string {
  return process.env.LAYA_CACHE ?? path.join(process.env.XDG_CACHE_HOME ?? path.join(homedir(), ".cache"), "receptron-laya");
}

async function fileSize(p: string): Promise<number | null> {
  try {
    return (await stat(p)).size;
  } catch {
    return null;
  }
}

/** Ensure the bundle is on disk; returns the directory that holds it. */
export async function ensureBundle(opts: DownloadOptions = {}): Promise<string> {
  const repo = opts.repo ?? DEFAULT_REPO;
  const revision = opts.revision ?? "main";
  const sub = opts.subfolder ? opts.subfolder.replace(/^\/+|\/+$/g, "") + "/" : "";
  const dir = path.join(opts.cacheDir ?? defaultCacheDir(), repo.replace("/", "--"), revision, sub);
  const token = opts.token ?? process.env.HF_TOKEN;
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};

  for (const file of BUNDLE_FILES) {
    const dest = path.join(dir, file);
    const url = `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${sub}${file}`;
    const local = await fileSize(dest);
    if (local !== null) {
      // cheap freshness check: same byte count as the remote file. A HEAD that fails (offline, DNS, 5xx)
      // is treated as "unknown" so a populated cache keeps working without a network.
      const head = await fetch(url, { method: "HEAD", headers, redirect: "follow" }).catch(() => null);
      const remote = head?.ok ? Number(head.headers.get("x-linked-size") ?? head.headers.get("content-length")) : NaN;
      if (!Number.isFinite(remote) || remote === local) continue;
    }
    const res = await fetch(url, { headers, redirect: "follow" });
    if (!res.ok || !res.body) {
      throw new Error(`failed to download ${url}: ${res.status} ${res.statusText}`);
    }
    const total = Number(res.headers.get("content-length"));
    await mkdir(path.dirname(dest), { recursive: true });
    const tmp = `${dest}.part-${process.pid}`;
    let received = 0;
    const progress = opts.onProgress;
    const body = Readable.fromWeb(res.body as import("node:stream/web").ReadableStream);
    if (progress) {
      body.on("data", (chunk: Buffer) => {
        received += chunk.length;
        progress({ file, received, total: Number.isFinite(total) ? total : null });
      });
    }
    try {
      await pipeline(body, createWriteStream(tmp));
      await rename(tmp, dest);
    } catch (e) {
      await unlink(tmp).catch(() => undefined);
      throw e;
    }
  }
  return dir;
}
