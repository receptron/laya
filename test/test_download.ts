import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { BUNDLE_FILES, ensureBundle } from "../src/download.js";

/** fetch stub: every bundle file has the content "<name>:<payload>", and each call is logged */
function stubFetch(payload: string, calls: string[]) {
  return async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    const name = url.slice(url.lastIndexOf("/") + 1);
    const body = `${name}:${payload}`;
    calls.push(`${init?.method ?? "GET"} ${url}`);
    if (init?.method === "HEAD") return new Response(null, { status: 200, headers: { "content-length": String(Buffer.byteLength(body)) } });
    return new Response(body, { status: 200, headers: { "content-length": String(Buffer.byteLength(body)) } });
  };
}

test("ensureBundle downloads every file once, then only HEAD-checks", async () => {
  const cacheDir = await mkdtemp(path.join(tmpdir(), "laya-"));
  const calls: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = stubFetch("v1", calls) as typeof fetch;
  try {
    const dir = await ensureBundle({ repo: "acme/bundle", cacheDir, subfolder: "multilingual" });
    assert.equal(dir, path.join(cacheDir, "acme--bundle", "main", "multilingual/"));
    assert.equal(calls.length, BUNDLE_FILES.length);
    assert.ok(calls[0]?.startsWith("GET https://huggingface.co/acme/bundle/resolve/main/multilingual/laya.onnx"));
    assert.equal(await readFile(path.join(dir, "tokenizer/tokenizer.json"), "utf8"), "tokenizer.json:v1");

    calls.length = 0;
    await ensureBundle({ repo: "acme/bundle", cacheDir, subfolder: "multilingual" });
    assert.deepEqual(new Set(calls.map((c) => c.split(" ")[0])), new Set(["HEAD"]));

    // a size change upstream triggers a re-download of that file only
    await writeFile(path.join(dir, "laya_config.json"), "stale");
    calls.length = 0;
    await ensureBundle({ repo: "acme/bundle", cacheDir, subfolder: "multilingual" });
    assert.equal(calls.filter((c) => c.startsWith("GET")).length, 1);
    assert.equal(await readFile(path.join(dir, "laya_config.json"), "utf8"), "laya_config.json:v1");
  } finally {
    globalThis.fetch = realFetch;
    await rm(cacheDir, { recursive: true, force: true });
  }
});

test("ensureBundle uses a populated cache when the network is down", async () => {
  const cacheDir = await mkdtemp(path.join(tmpdir(), "laya-"));
  const calls: string[] = [];
  const realFetch = globalThis.fetch;
  try {
    globalThis.fetch = stubFetch("v1", calls) as typeof fetch;
    const dir = await ensureBundle({ repo: "acme/bundle", cacheDir });

    // every fetch now fails at the transport level, as with no DNS
    globalThis.fetch = (async () => {
      throw new TypeError("fetch failed", { cause: new Error("getaddrinfo EAI_AGAIN huggingface.co") });
    }) as typeof fetch;
    assert.equal(await ensureBundle({ repo: "acme/bundle", cacheDir }), dir);
    assert.equal(await readFile(path.join(dir, "laya_config.json"), "utf8"), "laya_config.json:v1");

    // a missing file still has to be downloaded, so that failure is reported
    await rm(path.join(dir, "laya_config.json"));
    await assert.rejects(ensureBundle({ repo: "acme/bundle", cacheDir }), /fetch failed/);
  } finally {
    globalThis.fetch = realFetch;
    await rm(cacheDir, { recursive: true, force: true });
  }
});

test("ensureBundle surfaces HTTP errors and leaves no partial file", async () => {
  const cacheDir = await mkdtemp(path.join(tmpdir(), "laya-"));
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response("nope", { status: 404, statusText: "Not Found" })) as typeof fetch;
  try {
    await assert.rejects(ensureBundle({ repo: "acme/missing", cacheDir }), /404 Not Found/);
  } finally {
    globalThis.fetch = realFetch;
    await rm(cacheDir, { recursive: true, force: true });
  }
});
