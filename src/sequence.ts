/**
 * Pure (model-free) port of the request rendering in the checkpoint's rl_common.py:
 * option rendering, sequence layout, temperature buckets, confidence. Kept separate from the
 * ONNX session so it can be unit-tested without the weights.
 */
import type { LayaConfig, Question, QuestionType } from "./types.js";

export const QTYPES: Record<QuestionType, number> = { choice: 0, score: 1, noul: 2 };
const QTYPE_NAMES: QuestionType[] = ["choice", "score", "noul"];

export interface InternalQ {
  t: QuestionType;
  ins: string;
  crit: Record<string, string | null> | string[] | { true?: string; false?: string } | undefined;
}

/** RLAgent._to_internal */
export function toInternal(q: Question): InternalQ {
  let crit: InternalQ["crit"] = q.criteria;
  if (q.type === "choice" && Array.isArray(crit)) {
    crit = Object.fromEntries(crit.map((c) => [c, null]));
  }
  return { t: q.type, ins: typeof q.instructions === "string" ? q.instructions : JSON.stringify(q.instructions), crit };
}

/** Option texts in label-index order. Noul is always [false, true] so p[1] == noul. */
export function renderOptions(q: InternalQ): string[] {
  if (q.t === "choice") {
    return Object.entries(q.crit as Record<string, string | null>).map(([k, v]) => (v ? `${k}: ${v}` : k));
  }
  if (q.t === "score") {
    return (q.crit as string[]).map((c, i) => `level ${i}: ${c}`);
  }
  const c = (q.crit ?? {}) as { true?: string; false?: string };
  return ["false: " + (c.false || "no, the statement does not hold"), "true: " + (c.true || "yes, the statement holds")];
}

/** Python's json.dumps(obj, ensure_ascii=False): ", " / ": " separators, insertion key order. */
export function pyJsonDumps(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : JSON.stringify(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return "[" + v.map(pyJsonDumps).join(", ") + "]";
  return (
    "{" +
    Object.entries(v as Record<string, unknown>)
      .map(([k, x]) => `${JSON.stringify(k)}: ${pyJsonDumps(x)}`)
      .join(", ") +
    "}"
  );
}

export function serializeState(state: unknown): string {
  return typeof state === "string" ? state : pyJsonDumps(state);
}

function sizeBucket(k: number): string {
  if (k <= 2) return "2";
  if (k <= 5) return "3-5";
  if (k <= 10) return "6-10";
  return "11+";
}

/** Key for the per-cardinality temperature: a 2-option noul and a 20-option choice need different scaling. */
export function tempBucket(qtype: number, k: number): string {
  return `${QTYPE_NAMES[qtype]}:${sizeBucket(k)}`;
}

export const TEMP_MIN = 0.5;
export const TEMP_MAX = 5;

/**
 * Return a finite temperature in the supported range. Invalid values use the supplied fallback,
 * so malformed checkpoint data cannot turn logits or softmax into NaN.
 */
export function clampTemperature(value: unknown, fallback = 1): number {
  const t = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(TEMP_MAX, Math.max(TEMP_MIN, t));
}

/** Resolve and clamp the temperature immediately before it is used for logits. */
export function temperatureFor(config: LayaConfig, qtype: number, k: number): number {
  const bucketValue = config.temperature_by_options?.[tempBucket(qtype, k)];
  const baseValue = config.temperature[qtype];
  const bucketIsFinite = typeof bucketValue === "number" && Number.isFinite(bucketValue);
  const baseIsFinite = typeof baseValue === "number" && Number.isFinite(baseValue);
  return clampTemperature(bucketIsFinite ? bucketValue : baseIsFinite ? baseValue : 1);
}

/** Clamp checkpoint temperatures without mutating the input config. */
export function clampTemperatures(config: LayaConfig): LayaConfig {
  const changed: string[] = [];
  const clamp = (key: string, t: unknown): number => {
    const v = clampTemperature(t);
    if (v !== t) changed.push(`${key} ${String(t)} -> ${v}`);
    return v;
  };
  const temperature = config.temperature.map((t, i) => clamp(QTYPE_NAMES[i] ?? String(i), t)) as [number, number, number];
  const temperature_by_options = Object.fromEntries(Object.entries(config.temperature_by_options ?? {}).map(([k, t]) => [k, clamp(k, t)]));
  if (changed.length > 0) {
    console.warn(
      `laya: this checkpoint ships invalid or out-of-range temperatures: ${changed.join(", ")}. Treat confidence from the affected entries as uncalibrated.`,
    );
  }
  return { ...config, temperature, temperature_by_options };
}

/** Jev-style confidence: 1 - normalized entropy of the answer distribution. */
export function confidenceFromProbs(p: number[]): number {
  const k = p.length;
  if (k < 2) return 1;
  let ent = 0;
  for (const x of p) ent -= x * Math.log(Math.max(x, 1e-12));
  return 1 - ent / Math.log(k);
}

export function softmax(z: number[]): number[] {
  const zmax = Math.max(...z);
  const e = z.map((v) => Math.exp(v - zmax));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export interface SpecialIds {
  cls: number;
  sep: number;
  mask: number;
  pad: number;
  /** the literal mask token text, scrubbed from user text so it cannot inject a marker */
  maskTok: string;
}

/** Tokenizer surface the sequence builder needs: text -> ids, no special tokens added. */
export type Encode = (text: string) => number[];

/**
 * rl_common.build_sequence:
 *   [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
 * Returns the ids and the position of each option's [MASK] marker.
 */
export function buildSequence(
  encode: Encode,
  ids: SpecialIds,
  state: unknown,
  q: InternalQ,
  maxLen: number,
  headMaxLen: number,
): { ids: number[]; markers: number[] } {
  const scrub = (s: string) => s.split(ids.maskTok).join(" ");
  const opts = renderOptions(q);
  let headIds = encode(`${q.t} question: ${scrub(q.ins)}`);
  let optIds = opts.map((o) => [ids.mask, ...encode(" " + scrub(o)).slice(0, 48)]);
  const total = (xs: number[][]) => xs.reduce((s, o) => s + o.length, 0);
  let optBudget = headMaxLen - total(optIds);
  if (optBudget < 16) {
    // too many / too long options: shrink every option text evenly
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - total(optIds);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));
  const seq = [ids.cls, ...headIds, ids.sep];
  const markers: number[] = [];
  for (const o of optIds) {
    markers.push(seq.length);
    seq.push(...o);
  }
  seq.push(ids.sep);
  const room = Math.max(0, maxLen - seq.length - 1);
  const st = encode(scrub(serializeState(state))).slice(0, room);
  seq.push(...st, ids.sep);
  return { ids: seq.slice(0, maxLen), markers: markers.filter((m) => m < maxLen) };
}
