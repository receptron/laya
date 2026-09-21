# @receptron/laya

Run **[Laya](https://huggingface.co/convaiinnovations/laya)** — the open-source, Jev-compatible
_System 1 decision model_ by Convai Innovations — from Node.js / TypeScript.

Laya does not generate text. You hand it a state (a ticket, an email, a JSON object) and typed
questions, and it returns every answer with calibrated probabilities in **one forward pass**:

- `choice` — pick one option, with a probability per option
- `score` — an expected level on an ordered rubric, with the distribution
- `noul` — a calibrated P(true) for a yes/no statement

This package runs the model with [ONNX Runtime](https://onnxruntime.ai/); PyTorch and Python are
not needed at runtime. The request/response shape is the same as the Python reference
implementation (`RLAgent.system_one`) and as TypeSafe Jev's `system_one` API, and the output
matches the Python implementation to four decimal places.

## Install

```sh
npm install @receptron/laya
```

Node.js 20 or newer. The ONNX weights (about 1.7 GB, fp32) are downloaded from Hugging Face on first
use and cached under `~/.cache/receptron-laya` (override with `LAYA_CACHE`). Budget roughly 2 GB of
RAM for the loaded model plus a few hundred MB per batch of questions.

## Usage

```ts
import { Laya } from "@receptron/laya";

const laya = await Laya.load();

const result = await laya.systemOne(
  { subject: "Refund not received", body: "I cancelled two weeks ago and still have no refund..." },
  {
    department: {
      type: "choice",
      instructions: "Which team should handle this ticket?",
      criteria: { billing: "payments, refunds, invoices", support: "product help and bugs", sales: "new purchases" },
    },
    urgency: {
      type: "score",
      instructions: "How urgent is this ticket?",
      criteria: ["not urgent", "somewhat urgent", "urgent", "critical"],
    },
    churn_risk: { type: "noul", instructions: "Is the customer likely to cancel or dispute?" },
  },
);

result.answers.department.choice; // "billing"
result.answers.department.probabilities; // { billing: 0.9415, support: 0.031, sales: 0.0275 }
result.answers.urgency.score; // 1.3886   (expected level, 0..3)
result.answers.churn_risk.noul; // 0.0988   (P(true))
result.usage.input_tokens; // 267

await laya.close();
```

The answer types follow the question types, so `result.answers.department` is a `ChoiceAnswer`
and `result.answers.churn_risk` a `NoulAnswer` without any casting.

### Options

```ts
await Laya.load({
  modelDir: "./onnx", // use a local export instead of downloading (see below)
  repo: "receptron/laya-onnx", // Hugging Face repo that holds the ONNX bundle
  subfolder: "multilingual", // a checkpoint variant inside that repo
  revision: "main", // pin a commit hash for reproducible results; "main" follows the repo
  cacheDir: "/var/cache/laya",
  token: process.env.HF_TOKEN, // for private repos
  onProgress: ({ file, received, total }) => {}, // download progress
  executionProviders: ["cpu"], // onnxruntime-node execution providers
  sessionOptions: { intraOpNumThreads: 4 },
});
```

Every question of one `systemOne` call is batched into a single run; a call with three questions
takes about 140 ms on an Apple-silicon CPU once the model is warm.

## Exporting the ONNX bundle yourself

`export/export_onnx.py` turns the Hugging Face checkpoint (ModernBERT encoder + Laya's decision head)
into one ONNX graph and copies the tokenizer and calibration values next to it. You only need this
to build a bundle from a newer checkpoint or from a variant that is not published:

```sh
cd export
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch transformers safetensors onnx onnxscript onnxruntime huggingface_hub
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('convaiinnovations/laya', local_dir='model', allow_patterns=['model.safetensors','encoder/*','tokenizer/*','rl_agent_config.json','rl_common.py','rl_agent_api.py'])"
.venv/bin/python export_onnx.py model ../onnx   # prints the max logit difference vs. PyTorch (≈1e-5)
```

Then `Laya.load({ modelDir: "./onnx" })`. The bundle is the five files listed in `BUNDLE_FILES`:
`laya.onnx`, `laya.onnx.data`, `laya_config.json`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`.

## Limits

- Each question's options must fit in `head_max_len` (192) tokens; `systemOne` throws otherwise.
  Fewer than about 20 options per `choice` question is the model's own recommendation.
- The state is truncated to `max_len` (512 tokens for the English checkpoint) after the question header.
- A JSON state is serialized like Python's `json.dumps(ensure_ascii=False)` so that tokens match the
  reference implementation; non-integer numbers may format differently between JS and Python.

## Development

```sh
yarn install
yarn test        # unit tests; the model test runs when ./onnx holds a bundle (or LAYA_MODEL_DIR)
yarn typecheck
yarn build
LAYA_MODEL_DIR=./onnx yarn example
```

## License

MIT. The Laya model weights are published by Convai Innovations under Apache 2.0.
