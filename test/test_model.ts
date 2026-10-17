/**
 * End-to-end check against numbers produced by the Python reference (rl_agent_api.RLAgent.system_one)
 * on the same input. Skips when no ONNX bundle is available locally (set LAYA_MODEL_DIR to point at one).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import path from "node:path";
import { Laya } from "../src/index.js";

const modelDir = process.env.LAYA_MODEL_DIR ?? path.resolve(import.meta.dirname, "../onnx");
const available = existsSync(path.join(modelDir, "laya.onnx.data"));

test("systemOne reproduces the Python reference output", { skip: !available && "no ONNX bundle on disk" }, async () => {
  const laya = await Laya.load({ modelDir });
  try {
    const r = await laya.systemOne(
      {
        subject: "Refund not received",
        body: "I cancelled my subscription two weeks ago and I still have not received my refund. This is the third time I am writing. If this is not resolved I will dispute the charge with my bank.",
      },
      {
        department: {
          type: "choice",
          instructions: "Which team should handle this ticket?",
          criteria: { billing: "payments, refunds, invoices", support: "product help and bugs", sales: "new purchases and upgrades" },
        },
        urgency: { type: "score", instructions: "How urgent is this ticket?", criteria: ["not urgent", "somewhat urgent", "urgent", "critical"] },
        churn_risk: { type: "noul", instructions: "Is the customer likely to cancel or dispute?" },
      },
    );
    assert.equal(r.usage.input_tokens, 267);
    assert.equal(r.answers.department.choice, "billing");
    assert.deepEqual(r.answers.department.probabilities, { billing: 0.9415, support: 0.031, sales: 0.0275 });
    assert.equal(r.answers.department.confidence, 0.7603);
    assert.equal(r.answers.urgency.score, 1.3886);
    assert.deepEqual(r.answers.urgency.probabilities, { "0": 0.1752, "1": 0.2947, "2": 0.4962, "3": 0.0338 });
    assert.equal(r.answers.churn_risk.noul, 0.0988);
    assert.equal(r.answers.department.answer_confidence, 0.9415);
    assert.equal(r.answers.urgency.answer_confidence, 0.4962);
    assert.equal(r.answers.churn_risk.answer_confidence, 0.9012);

    // a single-option choice runs (the graph needs two slots; the second is masked) and is fully decided
    const one = await laya.systemOne("anything", { only: { type: "choice", instructions: "Pick one", criteria: ["solo"] } });
    assert.equal(one.answers.only.choice, "solo");
    assert.deepEqual(one.answers.only.probabilities, { solo: 1 });
    assert.equal(one.answers.only.answer_confidence, 1);
    assert.equal(one.answers.only.confidence, 1);
    assert.ok(Number.isFinite(one.answers.only.rl_agent.act_probability));
  } finally {
    await laya.close();
  }
});
