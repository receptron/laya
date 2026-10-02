/** Request / response shapes. They follow TypeSafe Jev's `system_one` API, which Laya reproduces. */

export type QuestionType = "choice" | "score" | "noul";

export interface ChoiceQuestion {
  type: "choice";
  instructions: string | object;
  /** option -> short description (or null), or a plain list of option names */
  criteria: Record<string, string | null> | string[];
}

export interface ScoreQuestion {
  type: "score";
  instructions: string | object;
  /** ordered levels, index 0 = lowest */
  criteria: string[];
}

export interface NoulQuestion {
  type: "noul";
  instructions: string | object;
  criteria?: { true?: string; false?: string };
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  /** 1 - normalized entropy of the answer distribution */
  confidence: number;
  rl_agent: { act_probability: number };
}

export interface ScoreAnswer {
  type: "score";
  /** expected level (0 .. levels-1) */
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
  rl_agent: { act_probability: number };
}

export interface NoulAnswer {
  type: "noul";
  /** P(true) */
  noul: number;
  rl_agent: { act_probability: number };
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export type AnswerFor<Q extends Question> = Q extends ChoiceQuestion ? ChoiceAnswer : Q extends ScoreQuestion ? ScoreAnswer : NoulAnswer;

export interface SystemOneResult<Q extends Record<string, Question>> {
  model: string;
  answers: { [K in keyof Q]: AnswerFor<Q[K]> };
  usage: { input_tokens: number; output_tokens: number };
}

/** laya_config.json, written by export/export_onnx.py from the checkpoint's rl_agent_config.json */
export interface LayaConfig {
  max_len: number;
  head_max_len: number;
  temperature: [number, number, number];
  temperature_by_options?: Record<string, number>;
}
