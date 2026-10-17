export type EventKind =
  | "laboratory"
  | "imaging"
  | "visit"
  | "medication"
  | "procedure"
  | "vital"
  | "hospitalization"
  | "symptom"
  | "other";

export type Fact = {
  id: string;
  label: string;
  value: string;
  interpretation?: "normal" | "high" | "low" | "abnormal";
  evidence: string;
  attributes?: Record<string, unknown>;
};

export type TimelineEvent = {
  id: string;
  date: string;
  displayDate: string;
  type: EventKind;
  title: string;
  institution: string;
  summary: string;
  facts: Fact[];
  source: string;
  artifactId?: string;
};

export type TrendPoint = {
  date: string;
  value: number;
};
