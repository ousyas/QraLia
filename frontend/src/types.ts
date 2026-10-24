export type FieldView = {
  name: string;
  label: string;
  value: unknown;
  display_value: string;
  status: "extracted" | "derived" | "not_present" | "unreadable";
  confidence: number;
  page: number;
};

export type InvoiceResponse = {
  invoice_id: string;
  created_at: string;
  expires_at: string;
  document_confidence: number;
  fields: FieldView[];
  validation: {
    valid: boolean;
    scope_supported: boolean;
    critical_fields_complete: boolean;
    errors: string[];
    warnings: string[];
    low_confidence_fields: string[];
  };
};

export type QuestionResponse = {
  answer: string;
  route: "invoice" | "official_rag" | "both" | "unknown";
  language: string;
  abstained: boolean;
  confidence: number;
  evidence: Array<{
    source: "invoice" | "official_corpus";
    evidence_id: string;
    label: string;
    value?: unknown;
    confidence: number;
  }>;
  citations: Array<{ chunk_id: string; title: string; url: string }>;
  warnings: string[];
};

export type ConversationResponse = {
  conversation_id: string;
  invoice_id: string;
  preferred_language: string;
  created_at: string;
  updated_at: string;
  expires_at: string;
  turns: Array<{
    message_id: string;
    role: "user" | "assistant";
    text: string;
    language: string;
    created_at: string;
    standalone_question?: string | null;
    route?: QuestionResponse["route"] | null;
    evidence_ids: string[];
  }>;
};

export type ConversationMessageResponse = {
  conversation_id: string;
  user_message_id: string;
  assistant_message_id: string;
  resolution: {
    action: "standalone" | "rewrite" | "clarify";
    standalone_question?: string | null;
    clarification_question?: string | null;
    language: string;
    history_used: boolean;
    resolved_references: string[];
    method: "direct" | "qwen" | "deterministic_fallback";
  };
  answer?: QuestionResponse | null;
  clarification_question?: string | null;
};
