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
