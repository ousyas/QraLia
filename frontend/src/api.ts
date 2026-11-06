import type { ConversationMessageResponse, ConversationResponse, InvoiceResponse, QuestionResponse } from "./types";

const API_ROOT = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const body = await response.json().catch(() => null);
      if (body && typeof body.detail === "string") {
        throw new Error(body.detail);
      }
    }
    throw new Error(
      `Le serveur a répondu avec l’erreur ${response.status}. Consultez le terminal FastAPI.`,
    );
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    return (await fetch(`${API_ROOT}/health`)).ok;
  } catch {
    return false;
  }
}

export async function uploadInvoice(file: File): Promise<InvoiceResponse> {
  const data = new FormData();
  data.append("file", file);
  data.append("confirm_synthetic", "true");
  return checked(await fetch(`${API_ROOT}/v1/invoices`, { method: "POST", body: data }));
}

export async function askQuestion(
  invoiceId: string,
  question: string,
  language: string,
): Promise<QuestionResponse> {
  return checked(await fetch(`${API_ROOT}/v1/invoices/${invoiceId}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, language }),
  }));
}

export async function createConversation(
  invoiceId: string,
  language = "auto",
): Promise<ConversationResponse> {
  return checked(await fetch(`${API_ROOT}/v1/invoices/${invoiceId}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language }),
  }));
}

export async function sendConversationMessage(
  conversationId: string,
  message: string,
  language: string,
): Promise<ConversationMessageResponse> {
  return checked(await fetch(`${API_ROOT}/v1/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, language }),
  }));
}

export async function deleteInvoice(invoiceId: string): Promise<void> {
  await checked(await fetch(`${API_ROOT}/v1/invoices/${invoiceId}`, { method: "DELETE" }));
}
