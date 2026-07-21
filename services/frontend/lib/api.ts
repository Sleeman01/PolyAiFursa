import type { AwaitingConfirmation } from "./types";

const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8000";

export interface ChatReply {
  response?: string;
  annotated_image_base64?: string;
  thread_id: string;
  awaiting_confirmation?: AwaitingConfirmation;
}

async function parseReply(res: Response): Promise<ChatReply> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || res.statusText);
  }
  const data = await res.json();
  return {
    response: data.response as string | undefined,
    annotated_image_base64: data.annotated_image_base64 as string | undefined,
    thread_id: data.thread_id as string,
    awaiting_confirmation: data.awaiting_confirmation as AwaitingConfirmation | undefined,
  };
}

export async function sendMessage(
  message: string,
  imageBase64?: string,
  threadId?: string
): Promise<ChatReply> {
  const res = await fetch(`${AGENT_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      ...(imageBase64 ? { image_base64: imageBase64 } : {}),
      ...(threadId ? { thread_id: threadId } : {}),
    }),
  });
  return parseReply(res);
}

export async function sendConfirmation(
  threadId: string,
  confirmed: boolean
): Promise<ChatReply> {
  const res = await fetch(`${AGENT_URL}/chat/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, confirmed }),
  });
  return parseReply(res);
}
