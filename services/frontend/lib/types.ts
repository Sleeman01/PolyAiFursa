export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image_base64?: string;
}

export interface AwaitingConfirmation {
  type: string;
  proposed: Record<string, unknown>;
  message: string;
}
