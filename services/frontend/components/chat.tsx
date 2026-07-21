"use client";

import { useState, useRef, useEffect } from "react";
import { SendHorizontal, ImagePlus, X, Check } from "lucide-react";
import { toast } from "sonner";
import { sendMessage, sendConfirmation } from "@/lib/api";
import type { ChatMessage, AwaitingConfirmation } from "@/lib/types";
import MessageBubble from "./message-bubble";

function describeEdit(proposed: Record<string, unknown>): string {
  const { tool_name, ...rest } = proposed;
  const details = Object.entries(rest)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
  return details ? `${tool_name} (${details})` : String(tool_name ?? "this edit");
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [pendingConfirmation, setPendingConfirmation] = useState<AwaitingConfirmation | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const b64 = result.split(",")[1];
      setImageB64(b64);
      setImagePreview(result);
      // A newly picked image always starts a fresh conversation server-side.
      setThreadId(null);
      setPendingConfirmation(null);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  }

  function handleTextareaChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (loading || pendingConfirmation) return; // must resolve the pending edit first
    if (!input.trim() && !imageB64 && !threadId) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: input.trim() || "What's in this image?",
      ...(imageB64 ? { image_base64: imageB64 } : {}),
    };
    const next = [...messages, userMessage];
    setMessages(next);
    setInput("");
    const sentImageB64 = imageB64;
    setImageB64(null);
    setImagePreview(null);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setLoading(true);

    try {
      const reply = await sendMessage(userMessage.content, sentImageB64 ?? undefined, threadId ?? undefined);
      setThreadId(reply.thread_id);

      if (reply.awaiting_confirmation) {
        setPendingConfirmation(reply.awaiting_confirmation);
        setMessages([...next, {
          role: "assistant",
          content: `Confirm: ${describeEdit(reply.awaiting_confirmation.proposed)}?`,
        }]);
      } else {
        setMessages([...next, {
          role: "assistant",
          content: reply.response ?? "",
          ...(reply.annotated_image_base64 ? { image_base64: reply.annotated_image_base64 } : {}),
        }]);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function handleConfirm(confirmed: boolean) {
    if (!threadId || loading) return;
    setPendingConfirmation(null);
    setLoading(true);
    try {
      const reply = await sendConfirmation(threadId, confirmed);
      if (reply.awaiting_confirmation) {
        setPendingConfirmation(reply.awaiting_confirmation);
      }
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: reply.response ?? (confirmed ? "Done." : "Cancelled."),
        ...(reply.annotated_image_base64 ? { image_base64: reply.annotated_image_base64 } : {}),
      }]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (loading) return; // guard: Enter must not bypass the in-flight-request lock
      handleSubmit(e as unknown as React.FormEvent);
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      {/* Header */}
      <div className="border-b px-6 py-4 shrink-0 bg-gradient-to-r from-background to-muted/30">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <h1 className="text-lg font-semibold tracking-tight">Vision Agent</h1>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && (
          <p className="text-center text-muted-foreground mt-20">
            Send a message or upload an image to get started.
          </p>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {pendingConfirmation && (
          <div className="flex items-center gap-3 rounded-xl border bg-muted/40 px-4 py-3">
            <span className="text-sm flex-1">{pendingConfirmation.message}</span>
            <button
              onClick={() => handleConfirm(true)}
              disabled={loading}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-sm hover:opacity-90 disabled:opacity-40"
            >
              <Check className="w-4 h-4" /> Confirm
            </button>
            <button
              onClick={() => handleConfirm(false)}
              disabled={loading}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border text-sm hover:bg-muted disabled:opacity-40"
            >
              <X className="w-4 h-4" /> Cancel
            </button>
          </div>
        )}
        {loading && (
          <div className="flex gap-1 pl-1">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="w-2 h-2 bg-primary/60 rounded-full animate-bounce"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t px-6 py-4 shrink-0">
        {imagePreview && (
          <div className="mb-3 relative inline-block">
            <img
              src={imagePreview}
              alt="preview"
              className="h-20 w-20 rounded-lg border object-cover"
            />
            <button
              onClick={() => {
                setImageB64(null);
                setImagePreview(null);
              }}
              className="absolute -top-2 -right-2 bg-background border rounded-full p-0.5 hover:bg-muted"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )}
        <form onSubmit={handleSubmit} className="flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            title="Upload image"
            className="p-2 rounded-lg hover:bg-muted text-muted-foreground transition-colors shrink-0"
          >
            <ImagePlus className="w-5 h-5" />
          </button>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={!!pendingConfirmation}
            className="flex-1 resize-none rounded-xl border bg-muted/40 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary shadow-sm overflow-hidden disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={loading || !!pendingConfirmation || (!input.trim() && !imageB64 && !threadId)}
            className="p-2.5 rounded-xl bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-sm shrink-0"
          >
            <SendHorizontal className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}
