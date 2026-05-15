import React, { useRef, useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000/api";

function formatReduction(value) {
  if (typeof value === "number") {
    return `${value.toFixed(1)}% reduction`;
  }

  if (typeof value === "string") {
    return value.includes("%") ? `${value} reduction` : value;
  }

  return "Reduction unavailable";
}

export default function ChatInterface({ onSessionChange }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [status, setStatus] = useState("idle");
  const [pendingIntent, setPendingIntent] = useState(null);
  const [pendingChoices, setPendingChoices] = useState([]);
  const [draftInput, setDraftInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);

  const closeIntentCards = () => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (!msg.intent) {
          return msg;
        }

        return {
          ...msg,
          intentHandled: true,
        };
      }),
    );
  };

  const loadIntentStep = async (newSessionId) => {
    setSessionId(newSessionId);
    onSessionChange?.(newSessionId);

    const { data: intentData } = await axios.post(
      `${API_BASE}/extract-intent/${newSessionId}`,
    );

    if (!intentData.requires_confirmation) {
      const choices = intentData.choices || [];
      setPendingIntent(null);
      setPendingChoices(choices);
      setStatus("needs_clarification");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: intentData.message || "Confidence is low. Please rewrite the prompt more clearly and try again.",
          intent: intentData.intent,
          intentHandled: choices.length === 0,
          clarificationChoices: choices,
        },
      ]);
      return;
    }

    setPendingIntent(intentData.intent);
    setPendingChoices([]);
    setStatus("awaiting_confirmation");
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: intentData.message || "I understood your request. Please confirm:",
        intent: intentData.intent,
        intentHandled: false,
      },
    ]);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    setMessages((prev) => [...prev, { role: "user", content: input }]);
    setDraftInput(input);
    setStatus("processing");

    try {
      const { data: session } = await axios.post(`${API_BASE}/process-text`, {
        text: input,
      });
      setDraftInput(input);
      await loadIntentStep(session.session_id);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: error.response?.data?.detail || "Error processing request",
        },
      ]);
      setStatus("error");
    }

    setInput("");
  };

  const sendAudioFile = async (file, label = "Voice input recorded.") => {
    if (!file) return;

    setMessages((prev) => [...prev, { role: "user", content: label }]);
    setStatus("processing");

    try {
      const formData = new FormData();
      formData.append("audio", file);
      const { data: session } = await axios.post(`${API_BASE}/process-voice`, formData);
      setDraftInput(session.transcribed_text || "");
      await loadIntentStep(session.session_id);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: error.response?.data?.detail || "Error processing audio",
        },
      ]);
      setStatus("error");
    }
  };

  const resolveClarification = async (selectedTask) => {
    closeIntentCards();
    setStatus("processing");

    try {
      const { data } = await axios.post(`${API_BASE}/resolve-clarification`, {
        session_id: sessionId,
        selected_task: selectedTask,
      });

      setPendingIntent(data.intent);
      setPendingChoices([]);
      setStatus("awaiting_confirmation");
      setMessages((prev) => [
        ...prev,
        { role: "user", content: selectedTask },
        {
          role: "assistant",
          content: data.message || "Please confirm the selected task:",
          intent: data.intent,
          intentHandled: false,
        },
      ]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: error.response?.data?.detail || "Error resolving clarification",
        },
      ]);
      setStatus("error");
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      setIsRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        await sendAudioFile(file);
      };

      recorder.start();
      setIsRecording(true);
      setStatus("recording");
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Please allow microphone permission to record voice input.",
        },
      ]);
      setStatus("error");
    }
  };

  const confirmIntent = async (confirmed) => {
    if (!confirmed) {
      closeIntentCards();
      setPendingIntent(null);
      setPendingChoices([]);
      setStatus("idle");
      setInput(draftInput);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "The prompt is back in the input box. Edit it and press Send again.",
        },
      ]);
      return;
    }

    closeIntentCards();
    setStatus("optimizing");

    try {
      await axios.post(`${API_BASE}/confirm-intent`, {
        session_id: sessionId,
        confirmed,
        modifications: null,
      });

      const { data: optimized } = await axios.post(
        `${API_BASE}/optimize-prompt/${sessionId}`,
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Optimized prompt generated.",
          optimized,
        },
      ]);
      setStatus("completed");
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: error.response?.data?.detail || "Error optimizing request",
        },
      ]);
      setStatus("error");
    }

    setPendingIntent(null);
    setPendingChoices([]);
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="content">{msg.content}</div>

            {msg.intent && (
              <div className="intent-card">
                <h4>Detected Intent:</h4>
                <p><strong>Task:</strong> {msg.intent.task}</p>
                <p><strong>Domain:</strong> {msg.intent.domain}</p>
                <p><strong>Format:</strong> {msg.intent.output_format}</p>
                <p><strong>Audience:</strong> {msg.intent.audience}</p>
                <p><strong>Confidence:</strong> {(msg.intent.confidence_score * 100).toFixed(1)}%</p>

                {!msg.intentHandled && !msg.clarificationChoices?.length && (
                  <div className="confirmation-buttons">
                    <button onClick={() => confirmIntent(true)}>
                      Confirm
                    </button>
                    <button onClick={() => confirmIntent(false)}>
                      Modify
                    </button>
                  </div>
                )}

                {!msg.intentHandled && msg.clarificationChoices?.length > 0 && (
                  <div className="confirmation-buttons">
                    {msg.clarificationChoices.map((choice) => (
                      <button key={choice} onClick={() => resolveClarification(choice)}>
                        {choice}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {msg.optimized && (
              <div className="result-card">
                <h4>Optimized Prompt:</h4>
                <pre>{msg.optimized.optimized_prompt}</pre>
                <div className="metrics">
                  <span>Original: {msg.optimized.original_tokens} tokens</span>
                  <span>Optimized: {msg.optimized.optimized_tokens} tokens</span>
                  <span className="reduction">{formatReduction(msg.optimized.token_reduction)}</span>
                </div>
                {msg.optimized.memory_output && (
                  <>
                    <h4>Final Memory:</h4>
                    <pre>{JSON.stringify(msg.optimized.memory_output, null, 2)}</pre>
                  </>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Enter your prompt..."
          disabled={status === "awaiting_confirmation" || status === "optimizing"}
        />
        <button
          onClick={sendMessage}
          disabled={status === "awaiting_confirmation" || status === "optimizing"}
        >
          Send
        </button>
        <button
          onClick={toggleRecording}
          disabled={status === "awaiting_confirmation" || status === "optimizing"}
        >
          {isRecording ? "Stop Recording" : "Start Recording"}
        </button>
      </div>

      <div className="status-bar">
        Status: {status}{pendingIntent ? " - confirmation pending" : ""}{pendingChoices.length ? " - clarification pending" : ""}
      </div>
    </div>
  );
}
