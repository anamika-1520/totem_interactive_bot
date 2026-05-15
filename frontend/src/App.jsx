import { useState } from "react";
import ChatInterface from "./ChatInterface";
import GraphView from "./GraphView";

export default function App() {
  const [sessionId, setSessionId] = useState(null);

  return (
    <main className="app-shell">
      <section className="panel panel-chat">
        <div className="panel-header">
          <p className="eyebrow">Prompt Optimizer</p>
          <h1>Chat Workflow</h1>
          <p className="panel-copy">
            Send a prompt, confirm extracted intent, and review the optimized result.
          </p>
        </div>
        <ChatInterface onSessionChange={setSessionId} />
      </section>

      <section className="panel panel-graph">
        <div className="panel-header">
          <p className="eyebrow">Session Graph</p>
          <h2>Live Workflow View</h2>
          <p className="panel-copy">
            {sessionId
              ? `Tracking session ${sessionId}`
              : "Run a prompt flow to populate the graph."}
          </p>
        </div>
        <GraphView sessionId={sessionId} />
      </section>
    </main>
  );
}
