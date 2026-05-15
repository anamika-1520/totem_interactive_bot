import React, { useEffect, useState } from "react";
import axios from "axios";
import ReactFlow, { Background, Controls } from "reactflow";
import "reactflow/dist/style.css";

const nodeStyle = (node) => {
  const done = node.status === "done";
  return {
    borderRadius: node.shape === "diamond" ? 0 : 2,
    padding: 10,
    border: "1px solid rgba(255,255,255,0.35)",
    background: done ? "#25c98f" : "#202020",
    color: "#fff",
    width: node.shape === "diamond" ? 86 : 145,
    height: node.shape === "diamond" ? 86 : 34,
    display: "grid",
    placeItems: "center",
    fontSize: 10,
    textAlign: "center",
    clipPath: node.shape === "diamond" ? "polygon(50% 0, 100% 50%, 50% 100%, 0 50%)" : "none",
  };
};

export default function GraphView({ sessionId }) {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);

  useEffect(() => {
    if (!sessionId) return;

    const fetchGraph = async () => {
      try {
        const { data } = await axios.get(
          `http://localhost:8000/api/session/${sessionId}/graph`,
        );

        const flowNodes = data.nodes.map((node, i) => ({
          id: node.id,
          data: { label: node.label },
          position: { x: node.x ?? 180 + (i % 2) * 180, y: node.y ?? 40 + i * 110 },
          style: nodeStyle(node),
        }));

        const flowEdges = data.edges.map((edge) => ({
          id: `${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          animated: true,
          style: { stroke: "#6d6d6d" },
          labelStyle: { fill: "#fff", fontSize: 10 },
        }));

        setNodes(flowNodes);
        setEdges(flowEdges);
      } catch (error) {
        console.error(error);
        setNodes([]);
        setEdges([]);
      }
    };

    fetchGraph();
    const intervalId = setInterval(fetchGraph, 2000);

    return () => clearInterval(intervalId);
  }, [sessionId]);

  if (!sessionId) {
    return (
      <div className="graph-shell">
        <div className="graph-empty">
          <p>Workflow graph will appear here after a session starts.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="graph-shell">
      {nodes.length === 0 ? (
        <div className="graph-empty">
          <p>The workflow has started. Graph data will appear here shortly.</p>
        </div>
      ) : (
        <div className="graph-canvas">
          <ReactFlow nodes={nodes} edges={edges} fitView fitViewOptions={{ padding: 0.12 }}>
            <Background />
            <Controls />
          </ReactFlow>
        </div>
      )}
    </div>
  );
}
