// src/App.tsx
import { useState } from "react";
import { useModelComparison } from "./hooks/useModelComparison";
import { TokenizedText } from "./components/TokenizedText";

const AVAILABLE_MODELS = ["gpt-4", "gpt-3.5-turbo"]; // match your real ADAPTER_REGISTRY keys

function App() {
  const [text, setText] = useState("");
  const { modelStates, addModel, removeModel } = useModelComparison();

  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type text to tokenize..."
        aria-label="Text to tokenize"
        rows={4}
        style={{ width: "100%", maxWidth: "40rem", boxSizing: "border-box" }}
      />
      <div style={{ marginTop: 12 }}>
        {AVAILABLE_MODELS.map((model) => (
          <button key={model} onClick={() => addModel(text, model)} style={{ marginRight: 8 }}>
            Add {model}
          </button>
        ))}
      </div>

      <div style={{ marginTop: 24 }}>
        {Object.entries(modelStates).map(([model, state]) => (
          <div key={model} style={{ marginBottom: 16, border: "1px solid #333", padding: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <strong>{model}</strong>
              <button onClick={() => removeModel(model)}>Remove</button>
            </div>
            {state.status === "loading" && <p>Loading…</p>}
            {state.status === "success" && (
              <>
                <p>
                {state.data.token_count} tokens · {state.data.char_count} chars
                {state.data.estimated_cost_usd !== null && ` · $${state.data.estimated_cost_usd.toFixed(6)}`}
</p>
                <TokenizedText tokens={state.data.tokens} model={model} />
              </>
            )}
            {state.status === "unsupported_model" && <p style={{ color: "red" }}>{state.error.detail}</p>}
            {state.status === "validation_error" && <p style={{ color: "red" }}>Invalid request.</p>}
            {state.status === "network_error" && <p style={{ color: "red" }}>{state.message}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;