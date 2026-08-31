// src/components/TokenizedText.tsx
import "./TokenizedText.css";
import type { components } from "../api/schema";

type TokenStringResponse = components["schemas"]["TokenStringResponse"];

export function TokenizedText({ tokens, model }: { tokens: TokenStringResponse[]; model: string }) {
  return (
    <div role="group" aria-label={`Token boundaries for ${model}, ${tokens.length} tokens`} style={{ fontFamily: "monospace", lineHeight: 1.8 }}>
      {tokens.map((token, i) => {
        const isFragment = token.text === null;
        return (
          <span
            key={i}
            className={`token-span token-slot-${i % 8}${isFragment ? " token-fragment" : ""}`}
            title={isFragment ? `token_id: ${token.token_id} (byte fragment)` : `token_id: ${token.token_id}`}
          >
            {isFragment ? "·" : token.text}
          </span>
        );
      })}
    </div>
  );
}