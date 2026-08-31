import { useReducer, useRef } from "react";
import { analyzeText } from "../api/client";
import type { components } from "../api/schema";

type ModelState =
  | { status: "loading" }
  | { status: "success"; data: components["schemas"]["AnalyzeSuccessResponse"] }
  | { status: "unsupported_model"; error: components["schemas"]["UnsupportedModelErrorResponse"] }
  | { status: "validation_error"; error: components["schemas"]["HTTPValidationError"] }
  | { status: "network_error"; message: string };

type Action =
  | { type: "MODEL_LOADING"; model: string }
  | { type: "MODEL_RESOLVED"; model: string; state: ModelState }
  | { type: "MODEL_REMOVED"; model: string };

function reducer(state: Record<string, ModelState>, action: Action): Record<string, ModelState> {
  switch (action.type) {
    case "MODEL_LOADING":
      return { ...state, [action.model]: { status: "loading" } };
    case "MODEL_RESOLVED":
      return { ...state, [action.model]: action.state };
    case "MODEL_REMOVED": {
      const { [action.model]: _removed, ...rest } = state;
      return rest;
    }
  }
}

export function useModelComparison() {
  const [modelStates, dispatch] = useReducer(reducer, {});
  const controllers = useRef<Map<string, AbortController>>(new Map());

  async function addModel(text: string, model: string) {
    const controller = new AbortController();
    controllers.current.set(model, controller);
    dispatch({ type: "MODEL_LOADING", model });

    const result = await analyzeText({ text, model }, controller.signal);
    if (result.kind === "aborted") return; // removed mid-flight — stay silent, UI already updated

    const newState: ModelState =
      result.kind === "success" ? { status: "success", data: result.data }
      : result.kind === "unsupported_model" ? { status: "unsupported_model", error: result.error }
      : result.kind === "validation_error" ? { status: "validation_error", error: result.error }
      : { status: "network_error", message: result.message };

    dispatch({ type: "MODEL_RESOLVED", model, state: newState });
    controllers.current.delete(model);
  }

  function removeModel(model: string) {
    controllers.current.get(model)?.abort();
    controllers.current.delete(model);
    dispatch({ type: "MODEL_REMOVED", model });
  }

  return { modelStates, addModel, removeModel };
}