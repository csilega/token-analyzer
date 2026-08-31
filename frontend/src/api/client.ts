import type { components } from "./schema";

const BASE_URL = "http://localhost:8000";

type AnalyzeResult =
  | { kind: "success"; data: components["schemas"]["AnalyzeSuccessResponse"] }
  | { kind: "unsupported_model"; error: components["schemas"]["UnsupportedModelErrorResponse"] }
  | { kind: "validation_error"; error: components["schemas"]["HTTPValidationError"] }
  | { kind: "network_error"; message: string }
  | { kind: "aborted" };

type CompareResult =
  | { kind: "success"; data: components["schemas"]["CompareResponse"] }
  | { kind: "validation_error"; error: components["schemas"]["HTTPValidationError"] }
  | { kind: "network_error"; message: string };


export async function analyzeText(
  body: components["schemas"]["AnalyzeRequest"],
  signal?: AbortSignal
): Promise<AnalyzeResult> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return { kind: "aborted" };
    }
    // fetch() only throws here for genuine network failure — DNS, connection refused, offline
    return { kind: "network_error", message: err instanceof Error ? err.message : "Unknown network error" };
  }

  if (response.status === 200) {
    const data = (await response.json()) as components["schemas"]["AnalyzeSuccessResponse"];
    return { kind: "success", data };
  }
  if (response.status === 404) {
    const error = (await response.json()) as components["schemas"]["UnsupportedModelErrorResponse"];
    return { kind: "unsupported_model", error };
  }
  // 422 — the only remaining documented case
  const error = (await response.json()) as components["schemas"]["HTTPValidationError"];
  return { kind: "validation_error", error };
}

export async function compareModels(
  body: components["schemas"]["CompareRequest"]
): Promise<CompareResult> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    // fetch() only throws here for genuine network failure — DNS, connection refused, offline
    return { kind: "network_error", message: err instanceof Error ? err.message : "Unknown network error" };
  }

  if (response.status === 200) {
    const data = (await response.json()) as components["schemas"]["CompareResponse"];
    return { kind: "success", data };
  }
  // 422 — the only remaining documented case
  const error = (await response.json()) as components["schemas"]["HTTPValidationError"];
  return { kind: "validation_error", error };
}