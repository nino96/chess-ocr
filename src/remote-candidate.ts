import {
  CANDIDATE_CLASSIFIER_MAX_BYTES,
  CANDIDATE_DETECTOR_MAX_BYTES,
  CANDIDATE_MANIFEST_MAX_BYTES,
  loadCandidateBytes,
  parseCandidateManifestBytes,
  type CandidateConfig,
} from "./candidate.ts";
import {
  REMOTE_CANDIDATE_ENDPOINTS,
  REMOTE_CANDIDATE_REQUEST_HEADER,
  REMOTE_CANDIDATE_REQUEST_VALUE,
} from "./remote-candidate-api.ts";

type RemoteCandidateRole = keyof typeof REMOTE_CANDIDATE_ENDPOINTS;

async function boundedResponse(
  role: RemoteCandidateRole,
  maximum: number,
  signal: AbortSignal,
  expected?: number,
): Promise<Uint8Array> {
  const response = await fetch(REMOTE_CANDIDATE_ENDPOINTS[role], {
    method: "GET",
    headers: {
      [REMOTE_CANDIDATE_REQUEST_HEADER]: REMOTE_CANDIDATE_REQUEST_VALUE,
    },
    credentials: "same-origin",
    cache: "no-store",
    redirect: "error",
    signal,
  });
  if (!response.ok) {
    await response.body?.cancel().catch(() => undefined);
    throw new Error(
      response.status === 401 || response.status === 403
        ? "The configured candidate session is unavailable. Reload the demo and try again."
        : "The configured remote candidate is unavailable or stale.",
    );
  }
  const declared = Number(response.headers.get("content-length"));
  if (
    !Number.isSafeInteger(declared) ||
    declared <= 0 ||
    declared > maximum ||
    (expected !== undefined && declared !== expected)
  ) {
    await response.body?.cancel().catch(() => undefined);
    throw new Error("Remote candidate response is outside its size bound");
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Remote candidate response is unavailable");
  const bytes = new Uint8Array(declared);
  let offset = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.byteLength > bytes.byteLength)
        throw new Error("Remote candidate response size changed");
      bytes.set(value, offset);
      offset += value.byteLength;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
  if (offset !== bytes.byteLength)
    throw new Error("Remote candidate response size changed");
  return bytes;
}

export async function loadConfiguredRemoteCandidate(
  signal: AbortSignal,
): Promise<CandidateConfig> {
  const controller = new AbortController();
  const abort = (): void => controller.abort(signal.reason);
  signal.addEventListener("abort", abort, { once: true });
  if (signal.aborted) abort();
  try {
    const manifestBytes = await boundedResponse(
      "manifest",
      CANDIDATE_MANIFEST_MAX_BYTES,
      controller.signal,
    );
    const manifest = parseCandidateManifestBytes(manifestBytes);
    let models: [Uint8Array, Uint8Array];
    try {
      models = await Promise.all([
        boundedResponse(
          "classifier",
          CANDIDATE_CLASSIFIER_MAX_BYTES,
          controller.signal,
          manifest.classifier.bytes,
        ),
        boundedResponse(
          "detector",
          CANDIDATE_DETECTOR_MAX_BYTES,
          controller.signal,
          manifest.detector.bytes,
        ),
      ]);
    } catch (error) {
      controller.abort(error);
      throw error;
    }
    return await loadCandidateBytes(manifestBytes, models[0], models[1]);
  } finally {
    signal.removeEventListener("abort", abort);
  }
}
