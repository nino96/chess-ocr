export const REMOTE_CANDIDATE_ENDPOINTS = {
  manifest: "/__ocr_candidate/manifest",
  classifier: "/__ocr_candidate/classifier",
  detector: "/__ocr_candidate/detector",
} as const;

export const REMOTE_CANDIDATE_REQUEST_HEADER = "x-chess-ocr-candidate";
export const REMOTE_CANDIDATE_REQUEST_VALUE = "load";
export const REMOTE_CANDIDATE_META = "chess-ocr-remote-candidate";
