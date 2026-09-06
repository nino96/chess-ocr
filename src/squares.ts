import { LABELS } from "./contract.ts";
const MODEL_LABELS = [
  "empty",
  "K",
  "Q",
  "R",
  "B",
  "N",
  "P",
  "k",
  "q",
  "r",
  "b",
  "n",
  "p",
];
export function imageSquares(probs: ArrayLike<number>) {
  if (probs.length !== 832) throw new Error("Invalid FENShot output shape");
  return Array.from({ length: 64 }, (_, i) => {
    const tile = (7 - Math.floor(i / 8)) * 8 + (i % 8);
    const probabilities = LABELS.map(
      (label) => probs[tile * 13 + MODEL_LABELS.indexOf(label)]!,
    );
    const confidence = Math.max(...probabilities);
    return {
      label: LABELS[probabilities.indexOf(confidence)]!,
      probabilities,
      uncertain: confidence < 0.7,
    };
  });
}
