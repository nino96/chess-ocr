import {
  type Board,
  type Label,
  type Orientation,
  type Result,
  resultSchema,
} from "./contract.ts";

/** Corrections belong to an input/selection generation, never to an inference request. */
export class Editor {
  requestId: string | null = null;
  board: Board | null = null;
  private edits = new Map<number, Label>();
  private orientation: Orientation | null = null;
  reset(board: Board | null = null): void {
    this.requestId = null;
    this.board = board;
    this.edits.clear();
    this.orientation = null;
  }
  begin(requestId: string): void {
    this.requestId = requestId;
  }
  invalidate(): void {
    this.requestId = null;
  }
  apply(value: unknown): boolean {
    const result: Result = resultSchema.parse(value);
    if (result.requestId !== this.requestId) return false;
    const incoming = result.boards[0];
    if (!incoming) return false;
    if (
      this.board &&
      this.edits.size &&
      JSON.stringify(this.board.corners) !== JSON.stringify(incoming.corners)
    )
      return false;
    this.board = structuredClone(incoming);
    for (const [index, label] of this.edits)
      this.board.squares[index]!.label = label;
    if (this.orientation !== null) {
      this.board.orientation = this.orientation;
      this.board.orientationEvidence = "user";
    }
    return true;
  }
  edit(index: number, label: Label): void {
    if (!Number.isInteger(index) || index < 0 || index > 63 || !this.board)
      throw new Error("Invalid square");
    this.edits.set(index, label);
    this.board.squares[index]!.label = label;
  }
  orient(orientation: Orientation): void {
    this.orientation = orientation;
    if (this.board) {
      this.board.orientation = orientation;
      this.board.orientationEvidence = "user";
    }
  }
  isEdited(index: number): boolean {
    return this.edits.has(index);
  }
  export(): object {
    return {
      schema: "chess-ocr-editor/1",
      board: this.board,
      corrections: Object.fromEntries(this.edits),
      // FEN's five unobservable state fields are deliberately absent.
      placement: this.placement(),
    };
  }
  placement(): string | null {
    if (
      !this.board ||
      this.board.orientation === "unknown" ||
      this.board.squares.some((s) => s.label === null)
    )
      return null;
    const labels = this.board.squares.map((s) => s.label!);
    if (this.board.orientation === "black-bottom") labels.reverse();
    return Array.from({ length: 8 }, (_, row) => {
      let out = "",
        empty = 0;
      for (const label of labels.slice(row * 8, row * 8 + 8)) {
        if (label === "empty") empty++;
        else {
          if (empty) out += empty;
          empty = 0;
          out += label;
        }
      }
      return out + (empty || "");
    }).join("/");
  }
}
