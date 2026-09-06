import {
  type Request,
  type Result,
  LIMITS,
  requestSchema,
  resultSchema,
} from "./contract.ts";
export interface WorkerLike {
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(message: unknown, transfer: Transferable[]): void;
  terminate(): void;
}
export class RecognitionClient {
  private active: {
    worker: WorkerLike;
    reject: (reason: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  } | null = null;
  private worker: WorkerLike | null = null;
  private factory: () => WorkerLike;
  private timeoutMs: number;
  constructor(factory: () => WorkerLike, timeoutMs: number = LIMITS.timeoutMs) {
    if (
      !Number.isFinite(timeoutMs) ||
      timeoutMs <= 0 ||
      timeoutMs > LIMITS.timeoutMs
    )
      throw new Error("Invalid timeout bound");
    this.factory = factory;
    this.timeoutMs = timeoutMs;
  }
  cancel(): void {
    this.worker?.terminate();
    this.worker = null;
    if (!this.active) return;
    const job = this.active;
    this.active = null;
    clearTimeout(job.timer);
    job.reject(new Error("Recognition cancelled"));
  }
  recognize(request: Request, rgba: Uint8ClampedArray): Promise<Result> {
    requestSchema.parse(request);
    if (rgba.length !== request.image.width * request.image.height * 4)
      throw new Error("Invalid raster length");
    if (this.active) this.cancel();
    return new Promise((resolve, reject) => {
      const worker = (this.worker ??= this.factory());
      const finish = (error: Error | null, result?: Result): void => {
        if (this.active?.worker !== worker) return;
        clearTimeout(this.active.timer);
        this.active = null;
        if (error || result?.status === "error") {
          worker.terminate();
          this.worker = null;
        }
        if (error) reject(error);
        else resolve(result!);
      };
      const timer = setTimeout(
        () => finish(new Error("Recognition timed out")),
        this.timeoutMs,
      );
      this.active = { worker, reject, timer };
      worker.onerror = () => finish(new Error("Recognition worker failed"));
      worker.onmessage = (event) => {
        try {
          const result = resultSchema.parse(event.data);
          if (result.requestId !== request.requestId) return;
          if (
            result.image.width !== request.image.width ||
            result.image.height !== request.image.height
          )
            throw new Error("Image mismatch");
          finish(null, result);
        } catch {
          finish(new Error("Invalid recognition response"));
        }
      };
      try {
        worker.postMessage({ request, rgba }, [rgba.buffer as ArrayBuffer]);
      } catch {
        finish(new Error("Could not start recognition"));
      }
    });
  }
}
