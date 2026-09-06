import { test } from "node:test";
import assert from "node:assert/strict";
import { RecognitionClient, type WorkerLike } from "../src/client.ts";
import { VERSION } from "../src/contract.ts";
class FakeWorker implements WorkerLike {
  onmessage: WorkerLike["onmessage"] = null;
  onerror: WorkerLike["onerror"] = null;
  ended = false;
  postMessage() {}
  terminate() {
    this.ended = true;
  }
  send(data: unknown) {
    this.onmessage?.({ data } as MessageEvent);
  }
}
const request = {
  schema: VERSION,
  requestId: "a",
  image: { width: 1, height: 1 },
  selection: null,
};
const response = {
  schema: VERSION,
  requestId: "a",
  image: request.image,
  status: "unsupported",
  boards: [],
  warnings: ["none"],
  model: null,
  preprocessing: "test",
  timings: { totalMs: 1 },
};
test("cancellation terminates CPU work, rejects promise and allows recovery", async () => {
  const workers: FakeWorker[] = [];
  const c = new RecognitionClient(() => {
    const w = new FakeWorker();
    workers.push(w);
    return w;
  });
  const p = c.recognize(request, new Uint8ClampedArray(4));
  c.cancel();
  await assert.rejects(p, /cancelled/);
  assert.ok(workers[0]!.ended);
  const next = c.recognize(request, new Uint8ClampedArray(4));
  workers[0]!.send(response);
  workers[1]!.send(response);
  assert.equal((await next).status, "unsupported");
});
test("timeout kills worker, stale output ignored, corrupt output rejected", async () => {
  const w = new FakeWorker();
  const c = new RecognitionClient(() => w, 10);
  const p = c.recognize(request, new Uint8ClampedArray(4));
  w.send({ ...response, requestId: "old" });
  await assert.rejects(p, /timed out/);
  assert.ok(w.ended);
  const w2 = new FakeWorker();
  const c2 = new RecognitionClient(() => w2);
  const p2 = c2.recognize(request, new Uint8ClampedArray(4));
  w2.send({ secret: "invalid" });
  await assert.rejects(p2, /Invalid recognition/);
});
test("bad input never starts a worker and worker errors reject", async () => {
  const w = new FakeWorker();
  const c = new RecognitionClient(() => w);
  assert.throws(() => c.recognize(request, new Uint8ClampedArray(3)), /raster/);
  const p = c.recognize(request, new Uint8ClampedArray(4));
  w.onerror?.({} as ErrorEvent);
  await assert.rejects(p, /worker failed/);
});
test("library timeout configuration cannot remove the work ceiling", () => {
  for (const value of [0, -1, Infinity, 60_000])
    assert.throws(
      () => new RecognitionClient(() => new FakeWorker(), value),
      /timeout/,
    );
});
