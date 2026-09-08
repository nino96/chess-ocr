import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import {
  mkdir,
  mkdtemp,
  rm,
  symlink,
  truncate,
  writeFile,
} from "node:fs/promises";
import type {
  IncomingHttpHeaders,
  IncomingMessage,
  ServerResponse,
} from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, test } from "node:test";
import type { Connect } from "vite";
import { LABELS } from "../src/contract.ts";
import {
  REMOTE_CANDIDATE_ENDPOINTS,
  REMOTE_CANDIDATE_REQUEST_HEADER,
  REMOTE_CANDIDATE_REQUEST_VALUE,
} from "../src/remote-candidate-api.ts";
import {
  ConfiguredRemoteCandidate,
  createRemoteCandidateMiddleware,
  type DEFAULT_REMOTE_CANDIDATE_PATHS,
} from "../scripts/remote-candidate-server.ts";

type Paths = typeof DEFAULT_REMOTE_CANDIDATE_PATHS;
const roots: string[] = [];
afterEach(async () => {
  await Promise.all(
    roots.splice(0).map((root) => rm(root, { recursive: true })),
  );
});
const sha256 = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex");

async function candidateFixture(): Promise<{
  root: string;
  paths: Paths;
  classifier: Buffer;
}> {
  const root = await mkdtemp(join(tmpdir(), "chess-ocr-remote-candidate-"));
  roots.push(root);
  const paths: Paths = {
    manifest: "work/candidates/bundle.json",
    classifier: "work/training/run/classifier.onnx",
    detector: "work/training/run/detector.onnx",
  };
  await mkdir(join(root, "work/candidates"), { recursive: true });
  await mkdir(join(root, "work/training/run"), { recursive: true });
  const classifier = Buffer.from("classifier fixture");
  const detector = Buffer.from("detector fixture");
  const manifest = {
    schema: "chess-ocr-candidate-bundle/3",
    name: "server fixture",
    version: "1",
    qualification: "synthetic-development-only",
    preprocessing: "yolox-rgb-imagenet-v2",
    classifier: {
      sha256: sha256(classifier),
      bytes: classifier.byteLength,
      input: "tiles",
      output: "logits",
      inputShape: ["squares", 3, 96, 96],
      outputShape: ["squares", 13],
      labels: LABELS,
    },
    detector: {
      sha256: sha256(detector),
      bytes: detector.byteLength,
      input: "images",
      output: "predictions",
      inputShape: [1, 3, 416, 416],
      proposalScoreThreshold: 0.01,
      calibratedAcceptanceThreshold: 1,
      nmsIou: 0.65,
    },
    refinement: {
      id: "nine-line-grid-refiner-v1",
      implementationSha256: "b".repeat(64),
      regionExpansion: 0.12,
      outputSize: 768,
      classifierTileSize: 96,
      maxCandidates: 4,
    },
  };
  await Promise.all([
    writeFile(join(root, paths.manifest), JSON.stringify(manifest)),
    writeFile(join(root, paths.classifier), classifier),
    writeFile(join(root, paths.detector), detector),
  ]);
  return { root, paths, classifier };
}

const acceptTensors = async (): Promise<void> => undefined;

test("configured bundle validates roots, links, bounds, manifest and hashes", async () => {
  const fixture = await candidateFixture();
  await ConfiguredRemoteCandidate.create(
    fixture.root,
    fixture.paths,
    acceptTensors,
  );
  await assert.rejects(
    ConfiguredRemoteCandidate.create(
      fixture.root,
      { ...fixture.paths, classifier: "../outside.onnx" },
      acceptTensors,
    ),
    /managed root/,
  );
  await rm(join(fixture.root, fixture.paths.classifier));
  await symlink(
    join(fixture.root, fixture.paths.detector),
    join(fixture.root, fixture.paths.classifier),
  );
  await assert.rejects(
    ConfiguredRemoteCandidate.create(
      fixture.root,
      fixture.paths,
      acceptTensors,
    ),
    /unsafe link/,
  );
});

test("configured bundle rejects oversize and invalid manifests", async () => {
  const fixture = await candidateFixture();
  await truncate(join(fixture.root, fixture.paths.manifest), 64 * 1024 + 1);
  await assert.rejects(
    ConfiguredRemoteCandidate.create(
      fixture.root,
      fixture.paths,
      acceptTensors,
    ),
    /size bound/,
  );
  await writeFile(join(fixture.root, fixture.paths.manifest), "{}");
  await assert.rejects(
    ConfiguredRemoteCandidate.create(
      fixture.root,
      fixture.paths,
      acceptTensors,
    ),
  );
});

type Reply = {
  status: number;
  headers: Record<string, string>;
  body: Buffer;
};

async function request(
  middleware: Connect.NextHandleFunction,
  url: string,
  options: { method?: string; headers?: IncomingHttpHeaders } = {},
): Promise<Reply> {
  return await new Promise((resolveReply) => {
    const responseHeaders: Record<string, string> = {};
    const responseEvents = new EventEmitter();
    const response = Object.assign(responseEvents, {
      statusCode: 200,
      setHeader(name: string, value: string | number | readonly string[]) {
        responseHeaders[name.toLowerCase()] = Array.isArray(value)
          ? value.join(", ")
          : String(value);
        return this;
      },
      end(body?: Uint8Array | string) {
        responseEvents.emit("finish");
        resolveReply({
          status: this.statusCode,
          headers: responseHeaders,
          body: body === undefined ? Buffer.alloc(0) : Buffer.from(body),
        });
        return this;
      },
    });
    const incoming = Object.assign(new EventEmitter(), {
      method: options.method ?? "GET",
      url,
      headers: { host: "127.0.0.1:4173", ...options.headers },
    }) as IncomingMessage;
    middleware(incoming, response as unknown as ServerResponse, () => {
      response.statusCode = url === "/" ? 200 : 404;
      response.end(url === "/" ? "<!doctype html>" : undefined);
    });
  });
}

test("candidate endpoint requires its active same-origin session and fixed role", async () => {
  const fixture = await candidateFixture();
  const bundle = await ConfiguredRemoteCandidate.create(
    fixture.root,
    fixture.paths,
    acceptTensors,
  );
  const middleware = createRemoteCandidateMiddleware(bundle);
  assert.equal(
    (await request(middleware, REMOTE_CANDIDATE_ENDPOINTS.classifier)).status,
    401,
  );
  const page = await request(middleware, "/");
  const session = page.headers["set-cookie"]?.split(";", 1)[0];
  assert.ok(session);
  const headers = {
    cookie: session,
    [REMOTE_CANDIDATE_REQUEST_HEADER]: REMOTE_CANDIDATE_REQUEST_VALUE,
    "sec-fetch-site": "same-origin",
  };
  const response = await request(
    middleware,
    REMOTE_CANDIDATE_ENDPOINTS.classifier,
    { headers },
  );
  assert.equal(response.status, 200);
  assert.equal(response.headers["cache-control"], "no-store");
  assert.equal(response.headers["access-control-allow-origin"], undefined);
  assert.deepEqual(response.body, fixture.classifier);
  assert.equal(
    (
      await request(middleware, REMOTE_CANDIDATE_ENDPOINTS.manifest, {
        headers: { ...headers, "sec-fetch-site": "cross-site" },
      })
    ).status,
    403,
  );
  assert.equal(
    (
      await request(
        middleware,
        REMOTE_CANDIDATE_ENDPOINTS.manifest + "?path=../x",
        { headers },
      )
    ).status,
    404,
  );
  assert.equal(
    (await request(middleware, "/__ocr_candidate/../../private", { headers }))
      .status,
    404,
  );
});

test("candidate endpoint denies bundle bytes changed after server startup", async () => {
  const fixture = await candidateFixture();
  const bundle = await ConfiguredRemoteCandidate.create(
    fixture.root,
    fixture.paths,
    acceptTensors,
  );
  const middleware = createRemoteCandidateMiddleware(bundle);
  const page = await request(middleware, "/");
  const session = page.headers["set-cookie"]?.split(";", 1)[0];
  assert.ok(session);
  await writeFile(
    join(fixture.root, fixture.paths.classifier),
    Buffer.from("Classifier fixture"),
  );
  const response = await request(
    middleware,
    REMOTE_CANDIDATE_ENDPOINTS.classifier,
    {
      headers: {
        cookie: session,
        [REMOTE_CANDIDATE_REQUEST_HEADER]: REMOTE_CANDIDATE_REQUEST_VALUE,
        "sec-fetch-site": "same-origin",
      },
    },
  );
  assert.equal(response.status, 409);
  assert.equal(response.body.byteLength, 0);
});

test("candidate endpoint expires sessions and rejects non-loopback or wrong-origin requests", async () => {
  const fixture = await candidateFixture();
  const bundle = await ConfiguredRemoteCandidate.create(
    fixture.root,
    fixture.paths,
    acceptTensors,
  );
  let time = 100;
  const middleware = createRemoteCandidateMiddleware(
    bundle,
    () => time,
    () => "d".repeat(64),
  );
  const page = await request(middleware, "/");
  assert.match(page.headers["set-cookie"] ?? "", /HttpOnly; SameSite=Strict/);
  const session = page.headers["set-cookie"]?.split(";", 1)[0];
  assert.ok(session);
  const validHeaders = {
    cookie: session,
    [REMOTE_CANDIDATE_REQUEST_HEADER]: REMOTE_CANDIDATE_REQUEST_VALUE,
    "sec-fetch-site": "same-origin",
  };
  assert.equal(
    (
      await request(middleware, REMOTE_CANDIDATE_ENDPOINTS.manifest, {
        method: "POST",
        headers: validHeaders,
      })
    ).status,
    405,
  );
  assert.equal(
    (
      await request(middleware, REMOTE_CANDIDATE_ENDPOINTS.manifest, {
        headers: { ...validHeaders, origin: "http://example.test" },
      })
    ).status,
    403,
  );
  assert.equal(
    (
      await request(middleware, REMOTE_CANDIDATE_ENDPOINTS.manifest, {
        headers: { ...validHeaders, host: "example.test" },
      })
    ).status,
    404,
  );
  time += 8 * 60 * 60 * 1000 + 1;
  assert.equal(
    (
      await request(middleware, REMOTE_CANDIDATE_ENDPOINTS.manifest, {
        headers: validHeaders,
      })
    ).status,
    401,
  );
});
