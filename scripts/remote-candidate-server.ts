import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath, stat } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { isAbsolute, relative, resolve, sep } from "node:path";
import * as ort from "onnxruntime-web/wasm";
import type { Connect, Plugin } from "vite";
import {
  CANDIDATE_CLASSIFIER_MAX_BYTES,
  CANDIDATE_DETECTOR_MAX_BYTES,
  CANDIDATE_MANIFEST_MAX_BYTES,
  parseCandidateManifestBytes,
  type CandidateManifest,
} from "../src/candidate.ts";
import {
  REMOTE_CANDIDATE_ENDPOINTS,
  REMOTE_CANDIDATE_META,
  REMOTE_CANDIDATE_REQUEST_HEADER,
  REMOTE_CANDIDATE_REQUEST_VALUE,
} from "../src/remote-candidate-api.ts";

type CandidateRole = keyof typeof REMOTE_CANDIDATE_ENDPOINTS;
type CandidatePaths = Record<CandidateRole, string>;
type ManagedRoots = Record<CandidateRole, string>;
type TensorVerifier = (
  manifest: CandidateManifest,
  classifier: Uint8Array,
  detector: Uint8Array,
) => Promise<void>;

export const DEFAULT_REMOTE_CANDIDATE_PATHS: CandidatePaths = {
  manifest: "work/candidates/synthetic-bootstrap-v2-detector-v3.json",
  classifier:
    "work/training/synthetic-bootstrap-v2-detector/classifier/selected.onnx",
  detector:
    "work/training/synthetic-bootstrap-v2-detector/detector/selected.onnx",
};

const LIMITS: Record<CandidateRole, number> = {
  manifest: CANDIDATE_MANIFEST_MAX_BYTES,
  classifier: CANDIDATE_CLASSIFIER_MAX_BYTES,
  detector: CANDIDATE_DETECTOR_MAX_BYTES,
};
const SESSION_COOKIE = "chess_ocr_candidate_session";
const SESSION_LIFETIME_MS = 8 * 60 * 60 * 1000;
const MAX_SESSIONS = 16;

function digest(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function within(root: string, path: string): boolean {
  const location = relative(root, path);
  return (
    location !== ".." &&
    !location.startsWith(`..${sep}`) &&
    !isAbsolute(location)
  );
}

async function rejectSymlinkComponents(
  repository: string,
  path: string,
): Promise<void> {
  const location = relative(repository, path);
  if (!within(repository, path) || !location)
    throw new Error("Candidate path is outside its managed root");
  let cursor = repository;
  for (const part of location.split(sep)) {
    cursor = resolve(cursor, part);
    if ((await lstat(cursor)).isSymbolicLink())
      throw new Error("Candidate path contains an unsafe link");
  }
}

async function safeBytes(
  repository: string,
  configuredPath: string,
  managedRoot: string,
  maximum: number,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const repositoryReal = await realpath(repository);
  const root = resolve(repositoryReal, managedRoot);
  const path = resolve(repositoryReal, configuredPath);
  if (!within(root, path))
    throw new Error("Candidate path is outside its managed root");
  await rejectSymlinkComponents(repositoryReal, path);
  const rootReal = await realpath(root);
  const pathReal = await realpath(path);
  if (!within(rootReal, pathReal))
    throw new Error("Candidate path is outside its managed root");
  const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const before = await handle.stat();
    if (!before.isFile() || before.size <= 0 || before.size > maximum)
      throw new Error("Candidate file is outside its size bound");
    if (process.platform !== "linux")
      throw new Error(
        "Configured remote candidates require Linux descriptor containment",
      );
    const openedPath = await realpath(`/proc/self/fd/${handle.fd}`);
    if (!within(rootReal, openedPath))
      throw new Error("Opened candidate file is outside its managed root");
    await rejectSymlinkComponents(repositoryReal, path);
    const pathAfterOpen = await realpath(path);
    const named = await stat(path);
    if (
      !within(rootReal, pathAfterOpen) ||
      named.dev !== before.dev ||
      named.ino !== before.ino
    )
      throw new Error("Candidate path changed while it was being verified");
    const bytes = Buffer.alloc(before.size + 1);
    let offset = 0;
    while (offset < bytes.byteLength) {
      signal?.throwIfAborted();
      const length = Math.min(1024 * 1024, bytes.byteLength - offset);
      const read = await handle.read(bytes, offset, length, null);
      if (read.bytesRead === 0) break;
      offset += read.bytesRead;
    }
    const after = await handle.stat();
    if (
      offset !== before.size ||
      after.size !== before.size ||
      after.mtimeMs !== before.mtimeMs
    )
      throw new Error("Candidate file changed while it was being verified");
    return bytes.subarray(0, offset);
  } finally {
    await handle.close();
  }
}

function validateBundleBytes(
  manifestBytes: Uint8Array,
  classifier: Uint8Array,
  detector: Uint8Array,
): CandidateManifest {
  const manifest = parseCandidateManifestBytes(manifestBytes);
  if (
    classifier.byteLength !== manifest.classifier.bytes ||
    digest(classifier) !== manifest.classifier.sha256
  )
    throw new Error("Configured classifier does not match its manifest");
  if (
    detector.byteLength !== manifest.detector.bytes ||
    digest(detector) !== manifest.detector.sha256
  )
    throw new Error("Configured detector does not match its manifest");
  return manifest;
}

function assertTensor(
  metadata: readonly ort.InferenceSession.ValueMetadata[],
  name: string,
  shape: readonly (number | string)[],
): void {
  if (metadata.length !== 1) throw new Error("Candidate tensor count mismatch");
  const value = metadata[0]!;
  if (
    value.name !== name ||
    !value.isTensor ||
    value.type !== "float32" ||
    value.shape.length !== shape.length ||
    value.shape.some((dimension, index) => dimension !== shape[index])
  )
    throw new Error("Candidate tensor contract mismatch");
}

export async function verifyCandidateTensorContracts(
  repository: string,
  manifest: CandidateManifest,
  classifier: Uint8Array,
  detector: Uint8Array,
): Promise<void> {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths =
    resolve(repository, "node_modules/onnxruntime-web/dist") + sep;
  let classifierSession: ort.InferenceSession | null = null;
  let detectorSession: ort.InferenceSession | null = null;
  try {
    classifierSession = await ort.InferenceSession.create(classifier, {
      executionProviders: ["wasm"],
    });
    detectorSession = await ort.InferenceSession.create(detector, {
      executionProviders: ["wasm"],
    });
    assertTensor(
      classifierSession.inputMetadata,
      manifest.classifier.input,
      manifest.classifier.inputShape,
    );
    assertTensor(
      classifierSession.outputMetadata,
      manifest.classifier.output,
      manifest.classifier.outputShape,
    );
    assertTensor(
      detectorSession.inputMetadata,
      manifest.detector.input,
      manifest.detector.inputShape,
    );
    assertTensor(
      detectorSession.outputMetadata,
      manifest.detector.output,
      [1, 3549, 6],
    );
  } finally {
    await Promise.allSettled([
      classifierSession?.release(),
      detectorSession?.release(),
    ]);
  }
}

export class ConfiguredRemoteCandidate {
  readonly #repository: string;
  readonly #paths: CandidatePaths;
  readonly #roots: ManagedRoots;
  readonly #manifestDigest: string;

  private constructor(
    repository: string,
    paths: CandidatePaths,
    roots: ManagedRoots,
    manifestDigest: string,
  ) {
    this.#repository = repository;
    this.#paths = paths;
    this.#roots = roots;
    this.#manifestDigest = manifestDigest;
  }

  static async create(
    repository: string,
    paths: CandidatePaths,
    verifyTensors: TensorVerifier,
  ): Promise<ConfiguredRemoteCandidate> {
    const roots: ManagedRoots = {
      manifest: "work/candidates",
      classifier: "work/training",
      detector: "work/training",
    };
    const bytes: [Uint8Array, Uint8Array, Uint8Array] = await Promise.all([
      safeBytes(repository, paths.manifest, roots.manifest, LIMITS.manifest),
      safeBytes(
        repository,
        paths.classifier,
        roots.classifier,
        LIMITS.classifier,
      ),
      safeBytes(repository, paths.detector, roots.detector, LIMITS.detector),
    ]);
    const manifest = validateBundleBytes(bytes[0], bytes[1], bytes[2]);
    await verifyTensors(manifest, bytes[1], bytes[2]);
    return new ConfiguredRemoteCandidate(
      await realpath(repository),
      { ...paths },
      roots,
      digest(bytes[0]),
    );
  }

  async read(role: CandidateRole, signal?: AbortSignal): Promise<Uint8Array> {
    const manifestBytes = await safeBytes(
      this.#repository,
      this.#paths.manifest,
      this.#roots.manifest,
      LIMITS.manifest,
      signal,
    );
    const manifest = parseCandidateManifestBytes(manifestBytes);
    if (digest(manifestBytes) !== this.#manifestDigest)
      throw new Error("Configured candidate manifest changed after startup");
    if (role === "manifest") {
      const [classifier, detector] = await Promise.all([
        safeBytes(
          this.#repository,
          this.#paths.classifier,
          this.#roots.classifier,
          LIMITS.classifier,
          signal,
        ),
        safeBytes(
          this.#repository,
          this.#paths.detector,
          this.#roots.detector,
          LIMITS.detector,
          signal,
        ),
      ]);
      validateBundleBytes(manifestBytes, classifier, detector);
      return manifestBytes;
    }
    const model = await safeBytes(
      this.#repository,
      this.#paths[role],
      this.#roots[role],
      LIMITS[role],
      signal,
    );
    const expected = manifest[role];
    if (
      model.byteLength !== expected.bytes ||
      digest(model) !== expected.sha256
    )
      throw new Error(`Configured ${role} does not match its manifest`);
    return model;
  }
}

function localHost(request: IncomingMessage): boolean {
  const host = request.headers.host;
  if (!host) return false;
  try {
    const hostname = new URL(`http://${host}`).hostname;
    return (
      hostname === "127.0.0.1" ||
      hostname === "localhost" ||
      hostname === "[::1]"
    );
  } catch {
    return false;
  }
}

function cookie(request: IncomingMessage, name: string): string | undefined {
  for (const item of (request.headers.cookie ?? "").split(";")) {
    const [key, ...value] = item.trim().split("=");
    if (key === name) return value.join("=");
  }
  return undefined;
}

function same(value: string | undefined, expected: string): boolean {
  if (!value) return false;
  const left = Buffer.from(value);
  const right = Buffer.from(expected);
  return left.byteLength === right.byteLength && timingSafeEqual(left, right);
}

function harden(response: ServerResponse): void {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
}

function end(response: ServerResponse, status: number): void {
  response.statusCode = status;
  harden(response);
  response.setHeader("Content-Length", "0");
  response.end();
}

export function createRemoteCandidateMiddleware(
  bundle: ConfiguredRemoteCandidate,
  now: () => number = Date.now,
  sessionToken: () => string = () => randomBytes(32).toString("hex"),
): Connect.NextHandleFunction {
  const sessions = new Map<string, number>();
  let activeReads = 0;
  const prune = (reserve = false): void => {
    const current = now();
    for (const [token, expires] of sessions)
      if (expires <= current) sessions.delete(token);
    while (reserve && sessions.size >= MAX_SESSIONS)
      sessions.delete(sessions.keys().next().value as string);
  };
  return (request, response, next): void => {
    const parsed = new URL(request.url ?? "/", "http://localhost");
    if (
      request.method === "GET" &&
      (parsed.pathname === "/" || parsed.pathname === "/index.html") &&
      localHost(request)
    ) {
      prune(true);
      const token = sessionToken();
      sessions.set(token, now() + SESSION_LIFETIME_MS);
      response.setHeader(
        "Set-Cookie",
        `${SESSION_COOKIE}=${token}; HttpOnly; SameSite=Strict; Path=/`,
      );
      next();
      return;
    }
    const role = (
      Object.keys(REMOTE_CANDIDATE_ENDPOINTS) as CandidateRole[]
    ).find(
      (candidateRole) =>
        REMOTE_CANDIDATE_ENDPOINTS[candidateRole] === parsed.pathname,
    );
    if (!role) {
      next();
      return;
    }
    if (request.method !== "GET") {
      end(response, 405);
      return;
    }
    if (parsed.search || !localHost(request)) {
      end(response, 404);
      return;
    }
    prune();
    const token = cookie(request, SESSION_COOKIE);
    const active = token
      ? [...sessions].find(([candidate]) => same(token, candidate))
      : undefined;
    if (!active || active[1] <= now()) {
      end(response, 401);
      return;
    }
    const fetchSite = request.headers["sec-fetch-site"];
    const origin = request.headers.origin;
    const expectedOrigin = `http://${request.headers.host}`;
    if (
      request.headers[REMOTE_CANDIDATE_REQUEST_HEADER] !==
        REMOTE_CANDIDATE_REQUEST_VALUE ||
      (fetchSite !== undefined && fetchSite !== "same-origin") ||
      (origin !== undefined && origin !== expectedOrigin)
    ) {
      end(response, 403);
      return;
    }
    if (activeReads >= 2) {
      end(response, 429);
      return;
    }
    activeReads++;
    const controller = new AbortController();
    let released = false;
    let readSettled = false;
    let responseDone = false;
    const release = (): void => {
      if (released || !readSettled || !responseDone) return;
      released = true;
      activeReads--;
      request.off("aborted", abort);
      response.off("finish", finish);
      response.off("close", close);
    };
    const abort = (): void => {
      responseDone = true;
      controller.abort();
      release();
    };
    const finish = (): void => {
      responseDone = true;
      release();
    };
    const close = (): void => {
      abort();
    };
    request.once("aborted", abort);
    response.once("finish", finish);
    response.once("close", close);
    void bundle
      .read(role, controller.signal)
      .then((bytes) => {
        if (controller.signal.aborted) return;
        response.statusCode = 200;
        harden(response);
        response.setHeader(
          "Content-Type",
          role === "manifest" ? "application/json" : "application/octet-stream",
        );
        response.setHeader("Content-Length", String(bytes.byteLength));
        response.end(bytes);
      })
      .catch(() => {
        if (!controller.signal.aborted) end(response, 409);
      })
      .finally(() => {
        readSettled = true;
        release();
      });
  };
}

export async function createRemoteCandidatePlugin(
  repository: string,
): Promise<Plugin> {
  const bundle = await ConfiguredRemoteCandidate.create(
    repository,
    DEFAULT_REMOTE_CANDIDATE_PATHS,
    (manifest, classifier, detector) =>
      verifyCandidateTensorContracts(
        repository,
        manifest,
        classifier,
        detector,
      ),
  );
  const middleware = createRemoteCandidateMiddleware(bundle);
  return {
    name: "configured-remote-candidate",
    configResolved(config) {
      const host = config.server.host;
      if (host !== "127.0.0.1" && host !== "localhost" && host !== "::1")
        throw new Error(
          "Configured remote candidates require an explicit loopback server host",
        );
    },
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    transformIndexHtml: {
      order: "pre",
      handler(_html, context) {
        if (!context.server) return;
        return [
          {
            tag: "meta",
            attrs: { name: REMOTE_CANDIDATE_META, content: "available" },
            injectTo: "head",
          },
        ];
      },
    },
  };
}
