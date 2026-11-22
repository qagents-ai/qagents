// Run Devvi's CLI on a Node older than its engines.node (>= 24.9.0), for the devvi_mock tests.
//
//   node --import <file URL of this file> <devvi-harness>/apps/cli/lib/bin.js --profile acp ...
//
// Devvi starts two entry points only when `import.meta.main` is true, which Node added in
// 24.2: the CLI (apps/cli/lib/bin.js) and the private subprocess runner that every shell
// tool call goes through on Windows (packages/subprocess/subprocess-local/lib/runner.js).
// On an older Node both load and exit 0 without doing anything. This preload registers a
// module hook that, for Devvi's modules only:
//
// * replaces `import.meta.main` with a check of the module against process.argv[1], and
// * starts the subprocess runner with this same preload (`--import`), because Devvi
//   removes NODE_* variables (NODE_OPTIONS) from the runner's environment.
//
// test_devvi_mock.py uses it only when the Node it finds is older than 24.9.0, and says so
// in the test report. Nothing is written to disk; the Devvi checkout is read only.
import { realpathSync } from "node:fs";
import { register } from "node:module";
import { fileURLToPath } from "node:url";
import { isMainThread } from "node:worker_threads";

const SELF = import.meta.url;
const MAIN_CHECK = "globalThis.__qagentsDevviIsMain(import.meta.url)";
const RUNNER = 'return [process.execPath, fileURLToPath(import.meta.resolve("@deepseek-ai/dsh-subprocess-local/runner"))]';
const RUNNER_WITH_PRELOAD = `return [process.execPath, "--import", ${JSON.stringify(SELF)}, fileURLToPath(import.meta.resolve("@deepseek-ai/dsh-subprocess-local/runner"))]`;

function samePath(url, path) {
  try {
    const a = realpathSync(fileURLToPath(url));
    const b = realpathSync(path);
    return process.platform === "win32" ? a.toLowerCase() === b.toLowerCase() : a === b;
  } catch {
    return false;
  }
}

if (isMainThread) {
  globalThis.__qagentsDevviIsMain = (url) => process.argv[1] !== undefined && samePath(url, process.argv[1]);
  register(SELF);
}

/** Module hook (runs in Node's hooks thread): rewrite Devvi's ESM sources as described above. */
export async function load(url, context, nextLoad) {
  const result = await nextLoad(url, context);
  if (result.format !== "module" || !url.startsWith("file:") || url.includes("/node_modules/.pnpm/")) {
    return result;
  }
  const original = typeof result.source === "string" ? result.source : new TextDecoder().decode(result.source);
  if (!original.includes("import.meta.main") && !original.includes(RUNNER)) return result;
  const source = original.replaceAll("import.meta.main", MAIN_CHECK).replace(RUNNER, RUNNER_WITH_PRELOAD);
  return { ...result, source, shortCircuit: true };
}
