import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

rmSync("dist", { recursive: true, force: true });
function compile(config) {
  const result = spawnSync(process.execPath, ["node_modules/typescript/bin/tsc", "-p", config], { stdio: "inherit" });
  if (result.status !== 0) throw new Error(`TypeScript build failed (${result.status ?? result.error})`);
}
compile("tsconfig.json");
// .cts gives CommonJS output and matching .d.cts declarations under NodeNext.
const temporary = mkdtempSync(join(tmpdir(), "basal-client-build-"));
try {
  for (const name of ["index", "questions"]) {
    const source = readFileSync(`src/${name}.ts`, "utf8").replace(/(from\s+["']\.\/[^"']+)\.js(["'])/g, "$1.cjs$2");
    writeFileSync(join(temporary, `${name}.cts`), source);
  }
  const config = JSON.parse(readFileSync("tsconfig.json", "utf8"));
  config.compilerOptions.rootDir = temporary;
  config.compilerOptions.outDir = resolve("dist/cjs");
  config.include = [join(temporary, "*.cts")];
  const configPath = join(temporary, "tsconfig.json");
  writeFileSync(configPath, JSON.stringify(config));
  compile(configPath);
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
