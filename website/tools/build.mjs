import { cp, mkdir, readdir } from "node:fs/promises";
import { resolve } from "node:path";

// Only this explicit public directory is ever included in the deployment.
// Research outputs, demonstrations, manuscripts and credentials stay outside it.
const root = resolve(import.meta.dirname, "..");
await mkdir(resolve(root, "dist"), { recursive: true });
for (const entry of await readdir(resolve(root, "public"))) {
  await cp(resolve(root, "public", entry), resolve(root, "dist", entry), {
    recursive: true,
  });
}
console.log("Built website/dist from website/public. No runtime dependencies.");
