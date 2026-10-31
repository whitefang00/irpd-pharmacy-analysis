import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

declare const process: { env: Record<string, string | undefined> };

// In dev, /api/* goes to the FastAPI server without the prefix.
// Default http://127.0.0.1:8000; set IRPD_API to point elsewhere.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: process.env.IRPD_API ?? "http://127.0.0.1:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
