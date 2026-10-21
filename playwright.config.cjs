const { defineConfig } = require("@playwright/test");
module.exports = defineConfig({
  testDir: "browser",
  workers: 1,
  retries: 0,
  reporter: [
    ["list"],
    ["json", { outputFile: "verification/browser-results.json" }],
  ],
  use: { viewport: { width: 1440, height: 1000 }, reducedMotion: "reduce" },
  webServer: process.env.EXPLORER_BASE
    ? undefined
    : {
        command: "python -m http.server 8091 --bind 127.0.0.1 --directory docs",
        url: "http://127.0.0.1:8091",
        reuseExistingServer: false,
      },
});
