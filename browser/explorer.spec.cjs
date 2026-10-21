const { test, expect } = require("@playwright/test");
const fs = require("node:fs");
const data = require("../docs/data.json");
const base = process.env.EXPLORER_BASE || "http://127.0.0.1:8091/";
const pct = (v) => (v === null ? "N/A" : (v * 100).toFixed(1) + "%");
for (const width of [1440, 390])
  test(`saved models, baseline and accessible layout at ${width}px`, async ({
    page,
  }) => {
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
    await page.goto(base);
    await expect(page.locator("#results")).toBeVisible();
    await expect(page.getByRole("heading", { level: 1 })).toContainText(
      "Different mistakes.",
    );
    await expect(page.locator("#comparison tr")).toHaveCount(4);
    for (const model of data.models) {
      await page.getByLabel("Classifier or baseline").selectOption(model.name);
      await expect(page.locator("#selected-title")).toHaveText(model.name);
      for (const [key, n] of Object.entries({
        tn: model.counts[0][0],
        fp: model.counts[0][1],
        fn: model.counts[1][0],
        tp: model.counts[1][1],
      }))
        await expect(page.locator("#" + key)).toHaveText(String(n));
      const metricCards = page.locator(".metric");
      for (const [i, key] of [
        "recall",
        "precision",
        "f1",
        "balanced_accuracy",
      ].entries())
        await expect(metricCards.nth(i).locator("strong")).toHaveText(
          pct(model.metrics[key]),
        );
      await expect(page.locator("#selection-status")).toContainText(
        `${model.counts[1][0]} missed of 64`,
      );
      await expect(page.locator(".comparison .selected th")).toHaveText(
        model.name,
      );
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
    }
    await expect(page.locator("#cv")).toContainText(
      "No cross-validation result",
    );
    await expect(page.locator("#baseline-story")).toContainText("62.6%");
    await page.getByLabel("Classifier or baseline").focus();
    await page.keyboard.press("Home");
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("Tab");
    await expect(page.locator("#selected-title")).toHaveText("Decision tree");
    await page
      .getByLabel("Classifier or baseline")
      .selectOption("Logistic regression");
    fs.mkdirSync("verification/screenshots", { recursive: true });
    await page.screenshot({
      path: `verification/screenshots/explorer-${width}.jpg`,
      type: "jpeg",
      quality: 85,
      fullPage: true,
    });
    if (width === 1440) {
      await page
        .locator(".selection")
        .evaluate((el) =>
          el.scrollIntoView({ block: "start", behavior: "instant" }),
        );
      await page.screenshot({
        path: "verification/screenshots/explorer.jpg",
        type: "jpeg",
        quality: 85,
      });
    }
    await page
      .getByLabel("Classifier or baseline")
      .selectOption("Training-majority baseline");
    await page.screenshot({
      path: `verification/screenshots/baseline-${width}.jpg`,
      type: "jpeg",
      quality: 85,
      fullPage: true,
    });
    if (width === 1440) {
      await page
        .locator(".selection")
        .evaluate((el) =>
          el.scrollIntoView({ block: "start", behavior: "instant" }),
        );
      await page.screenshot({
        path: "verification/screenshots/baseline.jpg",
        type: "jpeg",
        quality: 85,
      });
    }
    expect(errors).toEqual([]);
  });
test("loading failure provides a visible recovery state", async ({ page }) => {
  await page.route("**/data.json", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );
  await page.goto(base);
  await expect(page.getByRole("alert")).toContainText("Please reload");
  await expect(page.locator("#results")).toBeHidden();
});
