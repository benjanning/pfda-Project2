const { test, expect } = require("@playwright/test");
const fs = require("node:fs");
const base = process.env.EXPLORER_BASE || "http://127.0.0.1:8091/";
const lines = fs
  .readFileSync("results/holdout-predictions.csv", "utf8")
  .trim()
  .split(/\r?\n/);
const header = lines.shift().split(",");
const source = lines.map((line) =>
  Object.fromEntries(
    line.split(",").map((value, i) => [header[i], Number(value)]),
  ),
);
const names = [...header.slice(2), "Training-majority baseline"];
for (const row of source) row[names.at(-1)] = 0; // Independently verified training majority; no holdout selection.
const label = (n) => (n === 1 ? "Malignant (1)" : "Benign (0)");
const outcome = (y, p) =>
  y === p ? "Correct" : y === 1 ? "False negative" : "False positive";
const group = (row, a, b) =>
  row[a] !== row.actual
    ? row[b] !== row.actual
      ? "both_wrong"
      : "only_a_wrong"
    : row[b] !== row.actual
      ? "only_b_wrong"
      : "both_correct";
for (const width of [1440, 390])
  test(`paired saved row evidence at ${width}px`, async ({ page }) => {
    test.setTimeout(60000);
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
    await page.goto(base);
    await expect(page.locator("#results")).toBeVisible();
    for (const a of names)
      for (const b of names) {
        await page.getByLabel("Approach A", { exact: true }).selectOption(a);
        await page.getByLabel("Approach B", { exact: true }).selectOption(b);
        for (const key of [
          "both_wrong",
          "only_a_wrong",
          "only_b_wrong",
          "both_correct",
        ]) {
          const expected = source.filter((row) => group(row, a, b) === key);
          await expect(page.locator(`[data-group="${key}"] strong`)).toHaveText(
            String(expected.length),
          );
          await page
            .getByLabel("Inspect source rows", { exact: true })
            .selectOption(key);
          await expect(page.locator("#paired-rows tr")).toHaveCount(
            expected.length,
          );
          expect(
            await page
              .locator("#paired-rows tr")
              .evaluateAll((rows) =>
                rows.map((row) =>
                  [...row.cells].map((cell) => cell.textContent),
                ),
              ),
          ).toEqual(
            expected.map((row) => [
              String(row.row_index),
              label(row.actual),
              label(row[a]),
              outcome(row.actual, row[a]),
              label(row[b]),
              outcome(row.actual, row[b]),
            ]),
          );
          await expect(page.locator("#paired-empty")).toBeVisible({
            visible: expected.length === 0,
          });
        }
      }
    await page
      .getByLabel("Approach A", { exact: true })
      .selectOption("Logistic regression");
    await page
      .getByLabel("Approach B", { exact: true })
      .selectOption("Random forest");
    await page
      .getByLabel("Inspect source rows", { exact: true })
      .selectOption("both_wrong");
    await expect(page.locator("#paired-status")).toContainText(
      "A makes 10 errors; B makes 10. 4 errors are shared.",
    );
    await page.getByLabel("Approach B", { exact: true }).focus();
    await page.keyboard.press("Home");
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("Tab");
    await expect(page.getByLabel("Approach B", { exact: true })).toHaveValue(
      "Decision tree",
    );
    await page
      .getByLabel("Approach B", { exact: true })
      .selectOption("Random forest");
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    fs.mkdirSync("verification/screenshots", { recursive: true });
    await page
      .locator(".paired")
      .evaluate((el) =>
        el.scrollIntoView({ block: "start", behavior: "instant" }),
      );
    await page.screenshot({
      path: `verification/screenshots/paired-${width}.jpg`,
      type: "jpeg",
      quality: 85,
    });
    await page
      .locator(".paired")
      .screenshot({
        path: `verification/screenshots/paired-detail-${width}.jpg`,
        type: "jpeg",
        quality: 85,
      });
    expect(errors).toEqual([]);
  });
