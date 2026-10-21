"use strict";
const $ = (id) => document.getElementById(id);
const percentage = (value) =>
  value === null ? "N/A" : (value * 100).toFixed(1) + "%";
function metric(title, value, explanation) {
  const panel = document.createElement("div");
  panel.className = "metric";
  const heading = document.createElement("h4");
  heading.textContent = title;
  const number = document.createElement("strong");
  number.textContent = percentage(value);
  const note = document.createElement("p");
  note.textContent = explanation;
  panel.append(heading, number, note);
  return panel;
}
async function start() {
  const response = await fetch("data.json");
  if (!response.ok) throw Error("Results unavailable");
  const data = await response.json();
  if (data.schema_version !== 1 || data.models.length !== 4)
    throw Error("Unexpected results");
  for (const model of data.models) {
    const option = document.createElement("option");
    option.textContent = model.name;
    option.value = model.name;
    $("model").append(option);
    const row = document.createElement("tr");
    row.dataset.model = model.name;
    const title = document.createElement("th");
    title.scope = "row";
    title.textContent = model.name;
    row.append(title);
    for (const key of [
      "accuracy",
      "balanced_accuracy",
      "precision",
      "recall",
      "f1",
    ]) {
      const cell = document.createElement("td");
      cell.textContent = percentage(model.metrics[key]);
      row.append(cell);
    }
    const missed = document.createElement("td");
    missed.textContent = `${model.counts[1][0]} / ${data.holdout_classes["1"]}`;
    row.append(missed);
    $("comparison").append(row);
  }
  const baseline = data.models.at(-1);
  $("baseline-story").textContent =
    `The training set has ${data.training_classes["0"]} benign and ${data.training_classes["1"]} malignant labels. Predicting its majority class (${data.baseline_label === 0 ? "benign" : "malignant"}) gives ${percentage(baseline.metrics.accuracy)} accuracy on this holdout, but misses ${baseline.counts[1][0]} of ${data.holdout_classes["1"]} malignant-labelled examples. A majority guess is useful context, not a useful classifier.`;
  $("receipt").textContent =
    `Source revision: ${data.source_revision}. Dataset SHA-256: ${data.dataset_sha256}. Saved predictions SHA-256: ${data.predictions_sha256}.`;
  function render() {
    const model = data.models.find((item) => item.name === $("model").value);
    const [[tn, fp], [fn, tp]] = model.counts;
    const m = model.metrics;
    $("selected-title").textContent = model.name;
    $("matrix-caption").textContent = model.name;
    for (const [key, count] of Object.entries({ tn, fp, fn, tp }))
      $(key).textContent = count;
    $("metrics").replaceChildren(
      metric(
        "Malignant recall",
        m.recall,
        `${tp} / ${tp + fn} actual malignant labels were found.`,
      ),
      metric(
        "Malignant precision",
        m.precision,
        tp + fp
          ? `${tp} / ${tp + fp} malignant predictions matched actual labels.`
          : "No malignant predictions: precision is undefined.",
      ),
      metric(
        "Malignant F1",
        m.f1,
        "Harmonic mean of precision and recall; zero when all positive labels are missed.",
      ),
      metric(
        "Balanced accuracy",
        m.balanced_accuracy,
        `Average of malignant recall and benign recall (${tn} / ${tn + fp}).`,
      ),
    );
    $("error-story").textContent =
      `${fn} of ${tp + fn} malignant-labelled examples were predicted benign (${percentage(m.false_negative_rate)} missed). ${fp} of ${tn + fp} benign-labelled examples were predicted malignant (${percentage(m.false_positive_rate)} false positives).`;
    $("cv").textContent = model.cv
      ? `Training-only five-fold CV balanced accuracy: ${percentage(model.cv.mean)} mean, ${(model.cv.fold_sd * 100).toFixed(2)} percentage-point fold SD. Fold SD is not a confidence interval.`
      : "This baseline is chosen from all training labels. No cross-validation result is claimed for it.";
    for (const row of $("comparison").rows)
      row.classList.toggle("selected", row.dataset.model === model.name);
    $("selection-status").textContent =
      `${model.name}: ${fn} missed of ${tp + fn} actual malignant labels; ${fp} false positives of ${tn + fp} actual benign labels.`;
  }
  $("model").addEventListener("change", render);
  render();
  $("results").hidden = false;
}
start().catch(() => {
  $("failure").hidden = false;
});
