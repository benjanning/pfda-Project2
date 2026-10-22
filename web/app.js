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
  const groupNames = {
    both_wrong: "Both wrong",
    only_a_wrong: "Only A wrong",
    only_b_wrong: "Only B wrong",
    both_correct: "Both correct",
  };
  for (const id of ["approach-a", "approach-b"])
    for (const model of data.models) {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.name;
      $(id).append(option);
    }
  $("approach-a").value = "Logistic regression";
  $("approach-b").value = "Random forest";
  const rowsByIndex = new Map(
    data.observations.map((row) => [row.row_index, row]),
  );
  const className = (value) => (value === 1 ? "Malignant (1)" : "Benign (0)");
  const outcome = (actual, prediction) =>
    actual === prediction
      ? "Correct"
      : actual === 1
        ? "False negative"
        : "False positive";
  function renderPair() {
    const a = $("approach-a").value,
      b = $("approach-b").value;
    const pair = data.pairs.find((pair) => pair.a === a && pair.b === b);
    const counts = Object.fromEntries(
      Object.entries(pair.groups).map(([key, rows]) => [key, rows.length]),
    );
    $("paired-context").textContent =
      `A: ${a}. B: ${b}. Each count uses the same ${data.holdout_rows} rows; the four groups form a complete partition.`;
    $("paired-counts").replaceChildren(
      ...Object.entries(groupNames).map(([key, title]) => {
        const card = document.createElement("div");
        card.dataset.group = key;
        const heading = document.createElement("h3");
        heading.textContent = title;
        const count = document.createElement("strong");
        count.textContent = counts[key];
        const denominator = document.createElement("span");
        denominator.textContent = `of ${data.holdout_rows} rows`;
        card.append(heading, count, denominator);
        return card;
      }),
    );
    const group = $("error-group").value;
    $("paired-status").textContent =
      `A makes ${counts.both_wrong + counts.only_a_wrong} errors; B makes ${counts.both_wrong + counts.only_b_wrong}. ${counts.both_wrong} errors are shared. Showing ${counts[group]} rows: ${groupNames[group].toLowerCase()}.`;
    $("prediction-a").textContent = `A: ${a}`;
    $("prediction-b").textContent = `B: ${b}`;
    $("paired-caption").textContent =
      `${groupNames[group]} · ${counts[group]} of ${data.holdout_rows} saved rows, in original holdout order.`;
    $("paired-empty").hidden = counts[group] > 0;
    $("paired-rows").replaceChildren(
      ...pair.groups[group].map((index) => {
        const row = rowsByIndex.get(index),
          tr = document.createElement("tr");
        tr.dataset.sourceIndex = index;
        [
          index,
          className(row.actual),
          className(row.predictions[a]),
          outcome(row.actual, row.predictions[a]),
          className(row.predictions[b]),
          outcome(row.actual, row.predictions[b]),
        ].forEach((value, i) => {
          const cell = document.createElement(i === 0 ? "th" : "td");
          if (i === 0) cell.scope = "row";
          cell.textContent = value;
          tr.append(cell);
        });
        return tr;
      }),
    );
  }
  for (const id of ["approach-a", "approach-b", "error-group"])
    $(id).addEventListener("change", renderPair);
  renderPair();
  $("model").addEventListener("change", render);
  render();
  $("results").hidden = false;
}
start().catch(() => {
  $("failure").hidden = false;
});
