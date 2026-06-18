window.SeedUi = (() => {
    function normalizeState(state) {
        return String(state || "").replace("-", "_");
    }

    function stateLabel(state) {
        return normalizeState(state).replace("_", "-");
    }

    function renderModelOptions(models, selectedId = "best_model") {
        const select = document.getElementById("model-select");
        const confusionSelect = document.getElementById("confusion-model-select");
        const options = models
            .map((model) => `<option value="${model.id}" ${model.id === selectedId ? "selected" : ""}>${model.name}</option>`)
            .join("");
        select.innerHTML = options;
        confusionSelect.innerHTML = options;
    }

    function renderModelCards(models) {
        document.getElementById("model-cards").innerHTML = models
            .map((model) => `
                <article class="model-card">
                    <h3>${model.name}</h3>
                    <dl>
                        <div><dt>Architecture</dt><dd>${model.architecture}</dd></div>
                        <div><dt>Input</dt><dd>${model.inputSize}</dd></div>
                        <div><dt>Classes</dt><dd>${model.classes}</dd></div>
                        <div><dt>Parameters</dt><dd>${model.parameters}</dd></div>
                        <div><dt>Epochs</dt><dd>${model.epochs}</dd></div>
                    </dl>
                    <span class="badge ${model.badge}">${model.status}</span>
                </article>
            `)
            .join("");
    }

    function renderMetricsTable(models) {
        document.getElementById("metrics-table-body").innerHTML = models
            .map((model) => `
                <tr>
                    <td><strong>${model.name}</strong></td>
                    <td>${model.task || "--"}</td>
                    <td>${model.primaryMetric || model.accuracy || "--"}</td>
                    <td>${model.secondaryMetric || model.f1 || "--"}</td>
                    <td>${model.epochs || "--"}</td>
                    <td>${model.notes}</td>
                </tr>
            `)
            .join("");
    }

    function renderSelectedMetrics(metrics) {
        document.getElementById("selected-metrics").innerHTML = metrics
            .map((metric) => `
                <div class="metric-card">
                    <span>${metric.name}</span>
                    <strong>${metric.value}</strong>
                    <p class="muted">${metric.change}</p>
                </div>
            `)
            .join("");
    }

    function renderFrameSummary(items) {
        document.getElementById("frame-summary-body").innerHTML = items
            .map((item) => `
                <tr>
                    <td>${item.frame}</td>
                    <td class="state-germinated">${item.germinated}</td>
                    <td class="state-non-germinated">${item.nonGerminated}</td>
                    <td class="state-transition">${item.transition}</td>
                    <td><strong>${item.percentage}</strong></td>
                </tr>
            `)
            .join("");
    }

    function renderBars(elementId, values, className = "") {
        const maxValue = Math.max(...values, 1);
        document.getElementById(elementId).innerHTML = values
            .map((value) => `<span class="bar ${className}" style="height:${Math.max(6, (value / maxValue) * 100)}%"></span>`)
            .join("");
    }

    function renderConfusionMatrix(matrix) {
        const labels = ["Germinated", "Non-germinated"];
        let html = '<div class="confusion-grid">';
        html += '<div class="confusion-cell"></div>';
        labels.forEach((label) => {
            html += `<div class="confusion-cell">${label}</div>`;
        });
        labels.forEach((label, rowIndex) => {
            html += `<div class="confusion-cell">${label}</div>`;
            labels.forEach((_, colIndex) => {
                const correct = rowIndex === colIndex ? "correct" : "";
                html += `<div class="confusion-cell ${correct}">${matrix[rowIndex][colIndex]}</div>`;
            });
        });
        html += "</div>";
        document.getElementById("confusion-matrix").innerHTML = html;
    }

    function renderBoundingBoxes(detections) {
        document.getElementById("bounding-boxes").innerHTML = detections
            .map((seed) => {
                const state = stateLabel(seed.state);
                return `
                    <button class="bounding-box ${state}" data-seed-id="${seed.id}"
                        style="left:${seed.bbox.x}%;top:${seed.bbox.y}%;width:${seed.bbox.width}%;height:${seed.bbox.height}%"
                        title="Seed ${seed.id}">
                    </button>
                    <span class="seed-label"
                        style="left:${seed.bbox.x}%;top:calc(${seed.bbox.y}% - 24px)">
                        ${state} ${seed.confidence}%
                    </span>
                `;
            })
            .join("");
    }

    function updateSummary(detections) {
        const total = detections.length;
        const germinated = detections.filter((seed) => normalizeState(seed.state) === "germinated").length;
        const nonGerminated = detections.filter((seed) => normalizeState(seed.state) === "non_germinated").length;
        const transition = detections.filter((seed) => normalizeState(seed.state) === "transition").length;
        const percentage = total ? Math.round((germinated / total) * 100) : 0;

        document.getElementById("total-seeds").textContent = total || "--";
        document.getElementById("germinated-count").textContent = germinated || "0";
        document.getElementById("non-germinated-count").textContent = nonGerminated || "0";
        document.getElementById("transition-count").textContent = transition || "0";
        document.getElementById("germination-percentage").textContent = total ? `${percentage}%` : "--";
        document.getElementById("germination-progress").style.width = `${percentage}%`;
    }

    function updateCurrentPrediction(seed) {
        document.getElementById("current-seed-id").textContent = seed.id;
        document.getElementById("current-seed-state").textContent = stateLabel(seed.state);
        document.getElementById("current-seed-confidence").textContent = `${seed.confidence}%`;
        const bbox = seed.pixelBbox || seed.bbox;
        document.getElementById("current-seed-bbox").textContent =
            `[x:${bbox.x}, y:${bbox.y}, w:${bbox.width}, h:${bbox.height}]`;
    }

    function resetCurrentPrediction() {
        document.getElementById("current-seed-id").textContent = "--";
        document.getElementById("current-seed-state").textContent = "--";
        document.getElementById("current-seed-confidence").textContent = "--";
        document.getElementById("current-seed-bbox").textContent = "--";
    }

    function syncOverlayToImage() {
        const content = document.getElementById("preview-content");
        const image = document.getElementById("preview-image");
        const overlay = document.getElementById("bounding-boxes");
        if (!content || !image || !overlay || !image.naturalWidth || !image.naturalHeight) {
            return;
        }

        const rect = content.getBoundingClientRect();
        const imageRatio = image.naturalWidth / image.naturalHeight;
        const containerRatio = rect.width / rect.height;
        let width;
        let height;
        let left;
        let top;

        if (containerRatio > imageRatio) {
            height = rect.height;
            width = height * imageRatio;
            left = (rect.width - width) / 2;
            top = 0;
        } else {
            width = rect.width;
            height = width / imageRatio;
            left = 0;
            top = (rect.height - height) / 2;
        }

        overlay.style.left = `${left}px`;
        overlay.style.top = `${top}px`;
        overlay.style.width = `${width}px`;
        overlay.style.height = `${height}px`;
    }

    return {
        renderModelOptions,
        renderModelCards,
        renderMetricsTable,
        renderSelectedMetrics,
        renderFrameSummary,
        renderBars,
        renderConfusionMatrix,
        renderBoundingBoxes,
        updateSummary,
        updateCurrentPrediction,
        resetCurrentPrediction,
        syncOverlayToImage,
    };
})();
