(() => {
    const data = window.SeedData;
    const ui = window.SeedUi;

    let selectedFile = null;
    let playbackTimer = null;
    let activeDetections = data.detections;
    let availableModels = data.models;

    function formatMetric(name, value) {
        if (value === undefined || value === null || value === "") {
            return "--";
        }
        const numericValue = Number(value);
        const displayValue = Number.isFinite(numericValue)
            ? `${(numericValue * 100).toFixed(2)}%`
            : value;
        return `${name} ${displayValue}`;
    }

    function initialize() {
        ui.renderModelOptions(data.models, "faster_rcnn_scratch");
        ui.renderDemoOptions(data.demoImages);
        ui.renderModelCards(data.models);
        ui.renderMetricsTable(data.models);
        ui.renderSelectedMetrics(data.selectedMetrics);
        ui.renderFrameSummary(data.frameSummary);
        ui.renderBars("germination-chart", data.germinationCurve, "germinated");
        ui.renderBars("loss-chart", data.lossCurve);
        ui.renderBars("f1-chart", data.f1Curve, "germinated");
        ui.renderConfusionMatrix(data.confusionMatrix);
        ui.renderBoundingBoxes(activeDetections);
        ui.updateSummary([]);
        bindEvents();
        updateActiveModelChip();
        tryLoadBackendModels();
        window.addEventListener("resize", ui.syncOverlayToImage);
    }

    async function tryLoadBackendModels() {
        try {
            const registry = await window.SeedApi.getModels();
            const registryModels = registry.models.map((model) => ({
                id: model.id,
                name: model.display_name || model.id,
                task: model.task_type || "--",
                taskType: model.task_type,
                architecture: model.architecture,
                inputSize: `${model.input_size || 224} x ${model.input_size || 224}`,
                classes: `${model.num_classes || 2} classes`,
                parameters: "registry",
                epochs: model.epochs || "registry",
                status: model.status,
                badge: model.id === registry.selected_model ? "best" : model.status === "ready" ? "ready" : "pending",
                primaryMetric: formatMetric(model.primary_metric || "Metric", model.primary_metric_value),
                secondaryMetric: formatMetric(model.secondary_metric || "Secondary", model.secondary_metric_value),
                accuracy: "--",
                precision: "--",
                recall: "--",
                f1: "--",
                notes: model.notes || "",
            }));
            const selectedDetectionModel = registry.selected_detection_model || registry.selected_model;
            if (registryModels.length) {
                availableModels = registryModels;
                ui.renderModelOptions(registryModels, selectedDetectionModel);
                ui.renderModelCards(registryModels);
                ui.renderMetricsTable(registryModels);
                document.getElementById("active-model-chip").textContent =
                    registryModels.find((model) => model.id === selectedDetectionModel)?.name || "Best Model";
            }
        } catch (error) {
            console.info("Backend API not available; using static dashboard data.");
        }
    }

    function bindEvents() {
        document.querySelectorAll(".tab-button").forEach((button) => {
            button.addEventListener("click", () => switchTab(button.dataset.tab));
        });

        const dropzone = document.getElementById("dropzone");
        const fileInput = document.getElementById("file-input");

        document.getElementById("choose-file").addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", (event) => {
            if (event.target.files.length) {
                handleFile(event.target.files[0]);
            }
        });

        dropzone.addEventListener("dragover", (event) => {
            event.preventDefault();
            dropzone.classList.add("dragover");
        });
        dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
        dropzone.addEventListener("drop", (event) => {
            event.preventDefault();
            dropzone.classList.remove("dragover");
            if (event.dataTransfer.files.length) {
                handleFile(event.dataTransfer.files[0]);
            }
        });

        document.getElementById("load-demo-image").addEventListener("click", loadDemoImage);
        document.getElementById("clear-input").addEventListener("click", clearInput);
        document.getElementById("analyze-btn").addEventListener("click", analyzeInput);
        document.getElementById("model-select").addEventListener("change", updateActiveModelChip);
        document.getElementById("export-report").addEventListener("click", () => alert("Report export endpoint is reserved for backend integration."));
        document.getElementById("download-results").addEventListener("click", () => alert("Annotated output download is reserved for backend integration."));

        document.getElementById("frame-slider").addEventListener("input", (event) => {
            document.getElementById("current-frame").textContent = event.target.value;
        });

        document.getElementById("play-pause").addEventListener("click", togglePlayback);

        document.getElementById("bounding-boxes").addEventListener("click", (event) => {
            const box = event.target.closest(".bounding-box");
            if (!box) return;
            const seed = activeDetections.find((item) => String(item.id) === box.dataset.seedId);
            if (seed) {
                ui.updateCurrentPrediction(seed);
            }
        });
    }

    function updateActiveModelChip() {
        const selectedModel = getSelectedModel();
        document.getElementById("active-model-chip").textContent = selectedModel?.name || "Selected Model";
    }

    function getSelectedModel() {
        const modelId = document.getElementById("model-select").value;
        return availableModels.find((model) => model.id === modelId) || data.models.find((model) => model.id === modelId);
    }

    async function loadDemoImage() {
        const demoId = document.getElementById("demo-select").value;
        const demo = data.demoImages.find((item) => item.id === demoId);
        if (!demo) return;

        try {
            const response = await fetch(demo.url);
            if (!response.ok) {
                throw new Error(`Unable to load demo image: ${response.status}`);
            }
            const blob = await response.blob();
            const file = new File([blob], demo.fileName, { type: blob.type || "image/jpeg" });
            file.demoSource = demo;
            handleFile(file);
            showPreview([]);
        } catch (error) {
            alert("Cannot load demo image. Run the backend and open http://127.0.0.1:8000/.");
        }
    }

    function switchTab(tab) {
        document.querySelectorAll(".tab-button").forEach((button) => {
            button.classList.toggle("active", button.dataset.tab === tab);
        });
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
        document.getElementById(`panel-${tab}`).classList.add("active");
    }

    function handleFile(file) {
        selectedFile = file;
        const fileType = file.type ? file.type.split("/")[0] : file.demo ? "video" : "unknown";
        document.getElementById("file-name").textContent = file.name || "selected_input";
        document.getElementById("file-type").textContent = fileType;
        document.getElementById("file-resolution").textContent = file.demoSource
            ? `${file.demoSource.species} / ${file.demoSource.sequence}`
            : fileType === "image" ? "uploaded image" : "sequence/video";
        document.getElementById("file-frames").textContent = fileType === "image" ? "1" : "90";
        document.getElementById("file-card").classList.remove("hidden");
    }

    function normalizeApiDetections(detections) {
        return detections.map((item, index) => ({
            id: item.id || index + 1,
            state: item.display_state || item.class_name,
            confidence: Math.round(Number(item.score || 0) * 100),
            bbox: item.bbox_percent,
            pixelBbox: item.bbox,
        }));
    }

    async function analyzeInput() {
        if (!selectedFile) {
            alert("Choose a file or load a test sequence first.");
            return;
        }
        document.getElementById("preview-processing").classList.remove("hidden");
        document.getElementById("preview-empty").classList.add("hidden");

        let detections = data.detections;
        if (selectedFile && selectedFile.type && selectedFile.type.startsWith("image/") && !selectedFile.demo) {
            try {
                const modelId = document.getElementById("model-select").value;
                const selectedModel = getSelectedModel();
                if (selectedModel?.taskType === "crop_classification") {
                    const prediction = await window.SeedApi.predictCrop(selectedFile, modelId);
                    detections = [
                        {
                            id: 1,
                            state: prediction.display_state,
                            confidence: Math.round(Number(prediction.confidence) * 100),
                            bbox: { x: 5, y: 5, width: 90, height: 90 },
                            pixelBbox: { x: 0, y: 0, width: "full", height: "full" },
                        },
                    ];
                } else {
                    const prediction = await window.SeedApi.predictDetections(selectedFile, modelId, 0.5);
                    detections = normalizeApiDetections(prediction.detections);
                }
            } catch (error) {
                console.info("Prediction API not available; using simulated detections.");
            }
        }

        setTimeout(() => {
            document.getElementById("preview-processing").classList.add("hidden");
            showPreview(detections);
        }, 500);
    }

    function showPreview(detections = data.detections) {
        activeDetections = detections;
        const previewImage = document.getElementById("preview-image");
        previewImage.onload = () => {
            ui.syncOverlayToImage();
            ui.renderBoundingBoxes(activeDetections);
        };
        if (selectedFile && selectedFile.type && selectedFile.type.startsWith("image/")) {
            previewImage.src = URL.createObjectURL(selectedFile);
        } else {
            previewImage.src = data.previewImage;
        }
        document.getElementById("preview-empty").classList.add("hidden");
        document.getElementById("preview-content").classList.remove("hidden");
        document.getElementById("timeline-controls").classList.add("hidden");
        document.getElementById("current-frame").textContent = "1";
        document.getElementById("total-frames").textContent = "1";
        const slider = document.getElementById("frame-slider");
        slider.max = "1";
        slider.value = "1";
        ui.renderBoundingBoxes(activeDetections);
        requestAnimationFrame(ui.syncOverlayToImage);
        ui.updateSummary(activeDetections);
        if (activeDetections.length) {
            ui.updateCurrentPrediction(activeDetections[0]);
        } else {
            ui.resetCurrentPrediction();
        }
    }

    function clearInput() {
        selectedFile = null;
        activeDetections = data.detections;
        document.getElementById("file-card").classList.add("hidden");
        document.getElementById("preview-empty").classList.remove("hidden");
        document.getElementById("preview-content").classList.add("hidden");
        document.getElementById("timeline-controls").classList.add("hidden");
        ui.updateSummary([]);
        ui.resetCurrentPrediction();
        stopPlayback();
    }

    function togglePlayback() {
        const icon = document.querySelector("#play-pause i");
        if (playbackTimer) {
            stopPlayback();
            return;
        }
        icon.classList.remove("fa-play");
        icon.classList.add("fa-pause");
        const slider = document.getElementById("frame-slider");
        playbackTimer = setInterval(() => {
            const next = Number(slider.value) + 1;
            if (next > Number(slider.max)) {
                stopPlayback();
                return;
            }
            slider.value = String(next);
            document.getElementById("current-frame").textContent = slider.value;
        }, 180);
    }

    function stopPlayback() {
        if (playbackTimer) {
            clearInterval(playbackTimer);
            playbackTimer = null;
        }
        const icon = document.querySelector("#play-pause i");
        icon.classList.remove("fa-pause");
        icon.classList.add("fa-play");
    }

    document.addEventListener("DOMContentLoaded", initialize);
})();
