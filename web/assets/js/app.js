(() => {
    const data = window.SeedData;
    const ui = window.SeedUi;
    const maxVideoFrames = 8;

    let selectedFile = null;
    let playbackTimer = null;
    let activeDetections = data.detections;
    let activeFrameResults = [];
    let availableModels = data.models.filter(isDetectionModel);

    function isDetectionModel(model) {
        return model?.taskType === "object_detection" || model?.task === "Object detection" || model?.task === "object_detection";
    }

    function getDetectionModels(models) {
        return models.filter(isDetectionModel);
    }

    function setProcessingMessage(message) {
        const node = document.getElementById("processing-message");
        if (node) {
            node.textContent = message;
        }
    }

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
        const detectionModels = getDetectionModels(data.models);
        availableModels = detectionModels;
        ui.renderModelOptions(detectionModels, "faster_rcnn_scratch", data.models);
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
            const detectionModels = getDetectionModels(registryModels);
            const selectedDetectionModel = registry.selected_detection_model || detectionModels[0]?.id || registry.selected_model;
            if (registryModels.length) {
                availableModels = detectionModels;
                ui.renderModelOptions(detectionModels, selectedDetectionModel, registryModels);
                ui.renderModelCards(registryModels);
                ui.renderMetricsTable(registryModels);
                document.getElementById("active-model-chip").textContent =
                    detectionModels.find((model) => model.id === selectedDetectionModel)?.name || "Detection Model";
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

        document.getElementById("clear-input").addEventListener("click", clearInput);
        document.getElementById("analyze-btn").addEventListener("click", analyzeInput);
        document.getElementById("model-select").addEventListener("change", updateActiveModelChip);
        document.getElementById("export-report").addEventListener("click", () => alert("Report export endpoint is reserved for backend integration."));
        document.getElementById("download-results").addEventListener("click", () => alert("Annotated output download is reserved for backend integration."));

        document.getElementById("frame-slider").addEventListener("input", (event) => {
            if (activeFrameResults.length > 1) {
                showVideoFrame(Number(event.target.value) - 1);
            } else {
                document.getElementById("current-frame").textContent = event.target.value;
            }
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
        return availableModels.find((model) => model.id === modelId);
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
        activeFrameResults = [];
        stopPlayback();
        const fileType = file.type ? file.type.split("/")[0] : "unknown";
        document.getElementById("file-name").textContent = file.name || "selected_input";
        document.getElementById("file-type").textContent = fileType;
        document.getElementById("file-resolution").textContent =
            fileType === "image" ? "uploaded image" : fileType === "video" ? "video frames" : "unsupported";
        document.getElementById("file-frames").textContent =
            fileType === "image" ? "1" : fileType === "video" ? `up to ${maxVideoFrames}` : "--";
        document.getElementById("file-card").classList.remove("hidden");
        ui.updateSummary([]);
        ui.resetCurrentPrediction();
    }

    function normalizeApiDetections(detections) {
        return detections.map((item, index) => ({
            id: item.id || index + 1,
            state: item.display_state || item.class_name,
            confidence: Math.round(Number(item.score || 0) * 100),
            bbox: item.bbox_percent || { x: 0, y: 0, width: 0, height: 0 },
            pixelBbox: item.bbox || null,
        }));
    }

    async function analyzeInput() {
        if (!selectedFile) {
            alert("Choose an image or video file first.");
            return;
        }

        const selectedModel = getSelectedModel();
        if (!selectedModel || selectedModel.taskType !== "object_detection") {
            alert("Monitoring only supports the Faster R-CNN detection model. Use Model Comparison to view the baseline.");
            return;
        }

        document.getElementById("preview-processing").classList.remove("hidden");
        document.getElementById("preview-empty").classList.add("hidden");

        const modelId = selectedModel.id;
        try {
            if (selectedFile.type.startsWith("image/")) {
                setProcessingMessage("Analyzing image with Faster R-CNN...");
                const prediction = await window.SeedApi.predictDetections(selectedFile, modelId, 0.5);
                showPreview(normalizeApiDetections(prediction.detections), URL.createObjectURL(selectedFile));
            } else if (selectedFile.type.startsWith("video/")) {
                setProcessingMessage("Sampling video frames...");
                const frames = await analyzeVideoFile(selectedFile, modelId);
                showVideoPreview(frames);
            } else {
                alert("Unsupported file type. Use an image or a short video.");
            }
        } catch (error) {
            console.error(error);
            alert("Cannot run real inference. Start the backend with: uvicorn src.webapp.api:app --reload");
            if (selectedFile.type.startsWith("image/")) {
                showPreview([], URL.createObjectURL(selectedFile));
            }
        } finally {
            document.getElementById("preview-processing").classList.add("hidden");
            setProcessingMessage("Analyzing frames...");
        }
    }

    async function analyzeVideoFile(file, modelId) {
        const frames = await extractVideoFrames(file);
        const results = [];
        for (const [index, frame] of frames.entries()) {
            setProcessingMessage(`Analyzing frame ${index + 1}/${frames.length}...`);
            const prediction = await window.SeedApi.predictDetections(frame.file, modelId, 0.5);
            results.push({
                ...frame,
                detections: normalizeApiDetections(prediction.detections),
            });
        }
        return results;
    }

    async function extractVideoFrames(file) {
        const video = await loadVideo(file);
        const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
        const frameCount = Math.min(maxVideoFrames, Math.max(1, Math.ceil(duration)));
        const canvas = document.createElement("canvas");
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 640;
        const context = canvas.getContext("2d");
        canvas.width = width;
        canvas.height = height;

        const frames = [];
        for (let index = 0; index < frameCount; index += 1) {
            const timestamp = Math.min(duration - 0.05, (duration * (index + 0.5)) / frameCount);
            await seekVideo(video, Math.max(0, timestamp));
            context.drawImage(video, 0, 0, width, height);
            const blob = await canvasToBlob(canvas);
            const name = `video_frame_${String(index + 1).padStart(3, "0")}.jpg`;
            frames.push({
                file: new File([blob], name, { type: "image/jpeg" }),
                imageUrl: URL.createObjectURL(blob),
                timestamp,
                width,
                height,
                detections: [],
            });
        }

        URL.revokeObjectURL(video.src);
        return frames;
    }

    function loadVideo(file) {
        return new Promise((resolve, reject) => {
            const video = document.createElement("video");
            video.preload = "metadata";
            video.muted = true;
            video.playsInline = true;
            video.src = URL.createObjectURL(file);
            video.onloadedmetadata = () => resolve(video);
            video.onerror = () => reject(new Error("Cannot load video metadata."));
        });
    }

    function seekVideo(video, time) {
        return new Promise((resolve, reject) => {
            if (Math.abs(video.currentTime - time) < 0.001) {
                requestAnimationFrame(resolve);
                return;
            }
            const cleanup = () => {
                video.removeEventListener("seeked", onSeeked);
                video.removeEventListener("error", onError);
            };
            const onSeeked = () => {
                cleanup();
                resolve();
            };
            const onError = () => {
                cleanup();
                reject(new Error("Cannot seek video frame."));
            };
            video.addEventListener("seeked", onSeeked);
            video.addEventListener("error", onError);
            video.currentTime = time;
        });
    }

    function canvasToBlob(canvas) {
        return new Promise((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error("Cannot export video frame."));
                }
            }, "image/jpeg", 0.92);
        });
    }

    function showPreview(detections = data.detections, imageUrl = null) {
        activeFrameResults = [];
        activeDetections = detections;
        const previewImage = document.getElementById("preview-image");
        previewImage.onload = () => {
            if (selectedFile?.type?.startsWith("image/")) {
                document.getElementById("file-resolution").textContent =
                    `${previewImage.naturalWidth} x ${previewImage.naturalHeight}`;
            }
            ui.syncOverlayToImage();
            ui.renderBoundingBoxes(activeDetections);
        };
        if (imageUrl) {
            previewImage.src = imageUrl;
        } else if (selectedFile && selectedFile.type && selectedFile.type.startsWith("image/")) {
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

    function showVideoPreview(frames) {
        activeFrameResults = frames;
        const slider = document.getElementById("frame-slider");
        slider.max = String(Math.max(frames.length, 1));
        slider.value = "1";
        document.getElementById("total-frames").textContent = String(frames.length);
        document.getElementById("timeline-controls").classList.toggle("hidden", frames.length <= 1);
        document.getElementById("file-frames").textContent = String(frames.length);
        if (frames[0]) {
            document.getElementById("file-resolution").textContent = `${frames[0].width} x ${frames[0].height}`;
        }
        showVideoFrame(0);
    }

    function showVideoFrame(index) {
        const frame = activeFrameResults[index];
        if (!frame) {
            return;
        }
        activeDetections = frame.detections;
        const previewImage = document.getElementById("preview-image");
        previewImage.onload = () => {
            ui.syncOverlayToImage();
            ui.renderBoundingBoxes(activeDetections);
        };
        previewImage.src = frame.imageUrl;
        document.getElementById("preview-empty").classList.add("hidden");
        document.getElementById("preview-content").classList.remove("hidden");
        document.getElementById("current-frame").textContent = String(index + 1);
        document.getElementById("frame-slider").value = String(index + 1);
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
        activeDetections = [];
        activeFrameResults = [];
        document.getElementById("file-input").value = "";
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
        const speed = Number(document.getElementById("speed-select").value) || 1;
        playbackTimer = setInterval(() => {
            const next = Number(slider.value) + 1;
            if (next > Number(slider.max)) {
                stopPlayback();
                return;
            }
            slider.value = String(next);
            showVideoFrame(next - 1);
        }, 650 / speed);
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
