window.SeedApi = (() => {
    const apiBaseUrl = window.SeedData.app.apiBaseUrl;

    async function requestJson(path, options = {}) {
        const response = await fetch(`${apiBaseUrl}${path}`, options);
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        return response.json();
    }

    async function getModels() {
        return requestJson("/models");
    }

    async function predictCrop(file, modelId) {
        const formData = new FormData();
        formData.append("file", file);
        if (modelId) {
            formData.append("model_id", modelId);
        }
        return requestJson("/predict/crop", {
            method: "POST",
            body: formData,
        });
    }

    async function predictDetections(file, modelId, scoreThreshold = 0.5) {
        const formData = new FormData();
        formData.append("file", file);
        if (modelId) {
            formData.append("model_id", modelId);
        }
        formData.append("score_threshold", String(scoreThreshold));
        return requestJson("/predict/detect", {
            method: "POST",
            body: formData,
        });
    }

    return {
        getModels,
        predictCrop,
        predictDetections,
    };
})();
