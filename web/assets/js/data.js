window.SeedData = {
    app: {
        apiBaseUrl: "http://localhost:8000/api",
    },
    models: [
        {
            id: "custom_cnn_baseline",
            name: "Custom CNN Baseline",
            task: "Crop classification",
            architecture: "custom_cnn",
            inputSize: "224 x 224",
            classes: "2 + decision layer",
            parameters: "101,858",
            epochs: "15",
            status: "ready",
            badge: "ready",
            primaryMetric: "Accuracy 83.36%",
            secondaryMetric: "Macro F1 83.16%",
            accuracy: "83.36%",
            precision: "83.27%",
            recall: "83.08%",
            f1: "83.16%",
            notes: "Baseline on cropped seed images",
        },
        {
            id: "faster_rcnn_scratch",
            name: "Faster R-CNN Scratch",
            task: "Object detection",
            architecture: "fasterrcnn_resnet50_fpn_scratch",
            inputSize: "640 x 640",
            classes: "background + 2 states",
            parameters: "detection model",
            epochs: "30",
            status: "selected",
            badge: "best",
            primaryMetric: "mAP@50 89.73%",
            secondaryMetric: "mAP 69.37%",
            accuracy: "--",
            precision: "--",
            recall: "--",
            f1: "--",
            notes: "Main model trained from raw images and XML bbox, no pretrained weights",
        },
    ],
    selectedMetrics: [
        { name: "mAP@50", value: "89.73%", change: "Faster R-CNN scratch" },
        { name: "mAP", value: "69.37%", change: "COCO-style metric" },
        { name: "Best Epoch", value: "6", change: "selected checkpoint" },
        { name: "Pretrained", value: "No", change: "trained from scratch" },
    ],
    frameSummary: [
        { frame: 1, germinated: 4, nonGerminated: 83, transition: 13, percentage: "4%" },
        { frame: 20, germinated: 18, nonGerminated: 68, transition: 14, percentage: "18%" },
        { frame: 40, germinated: 39, nonGerminated: 48, transition: 13, percentage: "39%" },
        { frame: 60, germinated: 57, nonGerminated: 35, transition: 8, percentage: "57%" },
        { frame: 80, germinated: 66, nonGerminated: 28, transition: 6, percentage: "66%" },
    ],
    germinationCurve: [4, 11, 18, 27, 39, 48, 57, 62, 66],
    lossCurve: [0.55, 0.48, 0.45, 0.43, 0.41, 0.39, 0.37, 0.36],
    f1Curve: [0.68, 0.75, 0.77, 0.80, 0.82, 0.84, 0.85, 0.86],
    confusionMatrix: [
        [8700, 2150],
        [1842, 11301],
    ],
    detections: [
        { id: 1, state: "germinated", confidence: 92, bbox: { x: 12, y: 24, width: 14, height: 18 } },
        { id: 2, state: "non_germinated", confidence: 88, bbox: { x: 34, y: 22, width: 13, height: 17 } },
        { id: 3, state: "transition", confidence: 56, bbox: { x: 54, y: 33, width: 15, height: 19 } },
        { id: 4, state: "germinated", confidence: 95, bbox: { x: 72, y: 28, width: 13, height: 17 } },
        { id: 5, state: "non_germinated", confidence: 91, bbox: { x: 21, y: 58, width: 14, height: 18 } },
        { id: 6, state: "germinated", confidence: 89, bbox: { x: 48, y: 64, width: 12, height: 16 } },
        { id: 7, state: "transition", confidence: 61, bbox: { x: 69, y: 62, width: 14, height: 18 } },
        { id: 8, state: "non_germinated", confidence: 86, bbox: { x: 84, y: 48, width: 13, height: 16 } },
    ],
    previewImage:
        "data:image/svg+xml;charset=UTF-8," +
        encodeURIComponent(`
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 520">
            <rect width="900" height="520" fill="#f3f4f6"/>
            <circle cx="450" cy="260" r="210" fill="#111827"/>
            <circle cx="450" cy="260" r="206" fill="#1f2937"/>
            <g fill="#e5e7eb" opacity="0.95">
                <ellipse cx="190" cy="150" rx="20" ry="28" transform="rotate(-20 190 150)"/>
                <ellipse cx="325" cy="140" rx="22" ry="30" transform="rotate(12 325 140)"/>
                <ellipse cx="500" cy="205" rx="23" ry="31" transform="rotate(-8 500 205)"/>
                <ellipse cx="665" cy="175" rx="20" ry="29" transform="rotate(18 665 175)"/>
                <ellipse cx="255" cy="330" rx="22" ry="30" transform="rotate(24 255 330)"/>
                <ellipse cx="430" cy="360" rx="20" ry="28" transform="rotate(-16 430 360)"/>
                <ellipse cx="620" cy="350" rx="22" ry="30" transform="rotate(10 620 350)"/>
                <ellipse cx="760" cy="280" rx="19" ry="27" transform="rotate(-28 760 280)"/>
            </g>
            <g stroke="#f9fafb" stroke-width="6" stroke-linecap="round" fill="none">
                <path d="M198 124 C225 95, 245 88, 274 83"/>
                <path d="M503 173 C535 145, 570 135, 612 138"/>
                <path d="M432 332 C463 302, 486 288, 522 279"/>
                <path d="M622 320 C642 292, 660 276, 693 260"/>
            </g>
        </svg>`),
};
