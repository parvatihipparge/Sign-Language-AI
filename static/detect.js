const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const labelElem = document.getElementById("label");
const confElem = document.getElementById("confidence");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");

let detecting = false;
let intervalId = null;
let stream = null;
let isProcessing = false;

// Loading indicator
const videoWrapper = document.querySelector(".video-wrapper");
const loadingIndicator = document.createElement("div");

loadingIndicator.className = "loading-indicator";
loadingIndicator.innerHTML =
    '<i class="fas fa-spinner fa-spin"></i><p>Initializing camera...</p>';

videoWrapper.appendChild(loadingIndicator);

// Notifications
function showNotification(message, type = "info") {
    const notification = document.createElement("div");

    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${
                type === "error"
                    ? "fa-exclamation-circle"
                    : "fa-info-circle"
            }"></i>
            <p></p>
        </div>
        <button class="notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;

    notification.querySelector("p").textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => notification.classList.add("show"), 10);

    const timeout = setTimeout(() => {
        dismissNotification(notification);
    }, 5000);

    notification
        .querySelector(".notification-close")
        .addEventListener("click", () => {
            clearTimeout(timeout);
            dismissNotification(notification);
        });
}

function dismissNotification(notification) {
    notification.classList.remove("show");

    setTimeout(() => notification.remove(), 300);
}

// Start camera
async function startCamera() {
    try {
        loadingIndicator.style.display = "flex";

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        loadingIndicator.style.display = "none";

    } catch (error) {
        loadingIndicator.style.display = "none";

        showNotification(
            "Camera access failed: " + error.message,
            "error"
        );
    }
}

// Stop camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
}

// Capture webcam frame and predict
async function captureAndPredict() {
    if (!detecting || isProcessing || !video.videoWidth) {
        return;
    }

    isProcessing = true;

    try {
        const ctx = canvas.getContext("2d");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Mirror the frame to match the training dataset.
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );
        ctx.restore();

        const imageData = canvas.toDataURL("image/jpeg", 0.8);

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: imageData,
                session_id: "webcam-session"
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        const output = document.querySelector(".output");

        output.classList.remove("fade-in", "error");
        void output.offsetWidth;

        if (data.label && data.label !== "No Hand Detected") {
            labelElem.textContent = data.label;

            confElem.textContent =
                (data.confidence * 100).toFixed(1) + "%";

            output.style.borderLeftColor = "var(--success)";
            output.style.background = "rgba(16, 185, 129, 0.1)";

        } else {
            labelElem.textContent = "No Hand Detected";
            confElem.textContent = "---";

            output.style.borderLeftColor = "var(--error)";
            output.style.background = "rgba(239, 68, 68, 0.1)";

            output.classList.add("error");
        }

        output.classList.add("fade-in");

    } catch (error) {
        console.error("Prediction error:", error);

        labelElem.textContent = "Error";
        confElem.textContent = error.message;

        const output = document.querySelector(".output");

        output.style.borderLeftColor = "var(--error)";
        output.style.background = "rgba(239, 68, 68, 0.1)";
        output.classList.add("error", "fade-in");

    } finally {
        isProcessing = false;
    }
}

// Start detection
function startDetection() {
    if (!stream) {
        showNotification("Camera is not available.", "error");
        return;
    }

    if (detecting) return;

    detecting = true;

    startBtn.style.display = "none";
    stopBtn.style.display = "inline-flex";

    captureAndPredict();

    intervalId = setInterval(captureAndPredict, 700);

    showNotification("Detection started", "info");
}

// Stop detection
function stopDetection() {
    detecting = false;

    if (intervalId !== null) {
        clearInterval(intervalId);
        intervalId = null;
    }

    startBtn.style.display = "inline-flex";
    stopBtn.style.display = "none";

    labelElem.textContent = "---";
    confElem.textContent = "---";

    showNotification("Detection stopped", "info");
}

// Button events
startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);

// Pause detection when leaving the page
document.addEventListener("visibilitychange", () => {
    if (document.hidden && detecting) {
        stopDetection();
        showNotification(
            "Detection paused because the page is not visible.",
            "info"
        );
    }
});

// Initialize camera
startCamera();