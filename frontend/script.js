const predictBtn = document.getElementById("predictBtn");
const inputText = document.getElementById("inputText");

const resultDiv = document.getElementById("result");
const labelSpan = document.getElementById("label");
const probabilitySpan = document.getElementById("probability");
const latencySpan = document.getElementById("latency");
const errorP = document.getElementById("error");

const API_URL = "http://localhost:8000/predict";

predictBtn.addEventListener("click", async () => {
    const text = inputText.value.trim();

    errorP.classList.add("hidden");
    resultDiv.classList.add("hidden");

    if (!text) {
        errorP.textContent = "Please enter some text.";
        errorP.classList.remove("hidden");
        return;
    }

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        const data = await response.json();

        labelSpan.textContent = data.label;
        probabilitySpan.textContent = data.probability.toFixed(4);
        latencySpan.textContent = data.latency_ms;

        labelSpan.className = "";
        if (data.label === "abusive") {
            labelSpan.classList.add("abusive");
        } else {
            labelSpan.classList.add("non-abusive");
        }

        resultDiv.classList.remove("hidden");

    } catch (err) {
        errorP.textContent = "Error calling prediction API.";
        errorP.classList.remove("hidden");
    }
});
