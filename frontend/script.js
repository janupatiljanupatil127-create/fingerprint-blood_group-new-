const form = document.getElementById("predict-form");
const fileInput = document.getElementById("fingerprint-input");
const uploadBox = document.getElementById("upload-box");
const uploadLabel = document.getElementById("upload-label");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const resultBox = document.getElementById("result");
const predictionLabel = document.getElementById("prediction-label");
const probabilityList = document.getElementById("probability-list");
const errorBox = document.getElementById("error");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  uploadLabel.hidden = true;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  resultBox.hidden = true;

  const file = fileInput.files[0];
  if (!file) return;

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  try {
    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    predictionLabel.textContent = `Predicted Blood Group: ${data.prediction}`;

    probabilityList.innerHTML = "";
    if (data.probabilities) {
      const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
      for (const [label, prob] of sorted) {
        const li = document.createElement("li");
        li.innerHTML = `<span>${label}</span><span>${(prob * 100).toFixed(1)}%</span>`;
        probabilityList.appendChild(li);
      }
    }

    resultBox.hidden = false;
  } catch (err) {
    errorBox.textContent = err.message || "Something went wrong. Please try again.";
    errorBox.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict Blood Group";
  }
});
