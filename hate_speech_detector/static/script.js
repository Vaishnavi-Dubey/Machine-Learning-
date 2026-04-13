document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("analyzeForm");
  const resultDiv = document.getElementById("result");
  const resultAlert = document.getElementById("resultAlert");
  const classificationSpan = document.getElementById("classification");
  const confidenceSpan = document.getElementById("confidence");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const text = document.getElementById("textInput").value;
    const analyzeBtn = document.getElementById("analyzeBtn");

    // Disable button and show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML =
      '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `text=${encodeURIComponent(text)}`,
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Show result
      resultDiv.style.display = "block";
      classificationSpan.textContent = data.label;
      confidenceSpan.textContent = `${(data.confidence * 100).toFixed(2)}%`;

      // Set appropriate alert class based on prediction
      resultAlert.className = "alert";
      switch (data.prediction) {
        case 0: // Hate Speech
          resultAlert.classList.add("alert-danger");
          break;
        case 1: // Offensive Language
          resultAlert.classList.add("alert-warning");
          break;
        case 2: // Neither
          resultAlert.classList.add("alert-success");
          break;
      }
    } catch (error) {
      resultDiv.style.display = "block";
      resultAlert.className = "alert alert-danger";
      classificationSpan.textContent = "Error";
      confidenceSpan.textContent = error.message;
    }

    // Reset button state
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = "Analyze Text";
  });
});
