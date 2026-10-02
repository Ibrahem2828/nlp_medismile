const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const textArea = document.getElementById("symptomsText");
const resultDiv = document.getElementById("result");

let recognition;

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
  alert("المتصفح الحالي لا يدعم التعرف الصوتي. يفضل استخدام Google Chrome.");
} else {
  recognition = new SpeechRecognition();
  recognition.lang = "ar-SA";
  recognition.continuous = true;
  recognition.interimResults = false;

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    textArea.value += transcript + " ";
  };
}

startBtn.onclick = () => {
  recognition?.start();
  startBtn.disabled = true;
  stopBtn.disabled = false;
};

stopBtn.onclick = () => {
  recognition?.stop();
  startBtn.disabled = false;
  stopBtn.disabled = true;
};

analyzeBtn.onclick = async () => {
  const text = textArea.value.trim();
  if (!text) {
    alert("يرجى إدخال وصف الأعراض النصية أولاً.");
    return;
  }

  resultDiv.style.display = "block";
  resultDiv.innerHTML = "<p>جاري تحليل الأعراض النصية...</p>";

  const response = await fetch("/api/analyze-text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  const data = await response.json();

  resultDiv.innerHTML = `
    <div class="patient-section">
      <h3>خلاصة للمريض</h3>
      <p>${data.patient_explanation}</p>
    </div>

    <div class="summary-section">
      <p><strong>الشدة:</strong> ${data.severity}</p>
      <p><strong>الإلحاح:</strong> ${data.urgency}</p>
    </div>

    <details class="technical-section">
      <summary>تفاصيل تقنية (القواعد / AraBERT)</summary>
      <p><strong>النص بعد التنظيف:</strong> ${data.text}</p>
      <p><strong>التشخيص النهائي:</strong> ${data.diagnosis}</p>
      <p><strong>تشخيص القواعد:</strong> ${data.rule_diagnosis}</p>
      <p><strong>تشخيص AraBERT:</strong> ${data.arabert_diagnosis}</p>
      <p><strong>درجة AraBERT:</strong> ${data.arabert_score}</p>
      ${
        data.arabert_second
          ? `<p><strong>تشخيص بديل:</strong> ${data.arabert_second} (${data.arabert_second_score})</p>`
          : ""
      }
    </details>
  `;
};
