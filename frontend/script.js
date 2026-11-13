const sessionId = crypto.randomUUID();
const API_URL = "";

// Upload PDF
async function uploadPDF() {

    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("uploadStatus");

    const file = fileInput.files[0];

    if (!file) {
        status.textContent = "Please select a PDF.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    status.textContent = "Uploading and processing...";

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed");
        }

        status.textContent =
            `Uploaded successfully! Chunks: ${data.total_chunks}`;

    } catch (error) {
        status.textContent = error.message;
    }
}


// Send question to RAG backend
async function sendQuestion() {

    const input = document.getElementById("question");
    const question = input.value.trim();

    if (!question) return;

    addMessage(question, "user");

    input.value = "";

    addMessage("Thinking...", "bot");

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                session_id: sessionId
            })
            });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Failed to get answer");
        }

        const messages = document.getElementById("chatMessages");
        messages.lastElementChild.remove();

        let answer = data.answer;

        let sources = "";

        if (data.sources && data.sources.length > 0) {
            sources = "\n\nSources:\n" +
                data.sources.map(
                    source => `Document ID: ${source.document_id}`
                ).join("\n");
        }

        addMessage(answer + sources, "bot");

    } catch (error) {
        const messages = document.getElementById("chatMessages");
        messages.lastElementChild.remove();

        addMessage(error.message, "bot");
    }
}


// Display messages
function addMessage(message, sender) {

    const chatMessages = document.getElementById("chatMessages");

    const div = document.createElement("div");

    div.className = `message ${sender}`;

    // Use textContent to safely display generated text
    div.textContent = message;

    chatMessages.appendChild(div);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// Send question when Enter is pressed
document.getElementById("question").addEventListener("keydown", function(event) {

    if (event.key === "Enter") {
        sendQuestion();
    }

});