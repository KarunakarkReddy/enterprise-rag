
// --------------------------------------------------
// APPLICATION STATE
// --------------------------------------------------

const sessionId = crypto.randomUUID();

let selectedDocumentId = null;
let isUploading = false;
let isSending = false;


// --------------------------------------------------
// API RESPONSE HELPER
// --------------------------------------------------

async function getResponseData(response) {
    try {
        return await response.json();
    } catch {
        throw new Error("The server returned an invalid response.");
    }
}


// --------------------------------------------------
// UPLOAD PDF
// --------------------------------------------------

async function uploadPDF() {
    if (isUploading) return;

    const fileInput = document.getElementById("pdfFile");
    const status = document.getElementById("uploadStatus");
    const uploadButton = document.getElementById("uploadBtn");

    const file = fileInput.files[0];

    if (!file) {
        status.textContent = "Please select a PDF first.";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
        status.textContent = "Only PDF files are allowed.";
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        status.textContent = "PDF size must not exceed 20 MB.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    isUploading = true;
    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading...";
    status.textContent = "Uploading and processing PDF...";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await getResponseData(response);

        if (!response.ok) {
            throw new Error(data.detail || "PDF upload failed.");
        }

        fileInput.value = "";

        const refreshed = await loadDocuments();

        status.textContent = refreshed
            ? `Uploaded successfully! ${data.total_chunks} chunks created.`
            : "PDF uploaded successfully, but the document list could not refresh.";

    } catch (error) {
        status.textContent = error.message;

    } finally {
        isUploading = false;
        uploadButton.disabled = false;
        uploadButton.textContent = "Upload Document";
    }
}


// --------------------------------------------------
// LOAD DOCUMENTS
// --------------------------------------------------

async function loadDocuments() {
    const documentList = document.getElementById("documentList");
    const select = document.getElementById("documentSelect");

    try {
        const response = await fetch("/documents");
        const documents = await getResponseData(response);

        if (!response.ok) {
            throw new Error(
                documents.detail || "Unable to load documents."
            );
        }

        if (!Array.isArray(documents)) {
            throw new Error("Invalid document list received from server.");
        }

        const previousSelection = selectedDocumentId;

        select.replaceChildren();
        documentList.replaceChildren();

        const allOption = document.createElement("option");
        allOption.value = "";
        allOption.textContent = "All Documents";
        select.appendChild(allOption);

        if (documents.length === 0) {
            documentList.textContent = "No documents uploaded yet.";
        }

        documents.forEach(doc => {
            const option = document.createElement("option");
            option.value = String(doc.id);
            option.textContent = doc.filename;
            select.appendChild(option);

            const card = document.createElement("div");
            card.className = "document-card";

            const name = document.createElement("p");
            name.textContent = doc.filename;

            const details = document.createElement("small");
            details.textContent =
                `${doc.total_chunks ?? 0} chunks | ${doc.status}`;

            const deleteButton = document.createElement("button");
            deleteButton.type = "button";
            deleteButton.textContent = "Delete";
            deleteButton.className = "delete-btn";

            deleteButton.addEventListener("click", () => {
                deleteDocument(doc.id, doc.filename);
            });

            card.append(name, details, deleteButton);
            documentList.appendChild(card);
        });

        const documentExists = documents.some(
            doc => Number(doc.id) === previousSelection
        );

        if (previousSelection !== null && documentExists) {
            selectedDocumentId = previousSelection;
            select.value = String(previousSelection);
        } else {
            selectedDocumentId = null;
            select.value = "";
        }

        updateSelectedDocumentLabel();

        return true;

    } catch (error) {
        documentList.textContent = error.message;
        return false;
    }
}


// --------------------------------------------------
// DOCUMENT SELECTION
// --------------------------------------------------

function handleDocumentSelection() {
    const select = document.getElementById("documentSelect");

    selectedDocumentId = select.value
        ? Number(select.value)
        : null;

    updateSelectedDocumentLabel();
}


// --------------------------------------------------
// UPDATE SELECTED DOCUMENT LABEL
// --------------------------------------------------

function updateSelectedDocumentLabel() {
    const label = document.getElementById("selectedDocumentLabel");
    const select = document.getElementById("documentSelect");

    if (selectedDocumentId === null) {
        label.textContent = "Searching all documents";
        return;
    }

    const selectedOption = select.selectedOptions[0];

    label.textContent = selectedOption
        ? `Searching: ${selectedOption.textContent}`
        : "Searching all documents";
}


// --------------------------------------------------
// DELETE DOCUMENT
// --------------------------------------------------

async function deleteDocument(documentId, filename) {
    if (isUploading || isSending) {
        alert("Please wait until the current operation finishes.");
        return;
    }

    const confirmed = confirm(
        `Are you sure you want to delete "${filename}"?`
    );

    if (!confirmed) return;

    try {
        const response = await fetch(
            `/documents/${documentId}`,
            { method: "DELETE" }
        );

        const data = await getResponseData(response);

        if (!response.ok) {
            throw new Error(data.detail || "Document deletion failed.");
        }

        if (selectedDocumentId === Number(documentId)) {
            selectedDocumentId = null;
        }

        await loadDocuments();

        addMessage("assistant", data.message || "Document deleted successfully.");

    } catch (error) {
        alert(error.message);
    }
}


// --------------------------------------------------
// SEND QUESTION TO RAG
// --------------------------------------------------

async function sendQuestion() {
    if (isSending || isUploading) return;

    const input = document.getElementById("questionInput");
    const sendButton = document.getElementById("sendBtn");

    const question = input.value.trim();

    if (!question) return;

    if (question.length > 2000) {
        addMessage("assistant", "Question cannot exceed 2000 characters.");
        return;
    }

    addMessage("user", question);
    input.value = "";

    isSending = true;
    sendButton.disabled = true;
    sendButton.textContent = "Thinking...";

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question,
                session_id: sessionId,
                document_id: selectedDocumentId
            })
        });

        const data = await getResponseData(response);

        if (!response.ok) {
            throw new Error(data.detail || "Unable to get an answer.");
        }

        if (typeof data.answer !== "string") {
            throw new Error("The server returned an invalid answer.");
        }

        addMessage("assistant", data.answer);

        if (Array.isArray(data.sources) && data.sources.length > 0) {
            addSources(data.sources);
        }

    } catch (error) {
        addMessage("assistant", `Error: ${error.message}`);

    } finally {
        isSending = false;
        sendButton.disabled = false;
        sendButton.textContent = "Send";
        input.focus();
    }
}


// --------------------------------------------------
// ADD CHAT MESSAGE
// --------------------------------------------------

function addMessage(role, text) {
    const chat = document.getElementById("chatMessages");

    const message = document.createElement("div");
    message.className = `message ${role}`;

    // textContent prevents HTML injection.
    message.textContent = String(text ?? "");

    chat.appendChild(message);
    chat.scrollTop = chat.scrollHeight;
}


// --------------------------------------------------
// DISPLAY SOURCE INFORMATION
// --------------------------------------------------

function addSources(sources) {
    const chat = document.getElementById("chatMessages");

    const container = document.createElement("div");
    container.className = "sources";

    const heading = document.createElement("strong");
    heading.textContent = "Sources";

    container.appendChild(heading);

    sources.forEach(source => {
        const item = document.createElement("p");

        const documentName = source.filename
            || `Document ${source.document_id}`;

        const chunkNumber = source.chunk_index ?? source.chunk_id ?? "Unknown";

        item.textContent =
            `${documentName} | Chunk ${chunkNumber}`;

        container.appendChild(item);
    });

    chat.appendChild(container);
    chat.scrollTop = chat.scrollHeight;
}


// --------------------------------------------------
// INITIALIZE APPLICATION
// --------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    loadDocuments();

    const questionInput = document.getElementById("questionInput");

    questionInput.addEventListener("keydown", event => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendQuestion();
        }
    });
});
