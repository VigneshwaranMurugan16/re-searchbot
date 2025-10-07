import React, { useState } from "react";

const FrontPage = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    const allowedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
      "application/vnd.openxmlformats-officedocument.presentationml.presentation", // .pptx
    ];
    if (uploadedFile && allowedTypes.includes(uploadedFile.type)) {
      setFile(uploadedFile);
    } else {
      alert("Please upload a valid PDF, DOCX, or PPTX file.");
    }
  };

  const handleUpload = async () => {
    if (file) {
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          alert(data.message);
        } else {
          const errorResponse = await response.json();
          alert(`Failed to upload file: ${errorResponse.detail || "Unknown error"}`);
        }
      } catch (error) {
        alert("An error occurred while uploading the file.");
      }
    } else {
      alert("Please select a file to upload.");
    }
  };

  const handleStart = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/start", {
        method: "POST",
      });

      if (response.ok) {
        const data = await response.json();
        if (data.redirect_url) {
          window.location.href = data.redirect_url; // Redirect to Streamlit app
        } else {
          alert(data.message || "Streamlit app is starting...");
        }
      } else {
        const errorResponse = await response.json();
        alert(`Failed to start Streamlit app: ${errorResponse.detail || "Unknown error"}`);
      }
    } catch (error) {
      alert("An error occurred while starting the Streamlit app.");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Chatbot with File Upload</h1>
      <div style={{ marginBottom: "20px" }}>
        <label>
          Upload File:
          <input
            type="file"
            accept=".pdf,.docx,.pptx"
            onChange={handleFileChange}
          />
        </label>
      </div>
      <button
        onClick={handleUpload}
        style={{ padding: "10px 20px", fontSize: "16px", marginRight: "10px" }}
      >
        Upload File
      </button>
      <button
        onClick={handleStart}
        style={{ padding: "10px 20px", fontSize: "16px" }}
      >
        Start Streamlit
      </button>
    </div>
  );
};

export default FrontPage;
