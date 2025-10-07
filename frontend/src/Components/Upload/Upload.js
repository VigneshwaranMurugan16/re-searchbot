import React, { useState } from "react";
import "./Upload.css";
import { MdCloudUpload } from "react-icons/md";
import { FiSend } from "react-icons/fi";
import { motion } from "framer-motion";

const Upload = () => {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      if (uploadedFile.type !== "application/pdf") {
        alert("Only PDF files are allowed!");
        return;
      }
      setFile(uploadedFile);
    }
  };

  const handleUpload = async () => {
    if (file) {
      const formData = new FormData();
      formData.append("file", file);

      setIsUploading(true);
      setProgress(0);

      // Simulate progress while waiting for server response
      const interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 10;
          if (newProgress >= 100) {
            clearInterval(interval);
          }
          return newProgress;
        });
      }, 300);

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
      } finally {
        clearInterval(interval);
        setIsUploading(false);
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
    <div className="upload">
      <div className="upload-container">
        <motion.div
          whileInView={{ opacity: 1, y: 0 }}
          initial={{ opacity: 0, y: 20 }}
          transition={{ duration: 1.5 }}
          className="upload-box"
        >
          <div className="file-info">
            <div
              className={`icon-placeholder ${isUploading ? "uploading" : ""}`}
              onClick={() => !isUploading && document.getElementById("file-upload").click()}
            >
              <MdCloudUpload size={50} color={file ? "#4caf50" : "#6C63FF"} />
              <div>{file ? file.name : "Required PDF"}</div>
            </div>
            <input
              id="file-upload"
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
              style={{ display: "none" }}
            />
            {isUploading && (
              <div className="progress-bar">
                <div className="progress" style={{ width: `${progress}%` }}></div>
                <div className="progress-percentage">{progress}%</div>
              </div>
            )}
          </div>
          <button className="submit-button" onClick={handleUpload} disabled={isUploading}>
            Submit
          </button>
          <div className="start-server">
            <button
              className="submit-button"
              onClick={handleStart}
            >
              <FiSend />
              <span>Start Chat</span>
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Upload;
