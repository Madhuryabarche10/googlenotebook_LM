// src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import UploadScreen from './components/UploadScreen';
import UploadingScreen from './components/UploadingScreen';
import ChatInterface from './components/ChatInterface';
import './App.css';

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [documentReady, setDocumentReady] = useState(false);
  const [documentId, setDocumentId] = useState(null);

  const handleFileUpload = async (file) => {
    setPdfFile(file);
    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('pdf', file);

    try {
      const response = await axios.post('http://localhost:5000/upload-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        },
      });

      const documentId = response.data.documentId;
      setDocumentId(documentId);
      setIsUploading(false);
      setDocumentReady(true);
    } catch (error) {
      console.error("Error uploading PDF:", error);
      setIsUploading(false);
      alert("Upload failed. Please try again.");
    }
  };

  const handleNewUpload = () => {
    setPdfFile(null);
    setIsUploading(false);
    setUploadProgress(0);
    setDocumentReady(false);
    setDocumentId(null);
  };

  return (
    <div className="app-container">
      {!pdfFile && !isUploading && !documentReady && (
        <UploadScreen onFileUpload={handleFileUpload} />
      )}

      {isUploading && (
        <UploadingScreen progress={uploadProgress} />
      )}

      {documentReady && pdfFile && documentId && (
        <ChatInterface
          pdfFile={pdfFile}
          documentId={documentId}
          onNewUpload={handleNewUpload}
        />
      )}
    </div>
  );
}

export default App;
