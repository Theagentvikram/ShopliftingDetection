import React, { useState } from 'react';
import { Box, Button, Typography, LinearProgress, Alert } from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
import axios from 'axios';

const VideoUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('video', selectedFile);

    setUploading(true);
    try {
      const response = await axios.post('http://localhost:5000/api/detect', formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        }
      });

      setResult({
        type: 'success',
        message: `Detection completed. Found ${response.data.detection_summary.suspicious_count} suspicious activities.`
      });
    } catch (error) {
      setResult({
        type: 'error',
        message: 'Error processing video: ' + error.message
      });
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Upload Video for Analysis
      </Typography>

      <Box sx={{ my: 3 }}>
        <input
          accept="video/*"
          style={{ display: 'none' }}
          id="video-upload"
          type="file"
          onChange={handleFileSelect}
        />
        <label htmlFor="video-upload">
          <Button
            variant="contained"
            component="span"
            startIcon={<CloudUpload />}
            disabled={uploading}
          >
            Select Video
          </Button>
        </label>
        {selectedFile && (
          <Typography variant="body2" sx={{ mt: 1 }}>
            Selected: {selectedFile.name}
          </Typography>
        )}
      </Box>

      {selectedFile && (
        <Button
          variant="contained"
          color="primary"
          onClick={handleUpload}
          disabled={uploading}
          sx={{ mt: 2 }}
        >
          Upload and Analyze
        </Button>
      )}

      {uploading && (
        <Box sx={{ width: '100%', mt: 2 }}>
          <LinearProgress variant="determinate" value={progress} />
          <Typography variant="body2" sx={{ mt: 1 }}>
            {progress}% Uploaded
          </Typography>
        </Box>
      )}

      {result && (
        <Alert severity={result.type} sx={{ mt: 2 }}>
          {result.message}
        </Alert>
      )}
    </Box>
  );
};

export default VideoUpload;
