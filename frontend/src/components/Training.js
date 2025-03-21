import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
  LinearProgress,
  Alert,
  TextField,
  Paper,
} from '@mui/material';
import { CloudUpload, School } from '@mui/icons-material';
import axios from 'axios';

const Training = () => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [selectedAnnotations, setSelectedAnnotations] = useState(null);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);

  const handleVideoSelect = (event) => {
    setSelectedVideo(event.target.files[0]);
  };

  const handleAnnotationsSelect = (event) => {
    setSelectedAnnotations(event.target.files[0]);
  };

  const handleTrain = async () => {
    if (!selectedVideo || !selectedAnnotations) return;

    const formData = new FormData();
    formData.append('video', selectedVideo);
    formData.append('annotations', selectedAnnotations);

    setTraining(true);
    try {
      const response = await axios.post('http://localhost:5000/api/train', formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        }
      });

      setResult({
        type: 'success',
        message: `Training completed successfully. New model version: ${response.data.model_version}`
      });
    } catch (error) {
      setResult({
        type: 'error',
        message: 'Error during training: ' + error.message
      });
    } finally {
      setTraining(false);
      setProgress(0);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Train Detection Model
      </Typography>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Upload Training Data
        </Typography>

        <Box sx={{ my: 3 }}>
          <input
            accept="video/*"
            style={{ display: 'none' }}
            id="video-upload"
            type="file"
            onChange={handleVideoSelect}
          />
          <label htmlFor="video-upload">
            <Button
              variant="contained"
              component="span"
              startIcon={<CloudUpload />}
              disabled={training}
            >
              Select Training Video
            </Button>
          </label>
          {selectedVideo && (
            <Typography variant="body2" sx={{ mt: 1 }}>
              Selected Video: {selectedVideo.name}
            </Typography>
          )}
        </Box>

        <Box sx={{ my: 3 }}>
          <input
            accept=".json"
            style={{ display: 'none' }}
            id="annotations-upload"
            type="file"
            onChange={handleAnnotationsSelect}
          />
          <label htmlFor="annotations-upload">
            <Button
              variant="contained"
              component="span"
              startIcon={<CloudUpload />}
              disabled={training}
            >
              Select Annotations File
            </Button>
          </label>
          {selectedAnnotations && (
            <Typography variant="body2" sx={{ mt: 1 }}>
              Selected Annotations: {selectedAnnotations.name}
            </Typography>
          )}
        </Box>

        {selectedVideo && selectedAnnotations && (
          <Button
            variant="contained"
            color="primary"
            startIcon={<School />}
            onClick={handleTrain}
            disabled={training}
            sx={{ mt: 2 }}
          >
            Start Training
          </Button>
        )}

        {training && (
          <Box sx={{ width: '100%', mt: 2 }}>
            <LinearProgress variant="determinate" value={progress} />
            <Typography variant="body2" sx={{ mt: 1 }}>
              Training Progress: {progress}%
            </Typography>
          </Box>
        )}

        {result && (
          <Alert severity={result.type} sx={{ mt: 2 }}>
            {result.message}
          </Alert>
        )}
      </Paper>
    </Box>
  );
};

export default Training;
