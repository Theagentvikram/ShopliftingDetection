import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { Box, Button, Alert } from '@mui/material';
import { PlayArrow, Stop } from '@mui/icons-material';

const LiveDetection = () => {
  const webcamRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [alert, setAlert] = useState(null);

  const startDetection = async () => {
    setIsDetecting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      webcamRef.current.srcObject = stream;
      
      // Start sending frames to backend
      const ws = new WebSocket('ws://localhost:5000/ws/detect');
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.suspicious_activity) {
          setAlert({
            severity: 'warning',
            message: `Suspicious activity detected: ${data.suspicious_activity}`
          });
        }
      };
    } catch (error) {
      console.error('Error accessing webcam:', error);
      setAlert({
        severity: 'error',
        message: 'Error accessing webcam'
      });
    }
  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (webcamRef.current.srcObject) {
      webcamRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 2 }}>
        {alert && (
          <Alert severity={alert.severity} onClose={() => setAlert(null)}>
            {alert.message}
          </Alert>
        )}
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <Webcam
          ref={webcamRef}
          audio={false}
          width={640}
          height={480}
          screenshotFormat="image/jpeg"
        />
      </Box>

      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<PlayArrow />}
          onClick={startDetection}
          disabled={isDetecting}
        >
          Start Detection
        </Button>
        <Button
          variant="contained"
          color="secondary"
          startIcon={<Stop />}
          onClick={stopDetection}
          disabled={!isDetecting}
        >
          Stop Detection
        </Button>
      </Box>
    </Box>
  );
};

export default LiveDetection;
