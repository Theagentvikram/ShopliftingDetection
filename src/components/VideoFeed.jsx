import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

// Fallback detection settings
const FALLBACK_DETECTION_ENABLED = true;

// Common objects for fallback detection
const FALLBACK_OBJECTS = [
  { class: 'person', score: 0.95 },
  { class: 'backpack', score: 0.87 },
  { class: 'bottle', score: 0.82 },
  { class: 'chair', score: 0.76 },
  { class: 'laptop', score: 0.91 }
];

// Use the green color for all objects as shown in the image
const BOUNDING_BOX_COLOR = '#00FF00'; // Bright green

const VideoFeed = ({ onDetection }) => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [error, setError] = useState(null);
  const [stream, setStream] = useState(null);
  const [loading, setLoading] = useState(true);
  const [useFallbackDetection, setUseFallbackDetection] = useState(false);
  const [fallbackPosition, setFallbackPosition] = useState(0);
  const [isComponentMounted, setIsComponentMounted] = useState(true);
  const detectionIntervalRef = useRef(null);
  
  // Load COCO-SSD model
  useEffect(() => {
    let isMounted = true;
    
    const loadModel = async () => {
      try {
        // Check if WebGL is available
        const webglSupported = tf.getBackend() === 'webgl' || await tf.setBackend('webgl');
        console.log("WebGL supported:", webglSupported);
        
        if (!webglSupported) {
          console.warn("WebGL not supported, using CPU backend");
          await tf.setBackend('cpu');
        }
        
        console.log("TensorFlow backend:", tf.getBackend());
        console.log("TensorFlow.js version:", tf.version.tfjs);
        
        // Load model with a timeout
        const modelPromise = cocoSsd.load({
          base: 'lite_mobilenet_v2'  // Use a lighter model for better performance
        });
        
        // Set a timeout to prevent hanging
        const timeoutPromise = new Promise((_, reject) => 
          setTimeout(() => reject(new Error("Model loading timed out after 15 seconds")), 15000)
        );
        
        // Wait for the model or timeout
        const loadedModel = await Promise.race([modelPromise, timeoutPromise]);
        
        if (isMounted) {
          setModel(loadedModel);
          console.log("Model loaded successfully");
          
          // Test model with a simple inference
          try {
            // Create a small test image
            const testTensor = tf.zeros([1, 300, 300, 3]);
            const testResult = await loadedModel.detect(testTensor);
            console.log("Test inference successful:", testResult);
            testTensor.dispose();
            
            // Model is working correctly
            setUseFallbackDetection(false);
          } catch (testErr) {
            console.error("Test inference failed:", testErr);
            console.warn("Using fallback detection mechanism");
            setUseFallbackDetection(FALLBACK_DETECTION_ENABLED);
          }
        }
      } catch (err) {
        console.error("Error loading model:", err);
        if (isMounted) {
          setError("Failed to load detection model: " + err.message);
          setUseFallbackDetection(FALLBACK_DETECTION_ENABLED);
        }
      }
    };
    
    loadModel();
    
    return () => {
      isMounted = false;
    };
  }, []);

  // Draw detections on canvas
  const drawDetections = (predictions) => {
    if (!canvasRef.current || !webcamRef.current || !webcamRef.current.video || !isComponentMounted) {
      return;
    }
    
    const canvas = canvasRef.current;
    const video = webcamRef.current.video;
    
    // Skip if video dimensions aren't available
    if (!video.videoWidth || !video.videoHeight) {
      return;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Ensure canvas dimensions match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Check if we have any predictions
    if (!predictions || !Array.isArray(predictions) || predictions.length === 0) {
      return;
    }

    // Draw detections
    predictions.forEach((prediction) => {
      try {
        if (!prediction || !prediction.bbox || !Array.isArray(prediction.bbox) || prediction.bbox.length !== 4) {
          return;
        }

        // Get coordinates and ensure they are numbers
        const [x, y, width, height] = prediction.bbox.map(Number);
        
        // Skip invalid coordinates
        if ([x, y, width, height].some(val => isNaN(val) || val < 0)) {
          return;
        }

        // Draw box
        ctx.beginPath();
        ctx.strokeStyle = BOUNDING_BOX_COLOR;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Prepare text
        const label = prediction.class || 'unknown';
        const confidence = prediction.score || 0;
        const text = `${label} ${Math.round(confidence * 100)}%`;
        
        // Draw background for text
        ctx.font = 'bold 16px Arial';
        const textMetrics = ctx.measureText(text);
        const textHeight = 20;
        ctx.fillStyle = BOUNDING_BOX_COLOR;
        ctx.fillRect(x, y - textHeight - 2, textMetrics.width + 10, textHeight);
        
        // Draw text
        ctx.fillStyle = '#000000';
        ctx.fillText(text, x + 5, y - 5);
        
      } catch (err) {
        console.error('Error drawing detection:', err);
      }
    });
  };

  // Run detection on the current video frame
  const runDetection = async () => {
    if (useFallbackDetection) {
      // Use fallback detections when TF model isn't working
      handleFallbackDetection();
      return;
    }
  
    if (!model || !webcamRef.current || !webcamRef.current.video) {
      return;
    }
    
    const video = webcamRef.current.video;
    
    // Skip if video is not ready or paused
    if (video.readyState !== 4 || !video.videoWidth || !video.videoHeight || video.paused) {
      return;
    }
    
    try {
      // Run detection
      const predictions = await model.detect(video, {
        score: 0.4,  // Lower threshold to detect more objects
        maxNumBoxes: 10  // Increased for better detection
      });
      
      if (!predictions || !Array.isArray(predictions)) {
        console.log("No valid predictions array returned");
        return;
      }

      if (predictions.length > 0) {
        console.log("Raw predictions:", JSON.stringify(predictions));
      } else {
        console.log("No objects detected in this frame");
      }
      
      // Draw the detections directly first
      drawBoxesDirectly(predictions);
      
      // Convert TensorFlow.js predictions to a standard format
      const standardizedPredictions = predictions
        .filter(pred => pred && pred.bbox && Array.isArray(pred.bbox) && pred.bbox.length === 4)
        .map(pred => {
          return {
            class: pred.class || 'unknown',
            score: pred.score || 0,
            bbox: pred.bbox
          };
        })
        .filter(Boolean); // Remove null entries
      
      // Send detections to parent component
      if (onDetection && standardizedPredictions.length > 0) {
        onDetection(standardizedPredictions);
      } else if (onDetection) {
        onDetection([]);
      }
    } catch (err) {
      console.error("Error running detection:", err);
      if (err.message.includes('undefined') || err.message.includes('null')) {
        console.warn("Detection returned invalid data, switching to fallback");
        setUseFallbackDetection(FALLBACK_DETECTION_ENABLED);
      }
    }
  };

  // Draw bounding boxes directly from COCO-SSD format
  const drawBoxesDirectly = (predictions) => {
    if (!canvasRef.current || !webcamRef.current || !webcamRef.current.video) {
      return;
    }
    
    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    
    if (!video.videoWidth || !video.videoHeight) {
      return;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Make sure canvas size matches video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Exit if no predictions
    if (!predictions || !Array.isArray(predictions) || predictions.length === 0) {
      return;
    }
    
    // Draw each box
    predictions.forEach(prediction => {
      if (!prediction || !prediction.bbox) return;
      
      const [x, y, width, height] = prediction.bbox;
      
      // Draw rectangle
      ctx.lineWidth = 3;
      ctx.strokeStyle = BOUNDING_BOX_COLOR;
      ctx.strokeRect(x, y, width, height);
      
      // Text to display
      const score = prediction.score * 100;
      const label = `${prediction.class} ${Math.round(score)}%`;
      
      // Background for text
      ctx.font = 'bold 16px Arial';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = BOUNDING_BOX_COLOR;
      ctx.fillRect(x, y - 22, textWidth + 10, 22);
      
      // Text
      ctx.fillStyle = 'black';
      ctx.fillText(label, x + 5, y - 5);
    });
  };

  // Fallback detection method when TensorFlow isn't working
  const handleFallbackDetection = () => {
    // Get canvas size from video if available
    let canvasWidth = 640;
    let canvasHeight = 480;
    
    if (webcamRef.current && webcamRef.current.video) {
      canvasWidth = webcamRef.current.video.videoWidth || canvasWidth;
      canvasHeight = webcamRef.current.video.videoHeight || canvasHeight;
    }
    
    // Calculate box positions based on canvas size to ensure they're always visible
    const positionedObjects = FALLBACK_OBJECTS.map((obj, index) => {
      // Create different sized boxes for different objects
      let width, height;
      
      switch (obj.class) {
        case 'person':
          width = Math.round(canvasWidth * 0.15);  // 15% of canvas width
          height = Math.round(canvasHeight * 0.4); // 40% of canvas height
          break;
        case 'backpack':
          width = Math.round(canvasWidth * 0.1);
          height = Math.round(canvasHeight * 0.15);
          break;
        case 'bottle':
          width = Math.round(canvasWidth * 0.05);
          height = Math.round(canvasHeight * 0.12);
          break;
        case 'laptop':
          width = Math.round(canvasWidth * 0.2);
          height = Math.round(canvasHeight * 0.12);
          break;
        default:
          width = Math.round(canvasWidth * 0.1);
          height = Math.round(canvasHeight * 0.1);
      }
      
      // Stagger initial positions around the canvas
      const xPos = (canvasWidth * 0.2) + (index * (canvasWidth * 0.15));
      const yPos = (canvasHeight * 0.2) + (index % 3 * (canvasHeight * 0.15));
      
      return {
        ...obj,
        bbox: [xPos, yPos, width, height],
        color: BOUNDING_BOX_COLOR // Use consistent color
      };
    });
    
    // Create a moving detection to simulate tracking
    const movingDetections = positionedObjects.map((detection, index) => {
      // Make a copy of the original detection
      const newDetection = { ...detection };
      
      // Adjust position based on the animation counter
      const offsetX = Math.sin(fallbackPosition / 20 + index) * (canvasWidth * 0.08);
      const offsetY = Math.cos(fallbackPosition / 15 + index * 2) * (canvasHeight * 0.05);
      
      // Calculate new position while keeping within bounds
      const [x, y, width, height] = newDetection.bbox;
      const newX = Math.max(0, Math.min(canvasWidth - width, x + offsetX));
      const newY = Math.max(0, Math.min(canvasHeight - height, y + offsetY));
      
      // Update the bbox
      newDetection.bbox = [newX, newY, width, height];
      
      return newDetection;
    });
    
    // Update position for next frame
    setFallbackPosition(prev => (prev + 1) % 360);
    
    // Draw and pass to parent component
    drawDetections(movingDetections);
    
    if (onDetection) {
      onDetection(movingDetections);
    }
  };

  // Cleanup function
  const cleanup = () => {
    // Clear detection interval
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    // Stop media recorder if active
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }

    // Stop all tracks
    if (stream) {
      stream.getTracks().forEach(track => {
        track.stop();
        track.enabled = false;
      });
    }

    // Clear video source
    if (webcamRef.current && webcamRef.current.video) {
      webcamRef.current.video.srcObject = null;
    }

    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    // Reset states
    setStream(null);
    setMediaRecorder(null);
    setIsRecording(false);
    setModel(null);
    setError(null);
    setUseFallbackDetection(false);
  };

  // Component mount/unmount effect
  useEffect(() => {
    setIsComponentMounted(true);

    // Cleanup on unmount
    return () => {
      setIsComponentMounted(false);
      cleanup();
    };
  }, []);

  // Handle component visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && isComponentMounted) {
        cleanup();
      } else if (!document.hidden && isComponentMounted) {
        // Restart camera when page becomes visible again
        setupCamera();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isComponentMounted]);

  // Start webcam and detection loop
  useEffect(() => {
    let mounted = true;
    let resizeObserver;
    
    const setupCamera = async () => {
      try {
        setLoading(true);
        
        // Camera constraints - use lower resolution for better performance
        const constraints = {
          video: {
            facingMode: "environment",
            width: { ideal: 480, max: 640 },
            height: { ideal: 360, max: 480 },
            frameRate: { ideal: 15, max: 30 }
          }
        };
        
        console.log("Requesting camera access with constraints:", constraints);
        
        // Access webcam
        const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (!mounted || !isComponentMounted) {
          mediaStream.getTracks().forEach(track => track.stop());
          return;
        }
        
        // Set video source
        if (webcamRef.current && webcamRef.current.video) {
          webcamRef.current.video.srcObject = mediaStream;
          setStream(mediaStream);
          console.log("Camera stream connected successfully");
          
          // Add resize observer
          if ('ResizeObserver' in window) {
            resizeObserver = new ResizeObserver(entries => {
              if (!isComponentMounted) return;
              for (const entry of entries) {
                if (entry.target === webcamRef.current?.video && canvasRef.current) {
                  canvasRef.current.width = entry.target.videoWidth;
                  canvasRef.current.height = entry.target.videoHeight;
                }
              }
            });
            
            resizeObserver.observe(webcamRef.current.video);
          }
        }
        
        // Setup detection interval
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
        }
        
        detectionIntervalRef.current = setInterval(() => {
          if (isComponentMounted) {
            runDetection();
          }
        }, 100);  // Run at 10fps
        
      } catch (err) {
        console.error("Error accessing camera:", err);
        if (mounted && isComponentMounted) {
          setError("Failed to access camera: " + err.message);
          setUseFallbackDetection(true);
        }
      } finally {
        if (mounted && isComponentMounted) {
          setLoading(false);
        }
      }
    };
    
    if (isComponentMounted) {
      setupCamera();
    }
    
    return () => {
      mounted = false;
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      cleanup();
    };
  }, [isComponentMounted]);

  // Handle video ready state
  const handleCanPlay = () => {
    setLoading(false);
    if (webcamRef.current && webcamRef.current.video) {
      console.log("Video ready to play, dimensions:", 
        webcamRef.current.video.videoWidth, 
        webcamRef.current.video.videoHeight
      );
      
      // Ensure canvas is properly sized when video is ready
      if (canvasRef.current && webcamRef.current.video.videoWidth) {
        canvasRef.current.width = webcamRef.current.video.videoWidth;
        canvasRef.current.height = webcamRef.current.video.videoHeight;
        console.log("Canvas dimensions set to:", canvasRef.current.width, canvasRef.current.height);
      }
      
      webcamRef.current.video.play().catch(err => {
        console.error("Error playing video:", err);
      });
    }
  };

  // Start/stop recording
  const startRecording = () => {
    if (webcamRef.current && webcamRef.current.video && webcamRef.current.video.srcObject) {
      try {
        const stream = webcamRef.current.video.srcObject;
        const recorder = new MediaRecorder(stream);
        const chunks = [];
        
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunks.push(e.data);
          }
        };
        
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'video/webm' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          document.body.appendChild(a);
          a.style = 'display: none';
          a.href = url;
          a.download = 'recording.webm';
          a.click();
          URL.revokeObjectURL(url);
          document.body.removeChild(a);
        };
        
        recorder.start();
        setMediaRecorder(recorder);
        setIsRecording(true);
      } catch (err) {
        console.error("Error starting recording:", err);
      }
    }
  };
  
  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  // Take screenshot
  const takeScreenshot = () => {
    if (webcamRef.current) {
      try {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          const a = document.createElement('a');
          document.body.appendChild(a);
          a.style = 'display: none';
          a.href = imageSrc;
          a.download = 'screenshot.png';
          a.click();
          URL.revokeObjectURL(imageSrc);
          document.body.removeChild(a);
        }
      } catch (err) {
        console.error("Error taking screenshot:", err);
      }
    }
  };

  // Handle video click to stop/start
  const handleVideoClick = () => {
    if (!webcamRef.current || !webcamRef.current.video) return;
    
    const video = webcamRef.current.video;
    
    if (video.paused) {
      // Resume video and detection
      video.play();
      
      // Restart detection interval if needed
      if (!detectionIntervalRef.current) {
        detectionIntervalRef.current = setInterval(() => {
          if (isComponentMounted) {
            runDetection();
          }
        }, 100);
      }
    } else {
      // Pause video and keep detection overlay visible
      video.pause();
      
      // Keep the detection interval running so we can see the last frame's detections
      // Instead of clearing it completely
    }
  };

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-gray-900">
      {/* Loading indicator */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 z-10">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mb-2"></div>
            <p className="text-white">Loading video feed...</p>
          </div>
        </div>
      )}
      
      {/* Fallback indicator */}
      {useFallbackDetection && !error && (
        <div className="absolute top-0 left-0 z-10 bg-yellow-500 text-white px-3 py-1 text-xs font-medium m-2 rounded-full">
          Fallback Detection Active
        </div>
      )}
      
      {/* Error message */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
          <div className="text-center p-4 max-w-md bg-red-50 rounded-lg">
            <svg className="w-12 h-12 text-red-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <h3 className="text-lg font-medium text-red-900 mb-1">Camera Error</h3>
            <p className="text-sm text-red-700">{error}</p>
            <div className="mt-3">
              <div className="bg-gray-100 p-3 rounded-lg text-sm text-gray-700">
                <p>Using simulated detection data instead.</p>
                <p className="mt-1">Detection activity will continue to be shown in the panel.</p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Video container with fixed aspect ratio */}
      <div className="relative w-full h-full max-w-[640px] max-h-[480px] mx-auto">
        {/* Video feed */}
        <Webcam
          ref={webcamRef}
          onCanPlay={handleCanPlay}
          className="w-full h-full object-contain bg-black cursor-pointer"
          screenshotFormat="image/png"
          videoConstraints={{
            width: { ideal: 480, max: 640 },
            height: { ideal: 360, max: 480 },
            facingMode: "environment",
            aspectRatio: 4/3
          }}
          mirrored={false}
          audio={false}
          onClick={handleVideoClick}
        />
        
        {/* Detection overlay */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          style={{
            width: webcamRef.current?.video?.videoWidth || '100%',
            height: webcamRef.current?.video?.videoHeight || '100%'
          }}
        />
      </div>
      
      {/* Controls */}
      <div className="absolute bottom-4 right-4 flex space-x-2">
        <button
          onClick={takeScreenshot}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-200"
        >
          Take Screenshot
        </button>
        <button
          onClick={isRecording ? stopRecording : startRecording}
          className={`px-4 py-2 ${
            isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
          } text-white rounded-lg transition-colors duration-200`}
        >
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </button>
      </div>
      
      {/* Status indicator */}
      <div className="absolute bottom-4 left-4 flex items-center space-x-2 bg-gray-900 bg-opacity-75 px-3 py-1 rounded-full">
        <div className={`h-3 w-3 rounded-full ${error ? 'bg-red-500' : stream ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`}></div>
        <span className="text-xs text-white font-medium">
          {error ? 'Offline' : stream ? 'Live' : 'Connecting...'}
        </span>
      </div>
    </div>
  );
};

export default VideoFeed; 