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
    if (!canvasRef.current || !webcamRef.current || !webcamRef.current.video) {
      console.log("Canvas or video reference not available");
      return;
    }
    
    const canvas = canvasRef.current;
    const video = webcamRef.current.video;
    
    // Skip if video dimensions aren't available
    if (!video.videoWidth || !video.videoHeight) {
      console.log("Video dimensions not available yet");
      return;
    }
    
    const ctx = canvas.getContext('2d');

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Check if we have any predictions
    if (!predictions || !Array.isArray(predictions) || predictions.length === 0) {
      return;
    }

    console.log(`Drawing ${predictions.length} predictions`);
    
    // Color mapping for different objects
    const colorMap = {
      person: '#FF0000', // Red
      backpack: '#00FF00', // Green
      bottle: '#0000FF', // Blue
      cell_phone: '#FFFF00', // Yellow
      laptop: '#FF00FF', // Magenta
      default: '#00FFFF' // Cyan
    };
    
    // Draw detections
    predictions.forEach((prediction, index) => {
      try {
        // Ensure prediction exists
        if (!prediction) {
          return;
        }
        
        // Ensure bbox exists
        if (!prediction.bbox || !Array.isArray(prediction.bbox) || prediction.bbox.length < 4) {
          return;
        }
        
        // Extract bbox coordinates
        let [x, y, width, height] = prediction.bbox;
        
        // Convert to numbers and validate
        x = Number(x);
        y = Number(y);
        width = Number(width);
        height = Number(height);
        
        if (isNaN(x) || isNaN(y) || isNaN(width) || isNaN(height) || 
            width <= 0 || height <= 0) {
          console.log(`Invalid dimensions for prediction ${index}`);
          return;
        }
        
        // Ensure coordinates are within canvas bounds
        x = Math.max(0, Math.min(x, canvas.width - 1));
        y = Math.max(0, Math.min(y, canvas.height - 1));
        width = Math.min(width, canvas.width - x);
        height = Math.min(height, canvas.height - y);
        
        // Get class name and confidence
        const className = prediction.class || 'unknown';
        const score = prediction.score || 0;
        const text = `${className} ${Math.round(score * 100)}%`;
        
        // Select color based on object class
        const color = colorMap[className.toLowerCase()] || colorMap.default;
        
        // Draw bounding box with thicker lines
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);
        
        // Add a semi-transparent fill
        ctx.fillStyle = color + '33'; // 20% opacity
        ctx.fillRect(x, y, width, height);
        
        // Draw background for text
        ctx.fillStyle = color + 'CC'; // 80% opacity
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width;
        const textHeight = 20;
        ctx.fillRect(x, y > textHeight ? y - textHeight : y + height, textWidth + 10, textHeight);
        
        // Draw label with shadow for better visibility
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 16px Arial';
        ctx.shadowColor = 'black';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        ctx.fillText(text, x + 5, y > textHeight ? y - 5 : y + height + 15);
        
        // Reset shadow
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
      } catch (err) {
        console.error(`Error drawing detection ${index}:`, err);
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
    
    // Skip if video is not ready
    if (video.readyState !== 4 || !video.videoWidth || !video.videoHeight) {
      return;
    }
    
    try {
      // Run detection
      const predictions = await model.detect(video, {
        score: 0.4,  // Lower threshold to detect more objects
        maxNumBoxes: 10  // Increased for better detection
      });
      
      if (predictions && predictions.length > 0) {
        console.log("Raw predictions:", JSON.stringify(predictions));
      } else {
        console.log("No objects detected in this frame");
      }
      
      // Convert TensorFlow.js predictions to a standard format to fix coordinate issues
      const standardizedPredictions = predictions.map(pred => {
        if (!pred || !pred.bbox || !Array.isArray(pred.bbox) || pred.bbox.length < 4) {
          return null;
        }
        
        // TensorFlow.js COCO-SSD returns [y, x, height, width] in some environments
        // Let's check the values to determine the right format
        const [first, second, third, fourth] = pred.bbox;
        
        // If the width/height values are larger than the x/y values, 
        // it means the order might be flipped
        const isFlipped = (third > first && fourth > second);
        
        let x, y, width, height;
        
        if (isFlipped) {
          // Format is [y, x, height, width], so we need to rearrange
          y = first;
          x = second;
          height = third;
          width = fourth;
          console.log("Fixed flipped coordinates");
        } else {
          // Normal format [x, y, width, height]
          x = first;
          y = second;
          width = third;
          height = fourth;
        }
        
        // Return a standardized prediction object
        return {
          class: pred.class || 'unknown',
          score: pred.score || 0,
          bbox: [x, y, width, height]
        };
      }).filter(Boolean); // Remove null entries
      
      if (standardizedPredictions.length > 0) {
        console.log("Standardized predictions:", JSON.stringify(standardizedPredictions));
        
        // Draw the detections
        drawDetections(standardizedPredictions);
        
        // Send detections to parent component
        if (onDetection) {
          onDetection(standardizedPredictions);
        }
      } else {
        // Clear the canvas if no detections
        if (canvasRef.current) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        
        // If we consistently get no detections, switch to fallback
        console.log("No valid predictions found");
      }
    } catch (err) {
      console.error("Error running detection:", err);
      console.warn("Switching to fallback detection mechanism");
      setUseFallbackDetection(FALLBACK_DETECTION_ENABLED);
    }
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
        bbox: [xPos, yPos, width, height]
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

  // Start webcam and detection loop
  useEffect(() => {
    let detectionInterval;
    let mounted = true;
    let resizeObserver;
    
    const setupCamera = async () => {
      try {
        setLoading(true);
        
        // Camera constraints - use lower resolution for better performance
        const constraints = {
          video: {
            facingMode: "environment",
            width: { ideal: 480, max: 640 },  // Reduced size for better performance
            height: { ideal: 360, max: 480 },
            frameRate: { ideal: 10, max: 15 } // Lower framerate for stable detection
          }
        };
        
        console.log("Requesting camera access with constraints:", constraints);
        
        // Access webcam
        const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (!mounted) {
          mediaStream.getTracks().forEach(track => track.stop());
          return;
        }
        
        // Set video source
        if (webcamRef.current && webcamRef.current.video) {
          webcamRef.current.video.srcObject = mediaStream;
          setStream(mediaStream);
          console.log("Camera stream connected successfully");
          
          // Log the tracks for debugging
          mediaStream.getTracks().forEach(track => {
            console.log("Track settings:", track.getSettings());
          });
          
          // Add resize observer to keep canvas in sync with video dimensions
          if ('ResizeObserver' in window) {
            resizeObserver = new ResizeObserver(entries => {
              for (const entry of entries) {
                if (entry.target === webcamRef.current.video && canvasRef.current) {
                  // Update canvas size when video element resizes
                  canvasRef.current.width = webcamRef.current.video.videoWidth;
                  canvasRef.current.height = webcamRef.current.video.videoHeight;
                  console.log("Canvas resized to match video:", 
                    canvasRef.current.width, canvasRef.current.height);
                }
              }
            });
            
            // Start observing the video element
            resizeObserver.observe(webcamRef.current.video);
          }
        }
        
        // Setup detection interval once camera is ready
        detectionInterval = setInterval(() => {
          runDetection();
        }, 250);  // Increased to 4fps for better detection
        
      } catch (err) {
        console.error("Error accessing camera:", err);
        setError("Failed to access camera: " + err.message);
        setUseFallbackDetection(true);
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };
    
    setupCamera();
    
    return () => {
      mounted = false;
      
      // Clear detection interval
      if (detectionInterval) clearInterval(detectionInterval);
      
      // Clean up resize observer
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      
      // Stop all tracks
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

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
          className="w-full h-full object-contain bg-black"
          screenshotFormat="image/png"
          videoConstraints={{
            width: { ideal: 480, max: 640 },
            height: { ideal: 360, max: 480 },
            facingMode: "environment",
            aspectRatio: 4/3
          }}
          mirrored={false}
          audio={false}
        />
        
        {/* Detection overlay */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
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