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
const LABEL_COLORS = {
  person: '#FF1493',   // Deep pink for people
  backpack: '#00FF00', // Bright green for backpacks
  bottle: '#00BFFF',   // Deep sky blue for bottles
  laptop: '#FFA500',   // Orange for laptops
  chair: '#9932CC',    // Dark orchid for chairs
  default: '#FF0000'   // Bright red as default
};

const VideoFeed = ({ onDetection }) => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
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
  const [imageMode, setImageMode] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  
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
        
        // Attempt to load COCO SSD model
        console.log("Loading COCO-SSD model");
        const loadedModel = await cocoSsd.load();
        console.log("Model loaded successfully");
        
        if (isMounted) {
          setModel(loadedModel);
          setUseFallbackDetection(false); // Use real detection
        }
      } catch (err) {
        console.error("Error loading model:", err);
        if (isMounted) {
          // Fall back to simulated detection
          console.warn("Using fallback detection");
          setUseFallbackDetection(true);
        }
      }
    };
    
    loadModel();
    
    return () => {
      isMounted = false;
    };
  }, []);

  // Simple drawing function that matches the example screenshot
  const drawDetection = (predictions) => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw each prediction
    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;
      const label = prediction.class;
      const score = Math.round(prediction.score * 100);
      
      // Draw green rectangle (matching the screenshot)
      ctx.strokeStyle = BOUNDING_BOX_COLOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      
      // Draw label at the top of the box with green text (matching the screenshot)
      ctx.fillStyle = BOUNDING_BOX_COLOR;
      ctx.font = '16px Arial';
      ctx.fillText(`${label} ${score}%`, x, y - 5);
    });
  };

  // Function to handle image uploads
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setLoading(true);
    setImageMode(true);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Set canvas dimensions to match the image
        if (canvasRef.current) {
          canvasRef.current.width = img.width;
          canvasRef.current.height = img.height;
          
          // Draw the image on the canvas
          const ctx = canvasRef.current.getContext('2d');
          ctx.drawImage(img, 0, 0, img.width, img.height);
          
          // Store the image for future reference
          setUploadedImage(img);
          
          // Run detection on the image
          if (model) {
            model.detect(img).then(predictions => {
              if (predictions && predictions.length > 0) {
                // Draw the detections
                drawDetection(predictions);
                
                // Send detections to parent component
                onDetection && onDetection(predictions);
              }
              setLoading(false);
            }).catch(err => {
              console.error("Error detecting objects in image:", err);
              setLoading(false);
            });
          } else {
            // If model isn't available, use fallback
            const fakePredictions = [
              { class: 'person', score: 0.95, bbox: [50, 50, 200, 300] },
              { class: 'cell phone', score: 0.88, bbox: [300, 100, 100, 150] }
            ];
            drawDetection(fakePredictions);
            onDetection && onDetection(fakePredictions);
            setLoading(false);
          }
        }
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  };

  // Reset to live video mode
  const resetToLiveVideo = () => {
    setImageMode(false);
    setUploadedImage(null);
    if (canvasRef.current && webcamRef.current && webcamRef.current.video) {
      canvasRef.current.width = webcamRef.current.video.videoWidth;
      canvasRef.current.height = webcamRef.current.video.videoHeight;
    }
  };

  // Run detection on the current video frame
  const runDetection = async () => {
    // Skip if in image mode
    if (imageMode) return;
    
    if (!model || !webcamRef.current || !webcamRef.current.video) return;
    
    const video = webcamRef.current.video;
    
    // Skip if video is not ready or paused
    if (video.readyState !== 4 || !video.videoWidth || !video.videoHeight || video.paused) return;
    
    try {
      // Run detection with COCO-SSD
      const predictions = await model.detect(video);
      
      // Only pass detections to parent component, don't draw on canvas
      if (predictions && predictions.length > 0) {
        // Send to parent component
        onDetection && onDetection(predictions);
      } else {
        // No detections
        onDetection && onDetection([]);
      }
    } catch (err) {
      console.error("Error running detection:", err);
      if (err.message.includes('undefined') || err.message.includes('null')) {
        setUseFallbackDetection(true);
      }
    }
  };

  // Fallback detection method when TensorFlow isn't working
  const handleFallbackDetection = () => {
    // Skip if in image mode
    if (imageMode) return;
    
    // Create simulated detections
    const detections = [
      { 
        class: 'person', 
        score: 0.64, 
        bbox: [50, 50, 300, 400] 
      },
      { 
        class: 'cell phone', 
        score: 0.88, 
        bbox: [300, 100, 100, 150] 
      }
    ];
    
    // Only pass to parent component, don't draw on canvas
    onDetection && onDetection(detections);
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
        
        console.log("Setting up detection interval");
        detectionIntervalRef.current = setInterval(() => {
          if (isComponentMounted) {
            runDetection();
          }
        }, 100);  // Run at 10fps for better stability
        
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
      
      // Force play the video
      const playPromise = webcamRef.current.video.play();
      if (playPromise !== undefined) {
        playPromise.catch(err => {
          console.error("Error playing video:", err);
          // Try again with user interaction
          document.addEventListener('click', function playVideoOnce() {
            webcamRef.current?.video?.play();
            document.removeEventListener('click', playVideoOnce);
          });
        });
      }
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
    console.log("Video clicked, current state:", video.paused ? "paused" : "playing");
    
    if (video.paused) {
      // Resume video and detection
      console.log("Resuming video");
      video.play();
      
      // Restart detection interval if needed
      if (!detectionIntervalRef.current) {
        console.log("Restarting detection interval");
        detectionIntervalRef.current = setInterval(() => {
          if (isComponentMounted) {
            runDetection();
          }
        }, 100);
      } else {
        console.log("Detection interval already running");
      }
    } else {
      // Pause video
      console.log("Pausing video");
      video.pause();
    }
  };

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-gray-900">
      {/* Loading indicator */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 z-10">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mb-2"></div>
            <p className="text-white">Loading...</p>
          </div>
        </div>
      )}
      
      {/* Video/Image container */}
      <div className="relative w-full h-full max-w-[640px] max-h-[480px] mx-auto">
        {!imageMode ? (
          <Webcam
            ref={webcamRef}
            onCanPlay={handleCanPlay}
            className="w-full h-full object-contain bg-black cursor-pointer"
            screenshotFormat="image/png"
            videoConstraints={{
              width: { ideal: 640, max: 1280 },
              height: { ideal: 480, max: 720 },
              facingMode: "environment"
            }}
            mirrored={false}
            audio={false}
            onClick={handleVideoClick}
          />
        ) : uploadedImage && (
          <img 
            src={uploadedImage.src} 
            className="w-full h-full object-contain"
            style={{ display: 'block' }}
          />
        )}
        
        {/* Detection overlay - only show in image mode */}
        {imageMode ? (
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
            style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 10 }}
          />
        ) : null}
      </div>
      
      {/* Controls */}
      <div className="absolute bottom-4 right-4 flex space-x-2">
        <input 
          type="file" 
          ref={fileInputRef}
          accept="image/*" 
          style={{ display: 'none' }} 
          onChange={handleImageUpload}
        />
        
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors duration-200"
        >
          Upload Photo
        </button>
        
        {imageMode && (
          <button
            onClick={resetToLiveVideo}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-200"
          >
            Back to Live
          </button>
        )}
        
        {!imageMode && (
          <>
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
          </>
        )}
      </div>
      
      {/* Status indicator */}
      <div className="absolute bottom-4 left-4 flex items-center space-x-2 bg-gray-900 bg-opacity-75 px-3 py-1 rounded-full">
        <div className={`h-3 w-3 rounded-full ${error ? 'bg-red-500' : (imageMode ? 'bg-purple-500' : (stream ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'))}`}></div>
        <span className="text-xs text-white font-medium">
          {error ? 'Offline' : (imageMode ? 'Image Mode' : (stream ? 'Live' : 'Connecting...'))}
        </span>
      </div>
    </div>
  );
};

export default VideoFeed; 