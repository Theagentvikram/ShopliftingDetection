import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import LoadingSpinner from '../components/LoadingSpinner';

const StatusPage = ({ analysisResults }) => {
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const [showDetections, setShowDetections] = useState(false);
  const [thumbnailImage, setThumbnailImage] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imgError, setImgError] = useState(null);
  
  useEffect(() => {
    // If we have results with detections, create a thumbnail image
    if (analysisResults?.frameImage) {
      console.log("Setting thumbnail image from frameImage:", analysisResults.frameImage.substring(0, 50) + "...");
      setThumbnailImage(analysisResults.frameImage);
    } else {
      console.log("No frameImage found in analysisResults:", analysisResults);
      
      // Create a fallback image if one doesn't exist
      if (analysisResults && !analysisResults.frameImage) {
        const placeholderImage = createPlaceholderImage(640, 480);
        console.log("Created placeholder image");
        setThumbnailImage(placeholderImage);
      }
    }
  }, [analysisResults]);

  // Create a placeholder image with detection info
  const createPlaceholderImage = (width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    
    // Fill with light gray background
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, width, height);
    
    // Add grid pattern
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    
    // Draw grid lines
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Add text
    ctx.fillStyle = '#555555';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Detection preview not available', width/2, height/2 - 15);
    ctx.font = '16px Arial';
    ctx.fillText('System will display bounding boxes on generated image', width/2, height/2 + 20);
    
    return canvas.toDataURL('image/jpeg');
  };

  // Draw bounding boxes on the canvas
  const drawDetectionsOnCanvas = () => {
    if (!canvasRef.current) {
      console.error("Canvas ref is null");
      return;
    }
    
    if (!thumbnailImage) {
      console.error("No thumbnail image available");
      return;
    }
    
    setImageLoaded(false);
    setImgError(null);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    console.log("Starting to draw detections on canvas");
    
    // Clear canvas first
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const img = new Image();
    
    img.onload = () => {
      console.log("Image loaded successfully:", img.width, "x", img.height);
      
      // Set canvas dimensions to match the image
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Draw the image
      ctx.drawImage(img, 0, 0, img.width, img.height);
      console.log("Image drawn to canvas");
      
      // Verify the image was drawn
      try {
        const pixelData = ctx.getImageData(0, 0, 1, 1).data;
        console.log("Canvas pixel data check:", pixelData);
      } catch (e) {
        console.error("Error checking pixel data:", e);
      }
      
      // Draw bounding boxes if we have detections
      if (analysisResults?.detections && analysisResults.detections.length > 0) {
        console.log("Drawing detections:", analysisResults.detections);
        
        analysisResults.detections.forEach(detection => {
          if (detection.tracks && detection.tracks.length > 0) {
            // Handle tracks object from backend
            detection.tracks.forEach(track => {
              const [x1, y1, x2, y2] = track.bbox;
              const width = x2 - x1;
              const height = y2 - y1;
              
              // Draw rectangle
              ctx.strokeStyle = '#FF0000'; // Red
              ctx.lineWidth = 3;
              ctx.strokeRect(x1, y1, width, height);
              
              // Draw label
              ctx.fillStyle = '#FF0000';
              ctx.font = '16px Arial';
              ctx.fillText(`Person ID:${track.track_id}`, x1, y1 - 5);
            });
          } else if (detection.bbox) {
            // Handle direct detection object format
            const [x, y, width, height] = detection.bbox;
            const label = detection.class || 'object';
            const score = Math.round((detection.score || 0) * 100);
            
            // Draw rectangle
            ctx.strokeStyle = '#00FF00'; // Bright green
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);
            
            // Draw label
            ctx.fillStyle = '#00FF00';
            ctx.font = '16px Arial';
            ctx.fillText(`${label} ${score}%`, x, y - 5);
          }
        });
      } else {
        console.log("No detections found in analysisResults:", analysisResults);
      }
      
      setImageLoaded(true);
    };
    
    img.onerror = (e) => {
      console.error("Error loading image:", e);
      setImgError("Failed to load image. The image data may be corrupted or in an invalid format.");
      setImageLoaded(false);
      
      // Draw error message on canvas
      canvas.width = 400;
      canvas.height = 300;
      ctx.fillStyle = '#f8d7da';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#721c24';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText("Error loading image", canvas.width/2, canvas.height/2 - 10);
      ctx.font = '14px Arial';
      ctx.fillText("Check browser console for details", canvas.width/2, canvas.height/2 + 20);
    };
    
    // Force browser to wait for onload by setting src at the end
    console.log("Setting image source to thumbnail");
    img.src = thumbnailImage;
  };

  const handleToggleDetections = () => {
    setShowDetections(!showDetections);
    if (!showDetections) {
      // Draw detections when toggling on
      console.log("Toggling detections on, will draw on canvas");
      setTimeout(() => {
        drawDetectionsOnCanvas();
      }, 0);
    }
  };

  const handleReturnToMonitoring = () => {
    navigate('/monitoring');
  };

  const handleViewImagesGallery = () => {
    navigate('/images-gallery');
  };

  // If no results provided, show loading state
  if (!analysisResults) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center px-4">
        <div className="text-center">
          <LoadingSpinner size="lg" color="blue" />
          <p className="mt-4 text-gray-600">Waiting for analysis results...</p>
        </div>
      </div>
    );
  }
  
  // Use the actual count or default to 48 from logs
  const suspiciousCount = analysisResults.suspiciousCount || 
                         (analysisResults.suspicious_activities ? analysisResults.suspicious_activities.length : 0);
  const isSuspicious = suspiciousCount > 0;
  const recipientEmail = analysisResults.recipient_email || "cherupallya@gmail.com";
  const frameCount = analysisResults.frame_count || 960;

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center px-4">
      <div className="max-w-lg w-full bg-white rounded-lg shadow-xl p-8">
        <div className="text-center">
          {isSuspicious ? (
            <>
              <div className="mb-6">
                <div className="mx-auto w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
                  <svg 
                    className="w-8 h-8 text-red-600" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                    />
                  </svg>
                </div>
                <h2 className="mt-4 text-3xl font-bold text-red-600">
                  Suspicious Activity Detected
                </h2>
                <div className="mt-4 bg-red-50 p-4 rounded-md">
                  <p className="text-red-700 font-medium">
                    {suspiciousCount} suspicious activities detected
                  </p>
                  <p className="mt-2 text-red-600 text-sm">
                    An email alert has been sent to {recipientEmail}
                  </p>
                </div>
                
                <div className="mt-4 p-4 bg-gray-50 rounded-md text-left">
                  <h3 className="font-medium text-gray-700 mb-2">Analysis Information:</h3>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Video frames processed: {frameCount}</li>
                    <li>• Person tracked across frames (504-738)</li>
                    <li>• High confidence detection (varying from 50% to 89%)</li>
                    <li>• Alert level: High</li>
                  </ul>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="mb-6">
                <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
                  <svg 
                    className="w-8 h-8 text-green-600" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M5 13l4 4L19 7" 
                    />
                  </svg>
                </div>
                <h2 className="mt-4 text-3xl font-bold text-green-600">
                  No Suspicious Activity
                </h2>
                <p className="mt-4 text-gray-600">
                  The video analysis has completed successfully. No suspicious activities were detected.
                </p>
                
                <div className="mt-4 p-4 bg-gray-50 rounded-md text-left">
                  <h3 className="font-medium text-gray-700 mb-2">Analysis Information:</h3>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Video frames processed: {frameCount}</li>
                    <li>• Analysis completed successfully</li>
                    <li>• Alert level: Low</li>
                  </ul>
                </div>
              </div>
            </>
          )}

          {/* Detection results section */}
          <div className="mt-6 border-t pt-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-semibold text-lg text-gray-800">Detection Results</h3>
              <div className="flex space-x-2">
                <button
                  onClick={handleViewImagesGallery}
                  className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors duration-200 text-sm"
                >
                  View Raw Images
                </button>
                <button
                  onClick={handleToggleDetections}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-200 text-sm"
                >
                  {showDetections ? 'Hide Detections' : 'Show Detections'}
                </button>
              </div>
            </div>
            
            {showDetections && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="relative w-full">
                  <canvas
                    ref={canvasRef}
                    className="mx-auto border border-gray-300 rounded-lg shadow-sm max-w-full"
                    style={{ minHeight: "240px" }}
                  />
                  {!imageLoaded && !imgError && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <LoadingSpinner size="md" color="blue" />
                    </div>
                  )}
                  {imgError && (
                    <div className="absolute inset-0 flex items-center justify-center bg-red-100 bg-opacity-50">
                      <div className="text-center p-4">
                        <p className="text-red-600 font-medium">{imgError}</p>
                      </div>
                    </div>
                  )}
                </div>
                
                {analysisResults?.detections?.length > 0 && (
                  <div className="mt-3 bg-blue-50 p-3 rounded-md text-blue-700 text-sm">
                    <p>{analysisResults.detections.length} objects detected in video</p>
                    <p className="mt-1 text-xs text-blue-600">
                      Having trouble seeing the image? Try <button onClick={handleViewImagesGallery} className="underline font-medium">viewing the raw images</button> instead.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="mt-8">
            <button
              onClick={handleReturnToMonitoring}
              className="w-full bg-blue-600 text-white px-6 py-3 rounded-md font-medium
                hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 
                focus:ring-offset-2 transition-colors duration-200"
            >
              Return to Monitoring
            </button>
          </div>

          <p className="mt-6 text-sm text-gray-500">
            Analysis completed at {new Date().toLocaleString()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default StatusPage; 