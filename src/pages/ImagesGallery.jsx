import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import LoadingSpinner from '../components/LoadingSpinner';

const ImagesGallery = ({ analysisResults }) => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [frames, setFrames] = useState([]);

  useEffect(() => {
    // If we have results, extract frames
    if (analysisResults) {
      setLoading(true);
      
      // If we have a single frameImage, use it
      if (analysisResults.frameImage) {
        setFrames([{
          id: 'main-frame',
          src: analysisResults.frameImage,
          title: 'Detection Frame'
        }]);
      }
      
      // If we have multiple frames (future implementation)
      if (analysisResults.frames && analysisResults.frames.length > 0) {
        const processedFrames = analysisResults.frames.map((frame, index) => ({
          id: `frame-${index}`,
          src: frame.image || frame,
          title: `Frame ${index + 1}`,
          timestamp: frame.timestamp
        }));
        setFrames(prev => [...prev, ...processedFrames]);
      }
      
      // Create a placeholder image if no frames are available
      if ((!analysisResults.frameImage && (!analysisResults.frames || analysisResults.frames.length === 0)) && analysisResults) {
        const placeholderImage = createPlaceholderImage(640, 480);
        setFrames([{
          id: 'placeholder',
          src: placeholderImage,
          title: 'No Frames Available'
        }]);
      }
      
      setLoading(false);
    }
  }, [analysisResults]);

  // Create a placeholder image
  const createPlaceholderImage = (width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    
    // Fill with light gray background
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, width, height);
    
    // Add text
    ctx.fillStyle = '#555555';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('No image frames available', width/2, height/2 - 15);
    ctx.font = '16px Arial';
    ctx.fillText('Detection data exists but without image frames', width/2, height/2 + 20);
    
    return canvas.toDataURL('image/jpeg');
  };

  const handleReturnToAnalysis = () => {
    navigate('/status');
  };

  const handleReturnToMonitoring = () => {
    navigate('/monitoring');
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

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">Detection Images Gallery</h1>
            <p className="text-gray-600">Viewing raw frames without detection overlays</p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={handleReturnToAnalysis}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-200"
            >
              Back to Analysis
            </button>
            <button
              onClick={handleReturnToMonitoring}
              className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors duration-200"
            >
              Return to Monitoring
            </button>
          </div>
        </div>

        {/* Frame information */}
        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
          <h2 className="text-lg font-semibold mb-2">Video Analysis Information</h2>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="bg-gray-50 p-3 rounded-md">
              <span className="font-medium">Frames Processed:</span> {analysisResults.frame_count || 'Unknown'}
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <span className="font-medium">Detections:</span> {analysisResults.detections?.length || 0}
            </div>
            <div className="bg-gray-50 p-3 rounded-md">
              <span className="font-medium">Suspicious Activities:</span> {analysisResults.suspicious_activities?.length || 0}
            </div>
          </div>
        </div>

        {/* Gallery */}
        <div className="bg-white rounded-lg shadow-md p-6">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <LoadingSpinner size="lg" color="blue" />
            </div>
          ) : frames.length > 0 ? (
            <>
              <h2 className="text-lg font-semibold mb-4">Available Frames ({frames.length})</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {frames.map(frame => (
                  <div key={frame.id} className="border rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow">
                    <div className="relative pb-[56.25%]"> {/* 16:9 aspect ratio */}
                      <img 
                        src={frame.src} 
                        alt={frame.title}
                        className="absolute h-full w-full object-contain bg-gray-100" 
                      />
                    </div>
                    <div className="p-3 bg-white">
                      <h3 className="font-medium">{frame.title}</h3>
                      {frame.timestamp && (
                        <p className="text-sm text-gray-500">Timestamp: {frame.timestamp}s</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-400">No frames available from this analysis</p>
            </div>
          )}
        </div>

        <p className="mt-6 text-sm text-gray-500 text-center">
          Images shown are raw frames without detection markers or bounding boxes
        </p>
      </div>
    </div>
  );
};

export default ImagesGallery; 