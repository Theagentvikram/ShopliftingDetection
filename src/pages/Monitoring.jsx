import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import VideoFeed from '../components/VideoFeed';
import BehaviorAnalysis from '../components/BehaviorAnalysis';
import AlertSystem from '../components/AlertSystem';
import CheckCircleIcon from "../components/CheckCircleIcon";
import WarningIcon from "../components/WarningIcon";
import Header from "../components/Header";

const Monitoring = ({ onAnalysisComplete }) => {
  const navigate = useNavigate();
  const [detections, setDetections] = useState([]);
  const [currentAlert, setCurrentAlert] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState('main');
  const [uploadStatus, setUploadStatus] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [activeTab, setActiveTab] = useState('live'); // 'live' or 'upload'
  const [isLoading, setIsLoading] = useState(false);
  const [processingComplete, setProcessingComplete] = useState(false);

  // Reset state when changing tabs
  useEffect(() => {
    if (activeTab === 'upload') {
      // Clear any previous results when switching to upload tab
      if (processingComplete) {
        setProcessingComplete(false);
      }
    }
  }, [activeTab]);

  const cameras = [
    { id: 'main', name: 'Main Entrance' },
    { id: 'checkout', name: 'Checkout Area' },
    { id: 'storage', name: 'Storage Room' }
  ];

  const handleDetections = (newDetections) => {
    setDetections(newDetections);
  };

  const handleAlert = (alert) => {
    setCurrentAlert(alert);
    // Clear alert after 5 seconds
    setTimeout(() => {
      setCurrentAlert(null);
    }, 5000);
  };

  const resetUploadState = () => {
    setUploadStatus('');
    setAnalysisResults(null);
    setUploadError(null);
    setProcessingComplete(false);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Reset any previous state
    resetUploadState();
    setIsLoading(true);
    
    console.log("Starting upload with file:", file.name);
    setUploadStatus('Uploading...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      console.log("Sending request to backend...");
      const response = await fetch('http://localhost:8111/analyze-video', {
        method: 'POST',
        body: formData,
      });
      
      console.log("Response received:", response.status);
      
      if (!response.ok) {
        throw new Error(`Upload failed with status ${response.status}`);
      }

      const data = await response.json();
      console.log("Backend response:", data);
      
      // Extract a frame image if available (from the first detection with a thumbnail, if any)
      let frameImage = null;
      if (data.thumbnail_base64) {
        frameImage = `data:image/jpeg;base64,${data.thumbnail_base64}`;
      } else {
        // If backend doesn't provide a thumbnail, create a placeholder image
        frameImage = createPlaceholderImage(640, 480);
      }
      
      // Create analysis results object
      const results = {
        suspiciousCount: data.suspicious_count || 0,
        detections: data.detections || [],
        suspicious_activities: data.suspicious_activities || [],
        frame_count: data.frame_count,
        timestamp: new Date().toISOString(),
        recipient_email: data.recipient_email || "cherupallya@gmail.com",
        frameImage: frameImage
      };

      // Update state with results
      setAnalysisResults(results);
      setProcessingComplete(true);
      setIsLoading(false);
      
      // Show appropriate message based on results
      if (results.suspicious_activities && results.suspicious_activities.length > 0) {
        setUploadStatus('⚠️ SUSPICIOUS ACTIVITY DETECTED!');
        handleAlert({
          type: 'warning',
          message: 'Suspicious activity detected in uploaded video'
        });
      } else {
        setUploadStatus('✅ Analysis complete - No suspicious activity detected');
      }

      // Pass results to App component
      if (onAnalysisComplete) {
        onAnalysisComplete(results);
      }

      // Navigate to status page
      navigate('/status');
      
      // Clear the file input
      event.target.value = '';

    } catch (error) {
      console.error("Upload error:", error);
      setUploadError(error.message || "An error occurred during upload");
      setUploadStatus("❌ Upload failed");
      setIsLoading(false);
      
      // Clear the file input on error
      event.target.value = '';
    }
  };

  // Helper function to create a placeholder image if we don't get a thumbnail from backend
  const createPlaceholderImage = (width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    
    // Fill with light gray
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, width, height);
    
    // Add text
    ctx.fillStyle = '#555555';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('No preview available', width/2, height/2);
    
    return canvas.toDataURL('image/jpeg');
  };

  return (
    <div className="p-6 h-full">
      <div className="mb-6 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold text-gray-800">Surveillance System</h1>
          <p className="text-gray-600">Theft detection and monitoring</p>
        </div>
      </div>

      {/* Mode Selection Tabs */}
      <div className="mb-6">
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab('live')}
            className={`py-3 px-6 font-medium text-sm rounded-t-lg ${
              activeTab === 'live'
                ? 'bg-blue-600 text-white border-b-2 border-blue-600'
                : 'text-gray-600 hover:text-gray-800 bg-gray-100'
            }`}
          >
            Live Monitoring (CCTV)
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`py-3 px-6 font-medium text-sm rounded-t-lg ${
              activeTab === 'upload'
                ? 'bg-blue-600 text-white border-b-2 border-blue-600'
                : 'text-gray-600 hover:text-gray-800 bg-gray-100'
            }`}
          >
            Video Upload Analysis
          </button>
        </div>
      </div>

      {/* Live Monitoring View */}
      {activeTab === 'live' && (
        <>
          {/* Camera Selection */}
          <div className="mb-6 bg-white rounded-lg shadow-lg p-4">
            <h2 className="text-lg font-semibold mb-3">Camera Selection</h2>
            <div className="flex space-x-2">
              {cameras.map(camera => (
                <button
                  key={camera.id}
                  onClick={() => setSelectedCamera(camera.id)}
                  className={`px-4 py-2 rounded-lg transition-colors duration-200 ${
                    selectedCamera === camera.id
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  {camera.name}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-6 h-[calc(100vh-18rem)]">
            {/* Main Video Feed */}
            <div className="col-span-2 bg-white rounded-lg shadow-lg overflow-hidden">
              <div className="w-full h-full">
                <VideoFeed onDetection={handleDetections} />
              </div>
            </div>

            {/* Stats Panel */}
            <div className="space-y-6 overflow-y-auto">
              {/* Detection Stats */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-lg font-semibold mb-4">Detection Statistics</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-500">Objects Detected</h3>
                    <p className="text-2xl font-bold text-gray-900 mt-1">{detections.length}</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-500">Alert Status</h3>
                    <p className={`text-2xl font-bold mt-1 ${currentAlert ? 'text-red-600' : 'text-green-600'}`}>
                      {currentAlert ? 'ALERT' : 'Normal'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Recent Detections */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-lg font-semibold mb-4">Recent Detections</h2>
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {detections.map((detection, index) => (
                    <div 
                      key={index} 
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                    >
                      <span className="font-medium text-gray-700">{detection.class}</span>
                      <span className="text-sm text-gray-500">
                        {Math.round(detection.score * 100)}% confidence
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Quick Actions */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
                <div className="grid grid-cols-2 gap-4">
                  <button 
                    onClick={() => alert('Recording started')}
                    className="w-full px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors duration-200"
                  >
                    Start Recording
                  </button>
                  <button 
                    onClick={() => alert('Screenshot taken')}
                    className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-200"
                  >
                    Take Screenshot
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Video Upload Analysis View */}
      {activeTab === 'upload' && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Upload Video for Analysis</h2>
          
          {!processingComplete ? (
            <div className="mb-6">
              <p className="text-gray-600 mb-4">
                Upload a video file to be analyzed for suspicious activity. The system will process the video
                and return results showing any detected suspicious behavior.
              </p>
              <div className="flex items-center space-x-4">
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  disabled={isLoading}
                  className={`block w-full text-sm ${isLoading ? 'opacity-50 cursor-not-allowed' : ''} text-gray-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-full file:border-0
                    file:text-sm file:font-semibold
                    file:bg-blue-50 file:text-blue-700
                    hover:file:bg-blue-100`}
                />
                <div className={`text-sm ${
                  uploadStatus.includes('failed') || uploadStatus.includes('❌') 
                    ? 'text-red-500'
                    : uploadStatus.includes('⚠️')
                      ? 'text-orange-500 font-bold'
                      : 'text-green-500'
                }`}>
                  {uploadStatus}
                </div>
              </div>
              
              {/* Upload processing indicator */}
              {isLoading && (
                <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <p className="text-blue-600">Processing video... This may take a few moments.</p>
                  </div>
                  <p className="text-sm text-blue-500 mt-2">
                    Please be patient as we analyze your video for suspicious activity.
                  </p>
                </div>
              )}
              
              {/* Error message */}
              {uploadError && (
                <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
                  <strong>Error:</strong> {uploadError}
                  <div className="mt-1">
                    <p>Make sure the backend server is running at <code>http://localhost:8111</code></p>
                  </div>
                </div>
              )}
            </div>
          ) : null}

          {/* Analysis Results */}
          {analysisResults && (
            <div className={processingComplete ? "mt-0" : "mt-6"}>
              {processingComplete && (
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-xl font-semibold text-green-600">
                    ✅ Analysis Complete
                  </h3>
                  <button
                    onClick={resetUploadState}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-200"
                  >
                    Upload Another Video
                  </button>
                </div>
              )}
              
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-sm font-medium text-gray-500">Video Frames</h3>
                  <p className="text-2xl font-bold text-gray-900 mt-1">
                    {analysisResults.frame_count || 0}
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-sm font-medium text-gray-500">Detections</h3>
                  <p className="text-2xl font-bold text-gray-900 mt-1">
                    {analysisResults.detections ? analysisResults.detections.length : 0}
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-sm font-medium text-gray-500">Suspicious Activities</h3>
                  <p className="text-2xl font-bold text-gray-900 mt-1">
                    {analysisResults.suspicious_activities ? analysisResults.suspicious_activities.length : 0}
                  </p>
                </div>
              </div>
              
              {analysisResults.suspicious_activities && analysisResults.suspicious_activities.length > 0 ? (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <h3 className="text-md font-semibold text-red-700 mb-2">
                    ⚠️ Suspicious Activities Detected
                  </h3>
                  <ul className="space-y-2 max-h-[300px] overflow-y-auto">
                    {analysisResults.suspicious_activities.map((activity, idx) => (
                      <li key={idx} className="p-2 bg-white border border-red-100 rounded">
                        <div className="flex justify-between items-center">
                          <span className="font-medium">{activity.details || activity.type}</span>
                          <span className="text-sm text-gray-500">Frame: {activity.frame}</span>
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          Time: {activity.timestamp !== undefined ? activity.timestamp.toFixed(2) : '0.0'}s
                        </div>
                      </li>
                    ))}
                  </ul>
                  <div className="mt-3 text-sm text-red-600">
                    Email alert sent to {analysisResults.recipient_email || "configured email"}
                  </div>
                </div>
              ) : (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-green-700">✅ No suspicious activities detected</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Behavior Analysis (invisible component) */}
      <BehaviorAnalysis
        detections={detections}
        onAlert={handleAlert}
      />

      {/* Alert System */}
      <AlertSystem alert={currentAlert} />
    </div>
  );
};

export default Monitoring;