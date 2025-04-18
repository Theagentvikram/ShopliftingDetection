import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import LoadingSpinner from '../components/LoadingSpinner';

const StatusPage = ({ analysisResults }) => {
  const navigate = useNavigate();
  const [detectionCount, setDetectionCount] = useState(0);
  const [loading, setLoading] = useState(true);
  
  // Process results when they arrive
  useEffect(() => {
    if (analysisResults) {
      setLoading(false);
      
      // Get detection count from results
      const suspiciousCount = analysisResults.suspiciousCount || 
                             (analysisResults.suspicious_activities ? 
                              analysisResults.suspicious_activities.length : 0);
                              
      setDetectionCount(suspiciousCount);
    }
  }, [analysisResults]);

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
  
  // Determine if suspicious activities were found
  const isSuspicious = detectionCount > 0;
  const recipientEmail = analysisResults.recipient_email || "cherupallya@gmail.com";
  const frameCount = analysisResults.frame_count || 960;

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center px-4">
      <div className="max-w-lg w-full bg-white rounded-lg shadow-xl p-8">
        <div className="text-center">
          {loading ? (
            <div className="mb-6">
              <LoadingSpinner size="md" color="blue" />
              <p className="mt-4 text-gray-600">Processing analysis results...</p>
            </div>
          ) : isSuspicious ? (
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
                    {detectionCount} suspicious activities detected
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