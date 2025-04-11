import React, { useState } from 'react';
import VideoFeed from '../components/VideoFeed';
import BehaviorAnalysis from '../components/BehaviorAnalysis';
import AlertSystem from '../components/AlertSystem';

const Monitoring = () => {
  const [detections, setDetections] = useState([]);
  const [currentAlert, setCurrentAlert] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState('main');

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

  return (
    <div className="p-6 h-full">
      <div className="mb-6 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold text-gray-800">Live Surveillance</h1>
          <p className="text-gray-600">Real-time theft detection and monitoring</p>
        </div>
        
        {/* Camera Selection */}
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

      <div className="grid grid-cols-3 gap-6 h-[calc(100vh-12rem)]">
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