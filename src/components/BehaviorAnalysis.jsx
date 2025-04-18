import React, { useState, useEffect } from 'react';

const BehaviorAnalysis = ({ detections, onAlert }) => {
  const [trackedObjects, setTrackedObjects] = useState(new Map());
  
  // Constants for behavior analysis
  const LOITERING_THRESHOLD = 5000; // 5 seconds
  const CONCEALMENT_DISTANCE = 50; // pixels
  
  useEffect(() => {
    if (!detections) return;

    const currentTime = Date.now();
    const newTrackedObjects = new Map(trackedObjects);

    // Update tracked objects
    detections.forEach(detection => {
      const { bbox, class: objectClass, score } = detection;
      const [x, y, width, height] = bbox;
      const centerX = x + width / 2;
      const centerY = y + height / 2;

      // Generate unique ID based on position (simple tracking)
      const id = `${Math.round(centerX)}-${Math.round(centerY)}`;

      if (!newTrackedObjects.has(id)) {
        newTrackedObjects.set(id, {
          firstSeen: currentTime,
          lastSeen: currentTime,
          positions: [[centerX, centerY]],
          class: objectClass,
          timeStationary: 0,
        });
      } else {
        const trackedObject = newTrackedObjects.get(id);
        trackedObject.lastSeen = currentTime;
        trackedObject.positions.push([centerX, centerY]);
        
        // Keep only last 10 positions
        if (trackedObject.positions.length > 10) {
          trackedObject.positions.shift();
        }

        // Check for loitering
        if (isLoitering(trackedObject)) {
          onAlert({
            type: 'loitering',
            object: trackedObject,
            location: { x: centerX, y: centerY }
          });
        }

        // Check for concealment
        if (isConcealmentBehavior(trackedObject, detections)) {
          onAlert({
            type: 'concealment',
            object: trackedObject,
            location: { x: centerX, y: centerY }
          });
        }
      }
    });

    // Clean up old tracked objects
    for (const [id, object] of newTrackedObjects.entries()) {
      if (currentTime - object.lastSeen > 1000) {
        newTrackedObjects.delete(id);
      }
    }

    setTrackedObjects(newTrackedObjects);
  }, [detections]);

  const isLoitering = (trackedObject) => {
    const { firstSeen, lastSeen, positions } = trackedObject;
    
    // Check if object has been present for longer than threshold
    if (lastSeen - firstSeen < LOITERING_THRESHOLD) return false;

    // Check if object hasn't moved significantly
    const recentPositions = positions.slice(-5);
    if (recentPositions.length < 5) return false;

    const avgX = recentPositions.reduce((sum, pos) => sum + pos[0], 0) / recentPositions.length;
    const avgY = recentPositions.reduce((sum, pos) => sum + pos[1], 0) / recentPositions.length;

    const hasMovedSignificantly = recentPositions.some(pos => {
      const distance = Math.sqrt(Math.pow(pos[0] - avgX, 2) + Math.pow(pos[1] - avgY, 2));
      return distance > 20;
    });

    return !hasMovedSignificantly;
  };

  const isConcealmentBehavior = (trackedObject, allDetections) => {
    const { positions } = trackedObject;
    if (positions.length < 2) return false;

    const currentPos = positions[positions.length - 1];
    
    // Check if person is near objects that could be used for concealment
    return allDetections.some(detection => {
      if (detection.class === 'person') return false;
      
      const [x, y, width, height] = detection.bbox;
      const objectCenterX = x + width / 2;
      const objectCenterY = y + height / 2;
      
      const distance = Math.sqrt(
        Math.pow(currentPos[0] - objectCenterX, 2) + 
        Math.pow(currentPos[1] - objectCenterY, 2)
      );
      
      return distance < CONCEALMENT_DISTANCE;
    });
  };

  return null; // This is a logic-only component
};

export default BehaviorAnalysis; 