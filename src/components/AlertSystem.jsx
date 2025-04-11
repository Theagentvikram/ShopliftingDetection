import React, { useEffect, useRef } from 'react';

const AlertSystem = ({ alert }) => {
  const audioRef = useRef(null);

  useEffect(() => {
    if (alert) {
      playAlertSound();
      sendNotification(alert);
    }
  }, [alert]);

  const playAlertSound = () => {
    if (audioRef.current) {
      audioRef.current.play().catch(error => {
        console.error('Error playing alert sound:', error);
      });
    }
  };

  const sendNotification = (alert) => {
    // Check if browser notifications are supported and permitted
    if (!("Notification" in window)) {
      console.log("This browser does not support notifications");
      return;
    }

    if (Notification.permission === "granted") {
      new Notification("Security Alert", {
        body: `${alert.type} detected at location (${Math.round(alert.location.x)}, ${Math.round(alert.location.y)})`,
        icon: "/alert-icon.png"
      });
    } else if (Notification.permission !== "denied") {
      Notification.requestPermission().then(permission => {
        if (permission === "granted") {
          sendNotification(alert);
        }
      });
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <audio ref={audioRef} src="/alert-sound.mp3" />
      
      {alert && (
        <div className="bg-red-600 text-white p-4 rounded-lg shadow-lg animate-bounce">
          <h3 className="font-bold text-lg">Security Alert</h3>
          <p className="mt-1">
            {alert.type === 'loitering' ? 'Suspicious loitering detected' : 'Potential concealment behavior detected'}
          </p>
          <p className="text-sm mt-1">
            Location: ({Math.round(alert.location.x)}, {Math.round(alert.location.y)})
          </p>
        </div>
      )}
    </div>
  );
};

export default AlertSystem; 