import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Box, Button, CircularProgress, Typography, Alert, LinearProgress } from '@mui/material';

const BACKEND_URL = 'http://localhost:8111';

const VideoAnalysis = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [results, setResults] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith('video/')) {
            setSelectedFile(file);
            setError(null);
            setResults(null);
            setUploadProgress(0);
        } else {
            setError('Please select a valid video file (MP4, AVI, or MOV)');
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a video file first');
            return;
        }

        setLoading(true);
        setError(null);
        setUploadProgress(0);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await axios.post(`${BACKEND_URL}/analyze-video`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const progress = (progressEvent.loaded / progressEvent.total) * 100;
                    setUploadProgress(progress);
                },
            });

            setResults(response.data);
            visualizeResults(response.data);
        } catch (err) {
            console.error('Upload error:', err);
            setError(
                err.response?.data?.detail || 
                'Error processing video. Please make sure the backend server is running and try again.'
            );
        } finally {
            setLoading(false);
        }
    };

    const visualizeResults = (results) => {
        if (!videoRef.current || !canvasRef.current || !results) {
            console.error('Missing required elements for visualization:', {
                videoRef: !!videoRef.current,
                canvasRef: !!canvasRef.current,
                results: !!results
            });
            return;
        }

        console.log('Visualizing results:', results);
        
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        const updateCanvasSize = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            console.log(`Canvas resized to ${canvas.width}x${canvas.height}`);
        };

        // Update canvas size when video metadata is loaded
        video.addEventListener('loadedmetadata', updateCanvasSize);
        
        let animationFrameId;

        const drawFrame = () => {
            if (video.paused || video.ended) {
                cancelAnimationFrame(animationFrameId);
                return;
            }

            // Calculate current frame number based on video time and fps
            const currentFrameNumber = Math.floor(video.currentTime * results.fps);
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Find detections for current frame
            const frameData = results.detections.find(d => d.frame === currentFrameNumber);
            
            // Check if we have any suspicious activities at this frame
            const activities = results.suspicious_activities && results.suspicious_activities.length > 0 ? 
                results.suspicious_activities.filter(a => a.frame === currentFrameNumber) : [];
            
            // Draw suspicious activity indicator if any exist at this frame
            if (activities && activities.length > 0) {
                // Draw a red border around the frame to indicate suspicious activity
                ctx.strokeStyle = '#ff0000';
                ctx.lineWidth = 10;
                ctx.strokeRect(0, 0, canvas.width, canvas.height);
                
                // Draw text for each activity
                ctx.fillStyle = '#ff0000';
                ctx.font = 'bold 20px Arial';
                activities.forEach((activity, index) => {
                    ctx.fillText(
                        `ALERT: ${activity.type}`, 
                        20, 
                        30 + (index * 30)
                    );
                });
            }
            
            // Draw bounding boxes for detections
            if (frameData && frameData.tracks && frameData.tracks.length > 0) {
                console.log(`Drawing ${frameData.tracks.length} tracks for frame ${currentFrameNumber}`);
                
                frameData.tracks.forEach(track => {
                    try {
                        // Ensure bbox is valid
                        if (!track.bbox || track.bbox.length !== 4) {
                            console.warn('Invalid bbox format:', track.bbox);
                            return;
                        }
                        
                        const [x1, y1, x2, y2] = track.bbox;
                        
                        // Calculate width and height
                        const width = x2 - x1;
                        const height = y2 - y1;
                        
                        if (isNaN(width) || isNaN(height) || width <= 0 || height <= 0) {
                            console.warn('Invalid bbox dimensions:', track.bbox);
                            return;
                        }
                        
                        // Draw bounding box
                        ctx.strokeStyle = track.class === 0 ? '#00ff00' : '#ff0000';
                        ctx.lineWidth = 3;
                        ctx.strokeRect(x1, y1, width, height);

                        // Draw track ID with background
                        const label = `ID: ${track.track_id}`;
                        ctx.fillStyle = track.class === 0 ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)';
                        const textWidth = ctx.measureText(label).width;
                        ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);
                        
                        ctx.fillStyle = '#FFFFFF';
                        ctx.font = 'bold 16px Arial';
                        ctx.fillText(label, x1 + 5, y1 - 7);
                    } catch (err) {
                        console.error('Error drawing track:', err, track);
                    }
                });
            }

            animationFrameId = requestAnimationFrame(drawFrame);
        };

        // Start drawing when video plays
        video.addEventListener('play', () => {
            console.log('Video started playing, beginning visualization');
            drawFrame();
        });

        // Cleanup
        return () => {
            video.removeEventListener('loadedmetadata', updateCanvasSize);
            cancelAnimationFrame(animationFrameId);
        };
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (videoRef.current) {
                videoRef.current.pause();
                videoRef.current.src = '';
            }
        };
    }, []);

    return (
        <Box sx={{ p: 3 }}>
            <input
                accept="video/mp4,video/avi,video/quicktime"
                type="file"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
                id="video-upload"
            />
            <label htmlFor="video-upload">
                <Button variant="contained" component="span" sx={{ mb: 2 }}>
                    Select Video
                </Button>
            </label>

            {selectedFile && (
                <Box sx={{ mb: 2 }}>
                    <Typography variant="body1">
                        Selected: {selectedFile.name}
                    </Typography>
                    <Button
                        variant="contained"
                        onClick={handleUpload}
                        disabled={loading}
                        sx={{ mt: 1 }}
                    >
                        {loading ? <CircularProgress size={24} /> : 'Analyze Video'}
                    </Button>
                </Box>
            )}

            {loading && (
                <Box sx={{ width: '100%', mt: 2 }}>
                    <LinearProgress variant="determinate" value={uploadProgress} />
                    <Typography variant="body2" color="text.secondary" align="center">
                        {uploadProgress < 100 
                            ? `Uploading: ${Math.round(uploadProgress)}%`
                            : 'Processing video...'}
                    </Typography>
                </Box>
            )}

            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            {results && (
                <Box sx={{ mt: 2 }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>
                        Analysis Results
                    </Typography>
                    
                    <Box sx={{ position: 'relative', width: '100%', maxWidth: '800px' }}>
                        <video
                            ref={videoRef}
                            src={URL.createObjectURL(selectedFile)}
                            controls
                            style={{ width: '100%' }}
                        />
                        <canvas
                            ref={canvasRef}
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                pointerEvents: 'none',
                            }}
                        />
                    </Box>

                    {results.suspicious_activities.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="h6" color="error">
                                Suspicious Activities Detected
                            </Typography>
                            {results.suspicious_activities.map((activity, index) => (
                                <Typography key={index} variant="body1" sx={{ mb: 1 }}>
                                    {`${index + 1}. ${activity.type} at ${parseFloat(activity.timestamp).toFixed(2)}s (frame ${activity.frame}): ${activity.details}`}
                                </Typography>
                            ))}
                        </Box>
                    )}
                </Box>
            )}
        </Box>
    );
};

export default VideoAnalysis;