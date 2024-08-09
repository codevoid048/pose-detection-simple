import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import React, { useEffect, useRef } from 'react';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      await tf.setBackend('webgl');
      await tf.ready();

      const detector = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet, 
        { modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );

      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;

            videoRef.current.onloadedmetadata = () => {
              videoRef.current.play();
              startPoseDetection(detector);
            };
          }
        })
        .catch((err) => console.error("Error accessing webcam: ", err));
    };

    const startPoseDetection = async (detector) => {
      if (videoRef.current && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');

        const updateCanvasSize = () => {
          const videoWidth = videoRef.current.videoWidth;
          const videoHeight = videoRef.current.videoHeight;

          // Set desired size for video
          const desiredWidth = 640; // Adjust as needed
          const desiredHeight = (videoHeight / videoWidth) * desiredWidth;

          videoRef.current.width = desiredWidth;
          videoRef.current.height = desiredHeight;
          canvasRef.current.width = desiredWidth;
          canvasRef.current.height = desiredHeight;
        };

        const detectPose = async () => {
          if (videoRef.current.readyState === 4) {
            updateCanvasSize();
            const poses = await detector.estimatePoses(videoRef.current);
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            drawKeypoints(poses, ctx);

            requestAnimationFrame(detectPose);
          }
        };

        const drawKeypoints = (poses, ctx) => {
          poses.forEach(pose => {
            pose.keypoints.forEach(keypoint => {
              if (keypoint.score > 0.5) {
                const { x, y } = keypoint;
                // Flip keypoints horizontally
                const flippedX = canvasRef.current.width - x * canvasRef.current.width / videoRef.current.videoWidth;
                ctx.beginPath();
                ctx.arc(flippedX, y * canvasRef.current.height / videoRef.current.videoHeight, 5, 0, 2 * Math.PI);
                ctx.fillStyle = 'red';
                ctx.fill();
              }
            });
          });
        };

        detectPose();
      }
    };

    loadModel();
  }, []);

  return (
    <div className="App">
      <video ref={videoRef} autoPlay playsInline style={{ position: 'absolute', top: '10px', left: '10px', transform: 'scaleX(-1)' }} />
      <canvas ref={canvasRef} style={{ position: 'absolute', top: '10px', left: '10px' }} />
    </div>
  );
}

export default App;
