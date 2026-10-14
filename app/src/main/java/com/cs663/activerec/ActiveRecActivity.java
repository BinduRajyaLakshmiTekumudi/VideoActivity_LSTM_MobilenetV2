package com.cs663.activerec;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * ActiveRec screen: video activity recognition with MobileNetV2 (per frame) + LSTM.
 *  - "Pick Video": choose a clip from the phone; 40 frames are sampled across it.
 *  - "Start Live": rear camera; one frame every FRAME_INTERVAL_MS goes into a 40-frame
 *    sliding window, and the LSTM predicts every PREDICT_EVERY new frames.
 */
public class ActiveRecActivity extends AppCompatActivity {

    // 40 frames x 160 ms = about 6.4 s of motion
    private static final long FRAME_INTERVAL_MS = 160L;
    private static final int PREDICT_EVERY = 10;

    private PreviewView previewView;
    private TextView statusText;
    private TextView resultText;
    private Button liveButton;
    private Button videoButton;

    private volatile ActiveRecClassifier classifier;
    private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService workExecutor = Executors.newSingleThreadExecutor();
    private ProcessCameraProvider cameraProvider;

    private volatile boolean liveRunning = false;
    private long lastFrameTime = 0L;
    private int framesSincePrediction = 0;

    private final ActivityResultLauncher<String> pickVideo =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) classifyVideo(uri);
            });

    private final ActivityResultLauncher<String> requestCamera =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) startLive();
                else statusText.setText(R.string.camera_denied);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_active_rec);
        setTitle("ActiveRec");

        previewView = findViewById(R.id.previewView);
        statusText = findViewById(R.id.statusText);
        resultText = findViewById(R.id.resultText);
        liveButton = findViewById(R.id.liveButton);
        videoButton = findViewById(R.id.videoButton);
        liveButton.setEnabled(false);
        videoButton.setEnabled(false);
        statusText.setText(R.string.loading_models);

        // Load the two TFLite models off the UI thread
        workExecutor.execute(() -> {
            try {
                ActiveRecClassifier c = new ActiveRecClassifier(this);
                classifier = c;
                runOnUiThread(() -> {
                    statusText.setText(getString(R.string.models_ready, c.getLabels().size()));
                    liveButton.setEnabled(true);
                    videoButton.setEnabled(true);
                });
            } catch (Exception e) {
                runOnUiThread(() -> statusText.setText(getString(R.string.model_error, e.getMessage())));
            }
        });

        videoButton.setOnClickListener(v -> {
            if (liveRunning) stopLive();
            pickVideo.launch("video/*");
        });

        liveButton.setOnClickListener(v -> {
            if (liveRunning) {
                stopLive();
            } else if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED) {
                startLive();
            } else {
                requestCamera.launch(Manifest.permission.CAMERA);
            }
        });
    }

    // ---------------- Video file mode ----------------

    private void classifyVideo(Uri uri) {
        final ActiveRecClassifier c = classifier;
        if (c == null) return;
        resultText.setText("");
        videoButton.setEnabled(false);
        liveButton.setEnabled(false);

        workExecutor.execute(() -> {
            long start = SystemClock.elapsedRealtime();
            try {
                ActiveRecClassifier.Prediction prediction = c.classifyVideo(this, uri, done ->
                        runOnUiThread(() -> statusText.setText(getString(R.string.processing_frames,
                                done, ActiveRecClassifier.SEQUENCE_LENGTH))));
                double secs = (SystemClock.elapsedRealtime() - start) / 1000.0;
                runOnUiThread(() -> {
                    resultText.setText(prediction.display(3));
                    statusText.setText(getString(R.string.video_done, secs));
                });
            } catch (Exception e) {
                runOnUiThread(() -> statusText.setText(getString(R.string.video_error, e.getMessage())));
            } finally {
                runOnUiThread(() -> {
                    videoButton.setEnabled(true);
                    liveButton.setEnabled(true);
                });
            }
        });
    }

    // ---------------- Live camera mode ----------------

    private void startLive() {
        final ActiveRecClassifier c = classifier;
        if (c == null) return;
        c.resetLive();
        framesSincePrediction = 0;
        lastFrameTime = 0L;
        resultText.setText("");

        ListenableFuture<ProcessCameraProvider> providerFuture = ProcessCameraProvider.getInstance(this);
        providerFuture.addListener(() -> {
            try {
                ProcessCameraProvider provider = providerFuture.get();
                cameraProvider = provider;

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build();
                analysis.setAnalyzer(analysisExecutor, this::analyzeFrame);

                provider.unbindAll();
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis);

                liveRunning = true;
                liveButton.setText(R.string.stop_live);
                statusText.setText(getString(R.string.collecting, 0, ActiveRecClassifier.SEQUENCE_LENGTH));
            } catch (Exception e) {
                statusText.setText("Camera error: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void stopLive() {
        liveRunning = false;
        if (cameraProvider != null) cameraProvider.unbindAll();
        liveButton.setText(R.string.start_live);
        statusText.setText(R.string.live_stopped);
    }

    /** Runs on analysisExecutor for every camera frame. */
    private void analyzeFrame(ImageProxy image) {
        try {
            ActiveRecClassifier c = classifier;
            if (c == null || !liveRunning) return;

            long now = SystemClock.elapsedRealtime();
            if (now - lastFrameTime < FRAME_INTERVAL_MS) return;
            lastFrameTime = now;

            Bitmap bitmap = rotate(image.toBitmap(), image.getImageInfo().getRotationDegrees());
            int count = c.addLiveFrame(bitmap);
            if (count < ActiveRecClassifier.SEQUENCE_LENGTH) {
                runOnUiThread(() -> statusText.setText(getString(R.string.collecting,
                        count, ActiveRecClassifier.SEQUENCE_LENGTH)));
                return;
            }

            framesSincePrediction++;
            if (framesSincePrediction == 1 || framesSincePrediction % PREDICT_EVERY == 0) {
                long t0 = SystemClock.elapsedRealtime();
                ActiveRecClassifier.Prediction prediction = c.predictLive();
                if (prediction == null) return;
                long ms = SystemClock.elapsedRealtime() - t0;
                runOnUiThread(() -> {
                    resultText.setText(prediction.display(3));
                    statusText.setText(getString(R.string.live_status, ms));
                });
            }
        } finally {
            image.close();   // must always close, or CameraX stops sending frames
        }
    }

    private static Bitmap rotate(Bitmap source, int degrees) {
        if (degrees == 0) return source;
        Matrix m = new Matrix();
        m.postRotate(degrees);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), m, true);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        liveRunning = false;
        analysisExecutor.shutdown();
        workExecutor.execute(() -> {
            ActiveRecClassifier c = classifier;
            if (c != null) c.close();
        });
        workExecutor.shutdown();
    }
}
