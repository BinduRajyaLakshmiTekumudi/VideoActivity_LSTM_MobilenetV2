package com.cs663.activerec;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * ActiveRec pipeline (CNN + LSTM), matching the Colab notebook:
 *   frame -> resize 224x224 -> scale to [-1,1] -> MobileNetV2 TFLite -> 1x1280 feature vector
 *   40 feature vectors -> [1, 40, 1280] -> LSTM TFLite -> class probabilities
 *
 * Assets needed in app/src/main/assets (from the notebook, STEP 16):
 *   mobilenet_feature_extractor.tflite, lstm_activity.tflite, labels.txt
 */
public final class ActiveRecClassifier implements AutoCloseable {

    public static final int SEQUENCE_LENGTH = 40;   // must match SEQUENCE_LENGTH in the notebook
    public static final int IMG_SIZE = 224;         // MobileNetV2 input size
    public static final int FEATURE_DIM = 1280;     // MobileNetV2 pooled feature size

    private static final String FEATURE_MODEL = "mobilenet_feature_extractor.tflite";
    private static final String LSTM_MODEL = "lstm_activity.tflite";
    private static final String LABELS_FILE = "labels.txt";

    /** Called after each video frame is processed (for a progress display). */
    public interface ProgressListener {
        void onProgress(int framesDone);
    }

    /** One prediction: best label plus the top-k list (confidences 0..1). */
    public static class Prediction {
        public final String label;
        public final float confidence;
        public final List<String> topLabels;
        public final List<Float> topScores;

        Prediction(List<String> topLabels, List<Float> topScores) {
            this.topLabels = topLabels;
            this.topScores = topScores;
            this.label = topLabels.get(0);
            this.confidence = topScores.get(0);
        }

        /** Same display style as the CNN exercise ("Dog X%, Cat Y%"). */
        public String display(int k) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Math.min(k, topLabels.size()); i++) {
                if (i > 0) sb.append(", ");
                sb.append(topLabels.get(i)).append(' ')
                  .append(String.format(Locale.US, "%.1f%%", topScores.get(i) * 100f));
            }
            return sb.toString();
        }
    }

    private final Interpreter featureInterpreter;
    private final Interpreter lstmInterpreter;
    private final List<String> labels;

    private final ByteBuffer inputBuffer;
    private final int[] pixels = new int[IMG_SIZE * IMG_SIZE];
    private final float[][] featureOutput = new float[1][FEATURE_DIM];

    /** Sliding window of the most recent feature vectors (live camera mode). */
    private final ArrayDeque<float[]> liveWindow = new ArrayDeque<>();

    public ActiveRecClassifier(Context context) throws IOException {
        Interpreter.Options featureOptions = new Interpreter.Options();
        featureOptions.setNumThreads(4);
        featureInterpreter = new Interpreter(loadModel(context, FEATURE_MODEL), featureOptions);

        Interpreter.Options lstmOptions = new Interpreter.Options();
        lstmOptions.setNumThreads(2);
        lstmInterpreter = new Interpreter(loadModel(context, LSTM_MODEL), lstmOptions);

        labels = loadLabels(context);

        int[] outShape = lstmInterpreter.getOutputTensor(0).shape();
        int outClasses = outShape[outShape.length - 1];
        if (outClasses != labels.size()) {
            close();
            throw new IllegalStateException("labels.txt has " + labels.size()
                    + " labels but the LSTM model outputs " + outClasses + " classes");
        }

        inputBuffer = ByteBuffer.allocateDirect(4 * IMG_SIZE * IMG_SIZE * 3);
        inputBuffer.order(ByteOrder.nativeOrder());
    }

    public List<String> getLabels() {
        return labels;
    }

    private static MappedByteBuffer loadModel(Context context, String name) throws IOException {
        try (AssetFileDescriptor fd = context.getAssets().openFd(name);
             FileInputStream stream = new FileInputStream(fd.getFileDescriptor())) {
            FileChannel channel = stream.getChannel();
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
        }
    }

    private static List<String> loadLabels(Context context) throws IOException {
        List<String> result = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(LABELS_FILE)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) result.add(line);
            }
        }
        return result;
    }

    // ---------------- Step 1: one frame -> MobileNetV2 -> 1280 features ----------------

    public synchronized float[] extractFeatures(Bitmap bitmap) {
        // Squash-resize to 224x224, same as cv2.resize in the notebook
        Bitmap scaled = (bitmap.getWidth() == IMG_SIZE && bitmap.getHeight() == IMG_SIZE)
                ? bitmap
                : Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true);
        scaled.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE);

        inputBuffer.rewind();
        for (int p : pixels) {
            // mobilenet_v2.preprocess_input: x / 127.5 - 1   (RGB order)
            inputBuffer.putFloat(((p >> 16) & 0xFF) / 127.5f - 1f);
            inputBuffer.putFloat(((p >> 8) & 0xFF) / 127.5f - 1f);
            inputBuffer.putFloat((p & 0xFF) / 127.5f - 1f);
        }
        inputBuffer.rewind();

        featureInterpreter.run(inputBuffer, featureOutput);
        return featureOutput[0].clone();
    }

    // ---------------- Step 2: 40 feature vectors -> LSTM -> prediction ----------------

    public synchronized Prediction classifySequence(List<float[]> features, int topK) {
        if (features.size() != SEQUENCE_LENGTH) {
            throw new IllegalArgumentException("Need " + SEQUENCE_LENGTH
                    + " feature vectors, got " + features.size());
        }
        float[][][] input = new float[1][SEQUENCE_LENGTH][];   // shape [1][40][1280]
        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            input[0][t] = features.get(t);
        }
        float[][] output = new float[1][labels.size()];        // shape [1][num_classes]
        lstmInterpreter.run(input, output);

        final float[] probs = output[0];
        Integer[] order = new Integer[probs.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(probs[b], probs[a]));

        List<String> topLabels = new ArrayList<>();
        List<Float> topScores = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, order.length); i++) {
            topLabels.add(labels.get(order[i]));
            topScores.add(probs[order[i]]);
        }
        return new Prediction(Collections.unmodifiableList(topLabels),
                              Collections.unmodifiableList(topScores));
    }

    // ---------------- Live camera mode ----------------

    public void resetLive() {
        synchronized (liveWindow) {
            liveWindow.clear();
        }
    }

    /** Adds one camera frame; returns how many frames are buffered (max SEQUENCE_LENGTH). */
    public int addLiveFrame(Bitmap bitmap) {
        float[] features = extractFeatures(bitmap);
        synchronized (liveWindow) {
            liveWindow.addLast(features);
            while (liveWindow.size() > SEQUENCE_LENGTH) liveWindow.removeFirst();
            return liveWindow.size();
        }
    }

    /** Predicts on the current live window, or returns null if fewer than 40 frames are buffered. */
    public Prediction predictLive() {
        List<float[]> snapshot;
        synchronized (liveWindow) {
            if (liveWindow.size() < SEQUENCE_LENGTH) return null;
            snapshot = new ArrayList<>(liveWindow);
        }
        return classifySequence(snapshot, 5);
    }

    // ---------------- Video file mode ----------------

    /**
     * Samples 40 frames evenly across the whole video (like the notebook's every-Kth-frame
     * sampling), runs each through MobileNetV2, then the LSTM.
     */
    public Prediction classifyVideo(Context context, Uri uri, ProgressListener listener) throws IOException {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(context, uri);
            String durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            if (durationStr == null) throw new IOException("Could not read video duration");
            long durationMs = Long.parseLong(durationStr);

            List<float[]> features = new ArrayList<>(SEQUENCE_LENGTH);
            float[] last = null;
            for (int i = 0; i < SEQUENCE_LENGTH; i++) {
                long timeUs = durationMs * 1000L * i / SEQUENCE_LENGTH;
                Bitmap frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST);
                float[] f;
                if (frame != null) {
                    f = extractFeatures(frame);
                } else if (last != null) {
                    f = last;   // reuse the previous frame if one can't be decoded
                } else {
                    throw new IOException("Could not decode frames from this video");
                }
                features.add(f);
                last = f;
                if (listener != null) listener.onProgress(i + 1);
            }
            return classifySequence(features, 5);
        } finally {
            try {
                retriever.release();
            } catch (Exception ignored) {
                // release() declares IOException on newer API levels
            }
        }
    }

    @Override
    public void close() {
        if (featureInterpreter != null) featureInterpreter.close();
        if (lstmInterpreter != null) lstmInterpreter.close();
    }
}
