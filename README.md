# ActiveRec (Java): Video Activity Recognition with MobileNetV2 + LSTM

Android app for CS663 Exercise LSTM. Each video frame goes through **MobileNetV2** (TFLite) to get a 1x1280 feature vector, 40 of those are packaged into one `[1, 40, 1280]` tensor, and the **LSTM** (TFLite) predicts the activity.

## 1. Add the model files (required)

Copy these 3 files from Google Drive `MyDrive/CS663/LSTMVideoActivity/tflite/` (or from the notebook's STEP 16 download) into:

```
app/src/main/assets/
    mobilenet_feature_extractor.tflite
    lstm_activity.tflite
    labels.txt
```

Always copy `lstm_activity.tflite` and `labels.txt` **from the same training run** (e.g. both from the 14-class run, or both from the 101-class run).

## 2. Open and run in Android Studio (Mac)

1. Android Studio > **Open** > select this `ActiveRec` folder (the one containing `settings.gradle`).
2. Wait for **Gradle sync** to finish (first time downloads Gradle 8.7 and the libraries; a few minutes).
   - If Studio offers to upgrade the Android Gradle Plugin, you can click "Don't ask again" or accept; either works.
   - If it complains about the JDK: *Settings > Build, Execution, Deployment > Build Tools > Gradle > Gradle JDK* = the bundled JDK (jbr-17 or jbr-21).
3. Pick a device:
   - **Emulator:** *Device Manager > +* > Pixel 7 (or similar), API 34. Fine for **Pick Video**.
   - **Real phone:** enable *Developer options > USB debugging*, plug in. Needed for a meaningful **Start Live** demo.
4. Click **Run ▶**.

## 3. Use it

1. In the spinner, choose **ActiveRec**.
2. **Pick Video**: choose a clip. 40 frames are sampled across it, then MobileNetV2, then the LSTM. Result shows like `Archery 97.2%, BandMarching 1.1%, ...`
   - To get a test clip onto the emulator, drag a UCF-101 `.avi` from Finder onto the emulator window (it goes to *Downloads*). Use a class the model was trained on.
3. **Start Live** (real phone): point the rear camera at someone doing an activity. After 40 frames (~6.4 s) it predicts, then updates every 10 frames.

## Project layout

```
app/src/main/java/com/cs663/activerec/
    MainActivity.java          spinner with the "ActiveRec" option
    ActiveRecActivity.java     ActiveRec screen: Pick Video + Start Live (CameraX)
    ActiveRecClassifier.java   the pipeline: MobileNetV2 per frame -> 40 x 1280 -> LSTM
app/src/main/res/layout/       activity_main.xml, activity_active_rec.xml
app/src/main/assets/           the 3 model files go here
```

## Adding ActiveRec to the group's existing (Java) app

1. Copy `ActiveRecClassifier.java` and `ActiveRecActivity.java` into the app's package folder and change the first line `package com.cs663.activerec;` to the app's package.
2. Copy `activity_active_rec.xml` into `res/layout/`, and the ActiveRec `<string>` entries from `strings.xml` into the app's `strings.xml`.
3. Copy the 3 model files into the app's `assets/` folder.
4. In `app/build.gradle` add the CameraX lines, the `listenablefuture` line and the `noCompress 'tflite'` block (keep the app's existing TFLite dependency if it has one).
5. In `AndroidManifest.xml` add the CAMERA permission and `<activity android:name=".ActiveRecActivity" android:exported="false" />`.
6. Add `"ActiveRec"` to the spinner options and in the selection handler:
   ```java
   if (selected.equals("ActiveRec")) {
       startActivity(new Intent(this, ActiveRecActivity.class));
   }
   ```

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not load models ... assets" | The 3 files must be in `app/src/main/assets/` with exactly these names |
| "labels.txt has N labels but the LSTM model outputs M classes" | `labels.txt` and `lstm_activity.tflite` are from different runs; copy both from the same run |
| Live mode always predicts the same class | Fill the frame with the person, keep the phone still, do the action for the whole ~6 s. Phone video differs from UCF-101's YouTube clips, so Pick Video gives better accuracy |
| App crashes on an Android 15 device with 16 KB memory pages | Rare on current phones; switch the TFLite dependency to `com.google.ai.edge.litert:litert:1.0.1` (same `org.tensorflow.lite.Interpreter` API) |
| Build error you can't solve | Copy the error from the Build window and paste it to Claude |
