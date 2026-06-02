# Tuesday Meta Update

The project direction is now: AR/VR-native personal memory should first check self memory, then decide whether to access self raw evidence, self history, audio, another user's view/audio, static room cameras, or auxiliary modalities.

Progress made:

- EgoVis2026 questions were converted into a source-access taxonomy.
- CASTLE inventory shows routing signals that can be inspected before video decoding: transcripts, gaze, pose/IMU/GPS, heart rate, metadata, and source ownership from paths.
- Seed cases show why captions alone are often insufficient: exact counts, small visual details, labels/OCR, exact speech, off-FOV spatial facts, and cross-user co-presence.

Next technical step:

Build a small evidence-router prototype that takes a question plus lightweight AR/VR metadata and outputs a progressive retrieval plan with estimated communication/privacy cost.

What not to claim yet:

- No model accuracy.
- No benchmark completion.
- No trained model.
- No large-scale video processing.
