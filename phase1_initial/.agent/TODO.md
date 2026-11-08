# TODO

- [x] Exclude MS-DAYOLO by user decision.
- [x] Remove the retired multi-model full campaign and production auxiliary code.
- [x] Bound retained training paths to small-sample technical runs.
- [x] Run SSDA-YOLO, SF-YOLO and Adaptive Teacher R101-C4 with 1,000 source/target images and the same 1,000-image lighting validation sample.
- [x] Reject 120-update YOLO attempts with only 503 batches and rerun both at 1,015 batches/151 updates for full-pool coverage.
- [x] Fix R101 detector-to-teacher/student checkpoint initialization.
- [x] Select Adaptive Teacher Faster R-CNN R101-C4 and record contrary pilot evidence.
- [x] Keep the held-out test split unopened.
- [ ] When requested, design the R101-only source/adaptation/fine-tuning schedule for the proposal budget ladder.
