# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.1] - 2026-11-24

### Fixed

- **All model scores were wrong in earlier versions.** `core/xray_engine.py`
  applied `torch.sigmoid` to the output of TorchXRayVision's pretrained
  DenseNet, whose `forward` already returns `op_norm(sigmoid(logits))`, a score
  in [0, 1]. The second sigmoid mapped every score into 50.0-73.1 %: a score the
  model put near 0 was shown as about 50 %, and nothing could exceed 73.1 %.
  Because of this every pathology was at least MODERATE, every image was at
  least "NEEDS ATTENTION", and any shown percentage from v0.1.0 or earlier
  should not be relied on. On TorchXRayVision's public NIH sample image
  `00000001_000.png`, v0.1.0 reported the 18 scores as 50.3-65.9 %; v0.1.1
  reports 1.2-66.0 % (Fibrosis 63.0 % -> 53.3 %).
- The README's example response ("Effusion 82.0 HIGH", status "CRITICAL") could
  not have been produced by the old code. It is replaced by real output.

### Changed

- Score bands follow the model's own normalisation, in which 0.5 is each
  pathology's operating-point threshold: LOW < 50 %, MODERATE 50-70 %,
  HIGH > 70 % (previously 40 % / 70 %).
- Status labels describe the scores, not the patient: `NO ELEVATED SCORE`,
  `ELEVATED SCORE`, `HIGH SCORE` (previously `NORMAL`, `NEEDS ATTENTION`,
  `CRITICAL`).
- Removed medical directives. The CLI's "patient friendly" report with advice
  such as "Seek medical attention immediately" and "Consult a cardiologist" is
  replaced by `generate_plain_report`, which lists scores and what each label
  name refers to. The fixed explanation, the LLM prompt and the dashboard no
  longer recommend actions; the dashboard's "Recommended action" panel is now a
  "Note".
- Removed the dashboard's derived "Confidence" and "Abnormality" figures (they
  were rescalings of the top score, not measurements).
- Removed two README screenshots taken with an older interface that showed
  scores inflated by this bug; the dashboard screenshot is retaken.

### Added

- `tests/test_engine.py`: the inference wrapper against a fake model that
  follows TorchXRayVision's forward contract (a near-zero output must be
  reported near zero), against TorchXRayVision's own `DenseNet` class, and
  against the real pretrained weights on the public sample image.
- An `engine` CI job that installs CPU PyTorch and TorchXRayVision and runs
  those tests.

## [0.1.0] - 2026-11-18

First tagged release of the research prototype. Upload handling, score
banding, optional Ollama explanation, dashboard and CLI; tests with the model
replaced by a fake.

[Unreleased]: https://github.com/projectbin07-stack/xray-analysis-aiml/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/projectbin07-stack/xray-analysis-aiml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/projectbin07-stack/xray-analysis-aiml/releases/tag/v0.1.0
