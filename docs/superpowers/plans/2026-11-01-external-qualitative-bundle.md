# External Qualitative Bundle Implementation Plan

> **For agentic workers:** Execute the packaging steps inline in the current repository. The user requires local development without worktrees.

**Goal:** Download official EV-SpSegNet, PACT and PointEvent source and available EV-UAV checkpoints into one cloud upload bundle.

**Architecture:** Keep official source snapshots separate from small launch helpers. Record upstream revisions, checkpoint origins, hashes, original inference conventions and output contracts so returned event masks can be aligned for qualitative figures.

**Tech Stack:** Python standard library, GitHub official repositories, Linux shell and the environments documented upstream.

---

### Task 1: Freeze official inputs

**Files:** `remote_upload/external_qualitative_20261101/models/`, `manifest.json`.

- [x] Read each upstream README and inference entry, and resolve a commit SHA.
- [x] Download source archives at those revisions and official EV-UAV weights; identify missing resources explicitly.
- [x] Verify actual binary files rather than LFS pointers or download HTML, and record SHA256 hashes and sizes.

### Task 2: Prepare the cloud handoff

**Files:** `remote_upload/external_qualitative_20261101/README.md`, `tools/`, `SHA256SUMS`.

- [x] Document separate environments, real data layouts, checkpoint paths, native prediction commands and export limitations.
- [x] Add only necessary helpers for input selection and row-aligned prediction export; retain official model decisions and preprocessing.
- [x] Preserve raw event order and original coordinates/time; require matching source files across all methods and ground truth.

### Task 3: Verify delivery

**Files:** `remote_upload/external_qualitative_20261101.zip`, `.zip.sha256`.

- [x] Run helper syntax and CLI checks and focused checks for any row-alignment logic.
- [x] Check checkpoint state-dictionary metadata on CPU where supported; report GPU inference as unverified locally.
- [x] Create and reopen the upload ZIP, verify file hashes and deliver the absolute folder/archive paths.

## Delivery verification

All three source snapshots and EV-UAV model files are included. EV-SpSegNet was supplied by the user from D:/best_iou_seed37.pt and passed CPU weights-only loading. PACT official release digest matches. PointEvent original is retained, with 278 tensor entries preserved exactly in its inference-only copy. Eight focused checks, Ruff, syntax, three local validation input audits and archive/hash verification passed. CUDA inference remains intentionally unrun locally.

## Native Windows correction

The user clarified that the RTX 4090 host runs native Windows and cannot use WSL2. The original Linux-only handoff is not executable there.

- [x] Add a native PyTorch inference backend: first-occurrence voxel indexing / mean for the sparse networks, and PointEvent segmented reductions / selective state scan. Keep official network definitions and strict weight loading.
- [x] Compare portable operators with independent recurrence and voxel aggregation oracles, including long-state carry, and check official checkpoint structure on native Windows where dependencies allow.
- [x] Add PowerShell preflight, prediction and collection instructions; keep the original Linux route explicitly separate.
- [x] Refresh source snapshot, manifest, verification and ZIP hashes. State that native Windows GPU inference remains unverified locally.

Native Windows correction delivered with 15 focused checks and all three official checkpoints loaded strictly on CPU. PointEvent real-weight SSM 517-token forward agrees with independent serial recurrence within 2.98e-8 absolute error. RTX 4090 forward and parity with original fused kernels remain unverified locally.
