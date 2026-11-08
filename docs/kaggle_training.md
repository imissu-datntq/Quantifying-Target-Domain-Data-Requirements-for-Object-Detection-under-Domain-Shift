# Training on Kaggle

GPU work runs on the Kaggle account `kizhowl`. No dataset is re-split there: the
frozen protocol is shipped as the **private** dataset
`kizhowl/bdd100k-revised-protocol` and the images come from the public mirror
`solesensei/solesensei_bdd100k`, which is the archive the local copy was verified against.

## Private protocol dataset

| Path | Content |
| --- | --- |
| `code/` | `scripts/`, `scripts/bdd_pipeline/`, `tests/`, `kaggle/common.py` |
| `data/` | frozen annotations (train/test/val), manifests, `dataset_metadata.json`, `COMPLETE` |
| `expected_sha256.json` | SHA256 of every annotation file, the three manifests and the metadata |

Staging copies these files from the repository and `data/bdd100k_revised/`;
`kaggle datasets version -p <stage> --dir-mode zip` publishes a new version after
code changes.

## Rebuilding the dataset in a kernel

`kaggle/common.py:bootstrap()` copies the code to `/tmp/code` and runs
`scripts/kaggle_materialize.py`, which:

1. copies annotations, manifests and metadata and checks each against `expected_sha256.json`;
2. symlinks every manifest image from the mirror's `images/100k/{train,val}`;
3. rehashes every linked image against its manifest SHA256;
4. runs the same structural validator as the local dataset (55,000/14,863/10,000,
   zero cross-split basename/content overlap).

The pipeline then reads `/tmp/bdd100k_revised` exactly like the local root, so
manifest hashes recorded in checkpoints match local runs.

## Kernels

| Kernel | Accelerator | Purpose |
| --- | --- | --- |
| `kaggle/bdd_bootstrap_check.py` | CPU | Rebuild + unit tests + dry run, without spending GPU quota |
| `kaggle/bdd_gpu_pilot.py` | T4 | AMP/batch throughput and memory |
| `kaggle/bdd_source_train.py` | T4 | Source training resumed across sessions from the previous version's output |
| `kaggle/bdd_gap_gate.py` | T4 | Official zero-shot gap gate with the completed source checkpoint |

Kernel folders (script plus `kernel-metadata.json`) are staged outside the
repository and pushed with `kaggle kernels push -p <folder>`. All kernels are
private and attach both datasets. Weekly GPU quota is 30 hours and is shared with
the account's other projects; check it with `kaggle quota` before pushing.

## Campaign kernel (2026-10-30)

`kaggle/bdd_campaign.py` takes a `QUEUE` of run ids from `bdd_pipeline.campaign.grid()`, which is set in the pushed copy. It attaches three inputs:
- the protocol dataset;
- the `bdd100k-source-train` output (`runs/source_seed0/best.pt`);
- its own previous version's output, which carries `campaign/` forward.

Download each version's `campaign/*/predictions/` locally. Scoring happens locally, after the grid is complete.

**Kaggle accounts.**
- The CLI defaults to `buihuynhgiahuy` through `~/.kaggle/access_token.txt`.
- To act as `kizhowl`, set `KAGGLE_API_TOKEN` to a file containing its key.
- Account `ngdminh31` is not part of this project.
