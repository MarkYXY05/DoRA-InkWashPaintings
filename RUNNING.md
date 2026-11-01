# Running the experiments

[Back to project](README.md)

This is a notebook-first showcase of the existing experiments. The notebooks retain their original training logic; saved execution output and runtime metadata have been cleared for readability, with image outputs preserved separately in the gallery.

## Environment

- Use a CUDA-capable GPU or a GPU-backed Colab runtime for training.
- Follow the installation cells in the notebook you choose. Dependencies differ between the language, diffusion, and vision-language tracks; use a separate environment for each track.
- Main libraries: PyTorch, Transformers, PEFT, Diffusers, Accelerate, Datasets, torchvision, NumPy, pandas, Matplotlib, and h5py.
- Authenticate with Hugging Face through its login flow for any gated model you use. Do not embed tokens in notebook cells.

## Choose a track

| Track | Entry point | Configuration to update |
| --- | --- | --- |
| SDXL ink-wash adaptation | [Large-set notebook](notebooks/diffusion/inkwash-large.ipynb) | `MODEL_ID`, `DATA_DIR`, output paths, steps and rank |
| Smaller visual experiment | [Small-set notebook](notebooks/diffusion/inkwash-small.ipynb) | Training image directory, output paths |
| LLM reasoning | [Commonsense notebook](notebooks/language/commonsense.ipynb) | `BASE_MODEL`, dataset root, sample caps, rank and learning rate |
| Preference alignment | [Manual DPO notebook](notebooks/alignment/manual-dpo.ipynb) | Model access, dataset/sample configuration, output paths |
| Weight analysis | [Weight-decomposition notebook](notebooks/analysis/weight-decomposition.ipynb) | `DORA_ROOT`, checkpoint paths, analysis-module location |
| Vision-language implementation | [VQA notebook](notebooks/vision-language/vqa.ipynb) | Feature HDF5, question/answer data paths |

The notebooks originated in Colab and include project-specific Google Drive paths. Replace those paths with your own storage locations. Model checkpoints and raw training datasets are not bundled.

For the language track, select **one** evaluation implementation (NLL scoring or generation/parsing) before running the experiment loop; the notebook contains both. Use the smoke-test configuration first, then disable its overrides for the larger run.

For the weight-analysis notebook, place `src/dora_weight_decomposition_analysis.py` on the Python import path or update the notebook's import/location configuration.

## View results without training

- [Image gallery](GALLERY.md)
- [Language benchmark tables](results/commonsense)
- [SDXL parameter counts](results/inkwash-large/metrics/trainable_params_summary.csv)
- [DPO results](results/alignment/rlhf_dpo_results.csv)
- [Project report](reports/project-report.pdf)

This packaging pass did not rerun GPU training. The included results are the project's saved outputs.
