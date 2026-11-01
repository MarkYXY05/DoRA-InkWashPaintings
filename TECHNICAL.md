# Under the hood

[Back to project](README.md)

## Code tour

These are the fastest entry points from a visual result to the implementation that produced it. Search for the named symbols inside each linked notebook.

| Engineering question | Symbols / implementation | Where to look |
| --- | --- | --- |
| How is DoRA built into SDXL? | `LoRALinear`, `DoRALinear`, `inject_and_unfreeze` | [Packaged SDXL notebook](notebooks/archive/inkwash-small-packaged.ipynb) |
| How do training images become a style adapter? | `DreamBoothDataset`, `run_single_experiment`, `compute_param_stats_for_method` | [Diffusion workflow](notebooks/archive/inkwash-small-packaged.ipynb) |
| How are LLM target modules replaced and checked? | `replace_target_linears_with_dora`, `freeze_all_then_unfreeze_dora`, `run_minimal_dora_smoketest` | [Commonsense notebook](notebooks/language/commonsense.ipynb) |
| How are candidate answers scored? | `candidate_nll`, `evaluate_all_tasks`; alternative `generate_predictions` path | [Evaluation implementation](notebooks/language/commonsense.ipynb) |
| What makes this a manual DPO implementation? | `sequence_logps`, `add_reference_logps`, `dpo_loss`, `train_dpo_policy` | [DPO notebook](notebooks/alignment/manual-dpo.ipynb) |
| How are learned updates analyzed? | Checkpoint loading and per-layer magnitude/direction measurements | [Analysis module](src/dora_weight_decomposition_analysis.py) |

The compact repo export provides the language, small-set diffusion, DPO, vision-language, and analysis implementations. This showcase also retains the larger visual experiment collection from the original project folder.

## 1. Custom adapters, not just an API switch

The core `DoRALinear` layer starts from a frozen pretrained weight matrix. It learns a low-rank update to the weight direction and a separate magnitude vector. The forward pass combines those components; a merge path folds the update into the weight for inference.

In the language-model experiments, adapter injection targets attention and feed-forward projections including `q_proj`, `k_proj`, `v_proj`, `up_proj`, and `down_proj`. In SDXL, it targets U-Net attention projections `to_q`, `to_k`, `to_v`, and `to_out.0`.

**Read:** [Language adapters](notebooks/language/commonsense.ipynb) · [Diffusion adapters and checks](notebooks/diffusion/inkwash-large.ipynb).

## 2. A visual adaptation pipeline

The SDXL pipeline connects image loading and transforms, VAE latent encoding, sampled noise/timesteps, U-Net denoising loss, adapter-only optimization, and prompt-based image generation. The VAE and text encoders remain frozen.

The ink-wash experiments use rank-8 adapters. The large-set run records 12,277,760 trainable DoRA parameters across 560 attention projections, versus an adapted U-Net size of 2,579,741,444 parameters. That is **0.4759% trainable**.

The comparison called `full` in the original files tunes the selected attention layers. Here it is labeled **target-layer FT** so the comparison stays clear. Its 955,353,600 trainable parameters make the DoRA setup approximately **77.8× smaller in trainable-parameter count**; this is a parameter comparison, not a measured runtime or GPU-memory speedup.

**Read:** [Parameter counts](results/inkwash-large/metrics/trainable_params_summary.json) · [Large-set notebook](notebooks/diffusion/inkwash-large.ipynb) · [Small-set notebook](notebooks/diffusion/inkwash-small.ipynb).

## 3. Language evaluation with two views

The language track covers LLaMA-7B, LLaMA-13B, LLaMA-2-7B, and LLaMA-3-8B on BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA.

It includes both conditional-NLL candidate scoring and generation/parsing-based evaluation. The training implementation uses gradient accumulation, mixed precision, gradient checkpointing, AdamW, and warmup scheduling. Predictions and compact benchmark tables are exported separately.

The headline LLaMA-3-8B result uses the `own_methods` table: 81.7% average accuracy for DoRA-r32 and 79.8% for LoRA-r16. The formal configuration uses 10,000 training examples and up to 200 evaluation examples per task.

**Read:** [Commonsense notebook](notebooks/language/commonsense.ipynb) · [Tables](results/commonsense).

## 4. DPO from the training objective upward

The alignment track implements Direct Preference Optimization manually rather than relying on a trainer wrapper. It handles chosen/rejected response pairs, masks prompts when scoring responses, caches reference-model log probabilities, and evaluates implicit reward accuracy and margin.

Runs compare LoRA and DoRA at ranks 16 and 32 on Anthropic HH-RLHF. DoRA-r32 records 54.8% reward accuracy and a 0.0883 reward margin; LoRA-r32 records 55.9% and 0.0852. This track demonstrates objective implementation and evaluation design, rather than a universal winner between adapters.

**Read:** [Manual DPO](notebooks/alignment/manual-dpo.ipynb) · [Results](results/alignment/rlhf_dpo_results.csv).

## 5. Understanding the update

The analysis tooling measures magnitude and direction changes at intermediate checkpoints and visualizes trajectories for query/value projections. The exported query and value tables each contain 256 measurement rows.

The secondary vision-language track implements a T5-based architecture using 2,048-dimensional object features, bounding boxes, and text embeddings, with TSV-to-HDF5 conversion and compact validation features.

**Read:** [Analysis module](src/dora_weight_decomposition_analysis.py) · [Analysis notebook](notebooks/analysis/weight-decomposition.ipynb) · [VQA implementation](notebooks/vision-language/vqa.ipynb).
