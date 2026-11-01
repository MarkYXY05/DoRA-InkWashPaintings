# Project map

[Back to project](README.md) · [Visual gallery](GALLERY.md)

## Main implementation tracks

| Start here | What it demonstrates |
| --- | --- |
| [SDXL ink-wash, large set](notebooks/diffusion/inkwash-large.ipynb) | Custom adapters, image-to-latent training, GPU optimization, visual comparisons |
| [SDXL ink-wash, small set](notebooks/diffusion/inkwash-small.ipynb) | Smaller-data adaptation and alternative saved outputs |
| [Commonsense reasoning](notebooks/language/commonsense.ipynb) | Custom LLM layers, training orchestration, eight-task evaluation |
| [Manual DPO](notebooks/alignment/manual-dpo.ipynb) | Preference objective, cached reference scores, chosen/rejected masking, reward evaluation |
| [Weight decomposition](notebooks/analysis/weight-decomposition.ipynb) | Checkpoint trajectories and magnitude/direction analysis |
| [VQA implementation](notebooks/vision-language/vqa.ipynb) | T5-based vision-language architecture and feature-processing pipeline |

## Additional visual domains

- [Line-sketch adaptation](notebooks/diffusion/line-sketch.ipynb)
- [Cake-sketch adaptation](notebooks/diffusion/cake-sketch.ipynb)
- [Character adaptation](notebooks/diffusion/character.ipynb)

## Utilities

- [Weight-decomposition analysis](src/dora_weight_decomposition_analysis.py)
- [TSV → HDF5 features](src/tsv_to_h5.py)
- [Compact COCO validation features](src/coco_val_compact.py)

## Preserved notebook versions

These retain development context without competing with the main entry points:

- [Language notebook from the newer folder export](notebooks/language/commonsense-latest-export.ipynb)
- [Packaged small-set diffusion notebook](notebooks/archive/inkwash-small-packaged.ipynb)
- [Packaged VQA notebook](notebooks/archive/vqa-packaged.ipynb)
- [Early diffusion prototype](notebooks/archive/diffusion-prototype.ipynb)
- [Early ink-wash prototype](notebooks/archive/inkwash-prototype.ipynb)
- [Initial implementation notebook](notebooks/archive/initial-implementation.ipynb)
- [Early language notebook](notebooks/archive/early-language.ipynb)

## Visuals and results

- **72 unique generated images**, organized by domain and source method: [gallery](GALLERY.md).
- **31 unique saved notebook PNGs**, including outputs that were embedded in notebooks rather than separately exported: [index](assets/notebook-figures/index.json).
- **30 unique standalone experiment plots:** [index](assets/plots/index.json).
- [Commonsense results](results/commonsense), [diffusion metrics](results/inkwash-large), [alignment results](results/alignment), and [weight trajectories](results/weight-decomposition).
- [Original poster](reports/poster.pdf) and [public copy of the project report](reports/project-report.pdf).

The counts describe file categories; an exported plot and a notebook figure can depict the same experiment. The saved source grids retain their original labels; `Target Full FT` refers to the target-layer tuning comparator.

## Asset handling

Generated image files and copied plots are unchanged. Notebook PNG outputs were decoded losslessly. Notebook code and markdown are preserved while execution output, counts, and runtime metadata are cleared. Source mappings are available in [provenance.json](provenance.json).

The report's public copy replaces two internal repository URLs with a pointer to this README. The original report is retained in the source project folder.

Raw datasets, model checkpoints, large feature files, pickles, vendored libraries, and Git history are intentionally outside this lightweight showcase.
