# Absolute Localization

Run the commands below from `DiffSel_github/code_abs` in a Python environment with the dependencies listed in `requirements.txt` installed.

Training:

```bash
python -B Diffusion_Anchor_Selection/train_anchor_selection_diffusion.py --train_dataset Diffusion_Anchor_Selection/data/selection_square-N20-M5-Na4-7.npz
```

Evaluation:

```bash
python -B Diffusion_Anchor_Selection/test_model.py --ckpt_path Diffusion_Anchor_Selection/trained/Abs_Last.ckpt
```

Training outputs are saved to `Diffusion_Anchor_Selection/outputs/`.
