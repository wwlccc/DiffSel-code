# Relative Localization

Run the commands below from `DiffSel_github/code_rel` in a Python environment with the dependencies listed in `requirements.txt` installed.

Training:

```bash
python -B Diffusion_Node_Activation/train_node_activation_diffusion.py --train_dataset Diffusion_Node_Activation/data/activation_square-N20-Na4-7-4k.npz
```

Evaluation:

```bash
python -B Diffusion_Node_Activation/test_model.py --ckpt_path Diffusion_Node_Activation/pretrained/Rel_Last.ckpt
```

Training outputs are saved to `Diffusion_Node_Activation/outputs/`.
