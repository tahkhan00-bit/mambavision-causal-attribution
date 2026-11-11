# Size-Matched Nulls Reveal What Block Ablation Measures in MambaVision

Code for the paper **"Size-Matched Nulls Reveal What Block Ablation Measures in MambaVision"** (Tahmina Kousar and Tehseen Zia, COMSATS University Islamabad) and for the MS thesis **"A Causal Attribution Framework for Hybrid Mamba–Attention Vision Backbones"**.

- **Author:** Tahmina Kousar (Reg. SP25-RAI-020)
- **Supervisor:** Dr. Tehseen Zia
- **Experiments run on Kaggle by:** Toseef (Hugging Face account `Toseef11`, which hosts the checkpoint repository)

The study measures what the mixer blocks of MambaVision-T, -S and -B (Mamba and Attention mixers in Levels 2 and 3) contribute causally to ImageNet classification, with DeiT-Small and Swin-T as attention-only controls. It uses paired interventions on all 50,000 ImageNet-1k validation images:

- block, group and pairwise ablation,
- size-matched nulls (shuffled channels, sign flips; independent and shared permutations for groups and pairs),
- activation resampling,
- SSM vs gate pathway ablation,
- freezing of the selective-scan parameters Δ, B and C.

Every effect is reported with paired bootstrap confidence intervals, McNemar or Holm-corrected bootstrap tests, and equivalence tests. Content effects are also scaled by damage (content share R). The findings are replicated on MambaVision-S and -B, and the freezing method is validated with a positive control on a language Mamba.

---

## Repository contents

| File | Description |
|---|---|
| `mambavision_causal_attribution.ipynb` | The complete experiment notebook, Step 0 to Step 55. Run on Kaggle (one or two Tesla T4 GPUs). |
| `requirements.txt` | Package versions used. |
| `README.md` | This file. |

**Results are not stored in this repository.** Every step saves its output as a checkpoint (`<name>.pkl`) in the Hugging Face dataset repository below. All tables and figures of the report are generated from those checkpoints.

- **Checkpoints:** Hugging Face dataset `Toseef11/mambavision-Experiment-checkpoints` (private; access on request).

The notebook is saved without cell outputs; the results of every cell are in the checkpoints listed below. The Hugging Face token is read from a Kaggle Secret named `TOKEN` and is never stored in the notebook.

---

## Environment

| Item | Version |
|---|---|
| Platform | Kaggle notebook, 1 × NVIDIA Tesla T4 |
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| transformers | 4.51.3 |
| huggingface_hub | 0.36.2 |
| timm | 1.0.29 |
| numpy | 2.1.3 |
| mamba-ssm | 2.3.2.post1 (prebuilt wheel) |
| causal-conv1d | 1.7.0 (prebuilt wheel) |

`mamba-ssm` and `causal-conv1d` are installed in Step 2 from prebuilt wheels attached to the notebook as a Kaggle dataset (`mamba-deps-wheels-torch-2-11-0-cu128-cuda-12-8`), so nothing is compiled.

### Models and data (all from the Hugging Face Hub)

| Purpose | Identifier |
|---|---|
| Main model | `nvidia/MambaVision-T-1K` |
| Replication | `nvidia/MambaVision-S-1K` |
| External control and difficulty tiers | `facebook/deit-small-patch16-224` |
| Positive control | `state-spaces/mamba-130m-hf` |
| Dataset | `Tsomaros/Imagenet-1k_validation` (50,000 images) |

### Seeds

- Global seed **42**: image order, bootstrap resampling and the Step 16 permutations.
- Resampling bank: seed **777**.
- Step 27 permutations: `1000 + 100·block + j`.
- MambaVision-S nulls: `42 + block index`.

---

## How to run

1. **Create a Kaggle notebook** and import `mambavision_causal_attribution.ipynb`. Then:
   - set Accelerator to **GPU T4**,
   - turn **Internet on**,
   - attach the wheels dataset used in Step 2.
2. **Add a Kaggle Secret** named `TOKEN` containing a Hugging Face token with **write** access (Add-ons → Secrets). The token is read from the secret and is never printed.
3. **Change `HF_REPO_ID`** in the first code cell and in Step 0 to a repository you own, if you are not using the original one.
4. **Run the cells in order.** Every long step is resumable: it loads its checkpoint from Hugging Face and continues from the last completed block.

**Resuming in a new session.** Run the setup cells first:

1. Hugging Face cell
2. Steps 0–5b
3. The `batched_predict` helper and `mixer_blocks` cells
4. Step 8
5. Step 11b and the helper cell after it

Then run the step you need. Steps 25–30 also need Step 24B (the image cache).

The rebuilt pipeline reproduces the baseline exactly:
- 41,094 / 50,000 correct (82.188%),
- the fast cached path (Step 24) agrees with the original loader on 100% of predictions.

---

## Step map

Steps 18 and 23 are **result summaries only** (export cells, not experiments). The export cell at the end packs every checkpoint into one zip.

| Step | What it does | Checkpoint(s) | Report experiment |
|---|---|---|---|
| 0–5b | Kaggle setup, dependencies, MambaVision-T, preprocessing, dataset, seed-42 order | — | Setup |
| 6 | Full 50K baseline (predictions, logits, log-probabilities) | `baseline_logits_probs_new` | E1 |
| 7 | Architecture map of the 12 mixer blocks | — | Setup |
| 8 | Reference activations and per-channel means (256 images) | `step8` (saved in 11b) | Setup |
| 9 | Single-block mean ablation | `step9_block_ablation` | E2 |
| 10 | Paired CI, McNemar, Bonferroni for Step 9 | `step10` | E2 |
| 11 | 40 single-channel null trials (superseded, not used in the report) | `step11b_null_50k` | — |
| 11b | Session resume; summary | `step11b_summary` | — |
| 11c | Mann–Whitney: Mamba vs Attention | `partA_mannwhitney_50k` | E2 |
| 11d | Grouped ablation (all Mamba / all Attention off) | `partC_grouped_ablation` | E3 |
| 12A | Per-image predictions of the Step 9 ablations | `step9_ablated_preds` | E2, E4, E5 |
| 12B | Stratified ablation, internal tiers (superseded by 22F) | `step12_stratified` | — |
| 13 | Freezing of Δ, B, C: references, patch, check, 54-configuration grid | `step13_freeze_refs`, `step13_freezing_50k` | E8 |
| 14 | Joint freeze of all six Mamba blocks; selectivity fraction | `step14_freeze_all_mamba` | E9 |
| 15 | Activation resampling (bank seed 777) + Spearman vs mean ablation | `resampling_50k_v2`, `resampling_vs_meanablation_spearman` | E6 |
| 16 | Shuffled-channel null, real vs null, damage-vs-size | `shuffled_null_50k`, `damage_vs_perturbation_table`, `normalized_perturbation_sizes`, `perturbation_sizes` | E5 |
| 17 | SSM vs gate pathway ablation + paired contrast | `pathway_means_mamba_v2`, `pathway_ablation_50k_v3`, `tost_pathway_results` | E7 |
| 18 | Result summary (export only) | — | — |
| 19 | Statistical tests needing no GPU | `step19_report_tests` | Statistical summary |
| 20A | DeiT-Small control, first version (superseded by 22E) | `deit_control_50k` | — |
| 20B | Update norm, cosine, token concentration (2,000 images) | `update_norms_2k` | E11 |
| 20C | Receptive field, first version (superseded by 22D) | `figure2_attention_distance` | — |
| 21A | Additivity, per-block equivalence, sampling audit (48K disjoint images) | `step21a_review_checks` | E9, E13 |
| 22A | KL / logit no-op floor | `kl_noop_floor` | E1 |
| 22B | Logit drop and KL for the joint freeze | `step14_joint_freeze_logit_kl` | E9 |
| 22C | Logit drop and KL for block ablation | `step9_logit_kl` | E2 |
| 22D | Interaction range (implicit SSM attention vs attention, 1,000 images) | `figure2_attention_distance_v2` | E10 |
| 22E | DeiT-Small with reference preprocessing | `deit_control_50k_v2` | E1 |
| 22F | Stratified ablation with DeiT-defined difficulty tiers | `step12_stratified_external` | E4 |
| 22G | Positions of high-norm tokens | `high_norm_tokens` | E12 |
| 22H | L3B1 gate logit diagnostic | `l3b1_gate_logit_diagnostic` | E7 |
| 22I | All figures regenerated from one frozen data file | HF `figures/` | Figures |
| 23 | Result summary (export only) | HF `final_results/report_data.json` | — |
| 24 | Speed test, uint8 image cache, fast pass (100% identical to baseline) | `step24_speed` | Method |
| 25 | Joint freeze in mean vs position mode; per tensor (Δ, B, C) | `step25_joint_freeze_modes` | E9 |
| 26 | Group shuffled nulls (all Mamba, all Attention) | `step26_group_nulls` | E3, E5 |
| 27 | Shuffled null with 10 permutations per block; pooled bound (X = 0.10 pp) | `step27_multi_perm_null`, `step27_null_summary` | E5 |
| 28 | Positive control: copying in mamba-130m under Δ/B/C freezing | `step28_positive_control` | Validation |
| 29 | Replication on MambaVision-S (E2, E3, E5, E9) | `step29_mambavision_s` | Replication |
| 30 | High-norm token counts, token-0 causal test, log-prob drop for E2 | `step30_highnorm_heatmaps`, `step30_token_patch`, `step30_e2_logprob` | E2, E12 |
| 31 | Pathway ablation with the true-class log-probability drop | `step31_*` | Appendix C |
| 32 | Group shuffled nulls on MambaVision-S | `step32_s_group_nulls` | Sec. V |
| 33 | Frozen SSM as an explicit convolution kernel; truncation | `step33_*` | Sec. VI |
| 34 | DeiT-Small control (first run, superseded by Step 45) | `step34_*` | — |
| 35 | MambaVision-T at 384 × 384 | `step35_*` | Sec. VII |
| 36 | Pairwise block-interaction matrix (66 pairs) | `step36_pairwise` | Sec. V |
| 37 | Replication on MambaVision-B | `step37_mambavision_b` | Sec. IV–VI |
| 38 | Gain-matched truncation of the frozen-SSM kernel | `step38_*` | Sec. VI |
| 39 | ImageNet-C corruptions, one permutation (superseded by Step 48) | `step39_imagenet_c` | — |
| 40 | 384 px: shuffled null with several permutations | `step40_*` | Sec. VII |
| 41 | Pairwise interaction against independent shuffled nulls | `step41_pairwise_null` | Sec. V |
| 42 | MambaVision-B: extra permutations for key blocks (superseded by Step 51) | `step42_*` | — |
| 43 | Sign-flip null, rank metric, calibration (MambaVision-T) | `step43b_*` | Sec. IV, VI |
| 44 | Swin-T baseline and first control run | `step44_swin_*` | Sec. IV |
| 45 | DeiT-Small: ten permutations per layer | `step45_deit_*` | Sec. IV, App. A |
| 46 | Swin-T: ten permutations per block | `step46_swin_*` | Sec. IV, App. A |
| 47 | MambaVision-T resample null with per-image results | `step47_*` | Sec. IV |
| 48 | ImageNet-C corruptions, ten permutations | `step48_*` | Sec. VII, App. G |
| 49 | Freezing grid with log-probability and rank (54 configurations) | `step49_grid_*` | Sec. VI, App. E |
| 50 | MambaVision-S: ten permutations per block | `step50_s_*` | Sec. IV, App. A |
| 51 | MambaVision-B: ten permutations per block | `step51_b_*` | Sec. IV, App. A |
| 52 | Content share R relative to damage, all five models (CPU) | `step52_relative_content` | Sec. IV, Table I |
| 53 | MambaVision-T group nulls: independent vs shared permutations | `step53_group_*` | Sec. V, Table III |
| 54 | Pair nulls with a shared permutation; noise floor of R | `step54_pair_*` | Sec. V, Fig. 3 |
| 55 | MambaVision-S and -B group nulls: independent vs shared permutations | `step55_*` | Sec. V, Table III |
| Export | All checkpoints → one JSON + index, zipped | — | — |

---

## Main results (MambaVision-T, 50,000 images unless stated)

| Result | Value |
|---|---|
| Baseline top-1 (T / S / DeiT-Small) | 82.188% / 83.250% / 79.85% |
| Single-block ablation | −0.08 to −1.13 pp; 9/12 blocks significant after Bonferroni |
| Real vs size-matched null damage across blocks | Spearman ρ = 0.83 (10 permutations) |
| Content effect, pooled over 12 blocks | −0.016 pp [−0.049, +0.024]; one-sided 95% upper bound 0.016 pp (pre-set bound 0.10 pp) |
| Only block with a per-block content effect | L3B3: +0.190 pp [+0.097, +0.285], Holm p = 0.001 (also significant on MambaVision-S) |
| All Mamba off: real vs independent / shared group null (10 perms) | −3.90 vs −1.27 / −1.80 pp |
| All Attention off: real vs independent / shared group null (10 perms) | −20.58 vs −4.22 / −7.81 pp |
| Content share R (accuracy): MV-T / Swin-T / DeiT-Small | 0.20 / 0.24 / 0.91 |
| Joint freeze of Δ, B, C, mean mode | −1.136 pp; 29.1% [24.8, 33.5] of all-Mamba-off |
| Joint freeze, position mode (image-dependence only) | −0.790 pp; 20.3% [16.4, 24.3] of all-Mamba-off |
| Positive control (mamba-130m, copying) | Freezing B or C removes ~93–96% of the copy benefit; first-copy loss unchanged |
| MambaVision-S, joint freeze in position mode | −1.008 pp; 32.1% [27.2, 36.9] of all-Mamba-off |
| Scan-start token (largest update in 100% of images) | Patching it alone: −0.004 pp [−0.020, +0.012] |

---

## Statistics

- **Effects:** all paired against the unmodified model on the same images.
- **Confidence intervals:** paired bootstrap, 2,000 resamples, seed 42.
- **Tests:** McNemar χ² with continuity correction, exact when b + c < 25.
- **Multiple comparisons:**
  - Bonferroni within each family: 0.05/12 per-block, 0.05/6 SSM − gate, 0.05/54 freezing grid.
  - Holm for the 10-permutation per-block contrasts.
- **±0.15 pp band:** descriptive only.
