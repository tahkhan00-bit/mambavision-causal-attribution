# Causal Attribution in Hybrid Mamba–Attention Vision Backbones (MambaVision)

Code for the MS thesis **"A Causal Attribution Framework for Hybrid Mamba–Attention Vision Backbones"**.

- **Author:** Tahmina Kousar (Reg. SP25-RAI-020)
- **Supervisor:** Dr. Tehseen Zia
- **Experiments run on Kaggle by:** Toseef (Hugging Face account `Toseef11`, which hosts the checkpoint repository)

The study asks what the mixer blocks of a hybrid Mamba–Attention vision model contribute causally to ImageNet classification, and what block-level causal interventions can and cannot tell us. All effects are measured with paired interventions on the 50,000 ImageNet-1k validation images (subsets are stated where used):

- single-block, grouped and pairwise mean ablation,
- size-matched shuffled-channel nulls (10 permutations per block), group nulls and pair nulls,
- activation resampling,
- SSM vs gate pathway ablation,
- freezing of the selective-scan parameters Δ, B and C (mean, position, global and shuffle modes),
- the frozen scan rewritten as an explicit causal convolution, with truncation,
- replication on MambaVision-S and MambaVision-B, the same protocol on DeiT-Small,
- a positive control on a language Mamba (in-context copying),
- evaluation at 384 px and under four ImageNet-C corruptions.

Every effect is reported with paired bootstrap confidence intervals and McNemar tests, with Holm or Bonferroni correction.

---

## Repository contents

| File | Description |
|---|---|
| `mambavision_causal_attribution.ipynb` | The complete experiment notebook, Step 0 to Step 42 plus the export cell. Run on Kaggle (Tesla T4). |
| `requirements.txt` | Package versions used. |
| `README.md` | This file. |

**Results are not stored in this repository.** Every step saves its output as a checkpoint (`<name>.pkl`) in the Hugging Face dataset repository below. All tables and figures of the results report are generated from those checkpoints.

- **Checkpoints:** Hugging Face dataset `Toseef11/mambavision-Experiment-checkpoints` (private; access on request).

Many cells show no output in the notebook because they were run in earlier Kaggle sessions. Their results are in the checkpoints listed below.

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

| Purpose | Identifier | Top-1 on the probe |
|---|---|---|
| Main model | `nvidia/MambaVision-T-1K` | 82.188% |
| Replication | `nvidia/MambaVision-S-1K` | 83.250% |
| Replication | `nvidia/MambaVision-B-1K` | 84.126% |
| External control and difficulty tiers | `facebook/deit-small-patch16-224` | 79.848% |
| Positive control (copying) | `state-spaces/mamba-130m-hf` | — |
| Dataset | `Tsomaros/Imagenet-1k_validation` (50,000 images) | — |

### Seeds

- Global seed **42**: image order, bootstrap resampling, null permutation 0.
- Resampling bank: seed **777**.
- Further null permutations: `1000 + 100·block + j`.
- MambaVision-S and -B nulls: `42 + block index`.
- ImageNet-C Gaussian noise: seed **39**.

---

## How to run

1. **Create a Kaggle notebook** and import `mambavision_causal_attribution.ipynb`. Then:
   - set Accelerator to **GPU T4**,
   - turn **Internet on**,
   - attach the wheels dataset used in Step 2.
2. **Add a Kaggle Secret** named `TOKEN` containing a Hugging Face token with **write** access (Add-ons → Secrets). The token is read from the secret and is never printed.
3. **Change `HF_REPO_ID`** in the first code cell and in Step 0 to a repository you own, if you are not using the original one.
4. **Run the cells in order.** Every long step is resumable: it loads its checkpoint from Hugging Face and continues from the last completed run.

**Resuming in a new session.** Run the setup cells first:

1. Hugging Face cell
2. Steps 0–5b
3. The `batched_predict` helper and `mixer_blocks` cells
4. Step 8
5. Step 11b and the helper cell after it
6. Step 24B (image cache) and the Step 30 setup cell

Then run the step you need. Extra setup by step:

| Steps | Also run |
|---|---|
| 13–14, 25, 33, 38 (freezing of MambaVision-T) | Step 13 Part A and Part B; Step 38 also needs Step 33A |
| 29, 32 (MambaVision-S) | Steps 29A and 29B |
| 37, 42 (MambaVision-B) | Steps 37A and 37B |
| 40, 40-R (384 px) | Step 40A |
| 39 (ImageNet-C) | Step 39A |

The rebuilt pipeline reproduces the baseline exactly:
- 41,094 / 50,000 correct (82.188%),
- the fast cached path (Step 24) agrees with the original loader on 100% of predictions.

---

## Step map

Steps 18 and 23 are **result summaries only** (export cells, not experiments). The export cell at the end packs every checkpoint into one zip.

| Step | What it does | Checkpoint(s) |
|---|---|---|
| 0–5b | Kaggle setup, dependencies, MambaVision-T, preprocessing, dataset, seed-42 order | — |
| 6 | Full 50K baseline (predictions, logits, log-probabilities) | `baseline_logits_probs_new` |
| 7 | Architecture map of the 12 mixer blocks | — |
| 8 | Reference activations and per-channel means (256 images) | `step8` |
| 9–10 | Single-block mean ablation; paired CI, McNemar, Bonferroni | `step9_block_ablation`, `step10` |
| 11–11d | Session resume; Mamba vs Attention test; grouped ablation | `step11b_summary`, `partA_mannwhitney_50k`, `partC_grouped_ablation` |
| 12 | Per-image predictions of the Step 9 ablations; stratified ablation | `step9_ablated_preds`, `step12_stratified` |
| 13 | Freezing of Δ, B, C: references, patch, 54-configuration grid | `step13_freeze_refs`, `step13_freezing_50k` |
| 14 | Joint freeze of all six Mamba blocks; selectivity fraction | `step14_freeze_all_mamba` |
| 15 | Activation resampling and Spearman vs mean ablation | `resampling_50k_v2`, `resampling_vs_meanablation_spearman` |
| 16 | Shuffled-channel null (one permutation) and damage vs size | `shuffled_null_50k`, `damage_vs_perturbation_table` |
| 17 | SSM vs gate pathway ablation and paired contrast | `pathway_ablation_50k_v3`, `tost_pathway_results` |
| 18 | Result summary (export only) | — |
| 19 | Statistical tests needing no GPU | `step19_report_tests` |
| 20 | DeiT-Small first version; update norms; receptive field first version | `deit_control_50k`, `update_norms_2k` |
| 21A | Additivity, per-block equivalence, sampling audit | `step21a_review_checks` |
| 22A–22I | KL no-op floor; logit/KL for freeze and ablation; interaction range; DeiT with reference preprocessing; external difficulty tiers; high-norm tokens; L3B1 gate diagnostic; figures | `kl_noop_floor`, `figure2_attention_distance_v2`, `deit_control_50k_v2`, `step12_stratified_external`, `high_norm_tokens`, `l3b1_gate_logit_diagnostic` |
| 23 | Result summary (export only) | — |
| 24 | Speed test, uint8 image cache, fast pass (100% identical to baseline) | `step24_speed` |
| 25 | Joint freeze in mean vs position mode; Δ, B, C separately | `step25_joint_freeze_modes` |
| 26 | Group shuffled nulls (all Mamba, all Attention) | `step26_group_nulls` |
| 27 | Shuffled null with 10 permutations per block; pooled bound | `step27_multi_perm_null`, `step27_null_summary` |
| 28 | Positive control: copying in mamba-130m under Δ/B/C freezing | `step28_positive_control` |
| 29 | Replication on MambaVision-S | `step29_mambavision_s` |
| 30 | High-norm token maps, token-0 causal test, log-prob drop for block ablation | `step30_highnorm_heatmaps`, `step30_token_patch`, `step30_e2_logprob` |
| 31 | Pathway ablation with the log-probability drop | `step31_pathway_logprob` |
| 32 | Group shuffled nulls on MambaVision-S | `step32_s_group_nulls` |
| 33 | Frozen SSM as an explicit convolution kernel; truncation; throughput | `step33_conv_kernel` |
| 34 | DeiT-Small with the same ablation and null protocol | `step34_deit_protocol` |
| 35 | MambaVision-T at 384 px (global windows): joint freeze, per-block real vs null | `step35_highres_384` |
| 36 | Pairwise block-interaction matrix (66 pairs) | `step36_pairwise`, `step36_pairwise_summary` |
| 37 | Replication on MambaVision-B | `step37_mambavision_b` |
| 38 | Gain-matched truncation of the frozen kernel (scale, fold) | `step38_renorm_truncation` |
| 39 | Four ImageNet-C corruptions, real vs null (10K subset); clean log-prob reference | `step39_imagenet_c` |
| 40 | 384 px, four null permutations, 224-px reference means (10K subset) | `step40_384_multiperm` |
| 40-R | 384 px, four null permutations, 384-px reference means (10K subset) | `step40R_384_multiperm_384means` |
| 41 | Pairwise interaction against a shuffled-null baseline (4 pairs) | `step41_pairwise_null` |
| 42 | MambaVision-B: extra null permutations for four blocks | `step42_b_multiperm` |
| Export | All checkpoints → one zip (pkl files, JSON, index) | — |

---

## Main results (50,000 images unless stated)

| Result | Value |
|---|---|
| Baseline top-1 (T / S / B / DeiT-Small) | 82.188% / 83.250% / 84.126% / 79.848% |
| Real vs size-matched null damage across blocks (T) | Spearman ρ = 0.83 (10 permutations) |
| Content effect, pooled over 12 blocks (T) | −0.016 pp [−0.049, +0.024]; one-sided 95% bound 0.016 pp (bound X = 0.10 pp) |
| Pooled content effect, S / B | −0.006 pp (bound 0.029) / +0.036 pp (bound 0.060) |
| Block with a correction-surviving content effect | a Level-3 Attention block in each model (L3B3) |
| All Mamba off: real vs group null (T) | −3.90 vs −1.20 pp |
| All Attention off: real vs group null (T) | −20.58 vs −4.21 pp |
| Strongest pair interaction, real vs null (L2B4 + L2B5) | −2.16 vs −0.29 pp |
| Joint freeze of Δ, B, C, position mode (image-dependence only) | T 20.3%, S 32.1%, B 28.2% of the all-Mamba-off damage |
| Positive control (mamba-130m) | Freezing B or C removes in-context copying; first-copy loss unchanged |
| Explicit convolution vs frozen scan | Identical on 100% of images; 64 taps still 0.67 pp short of the full kernel |
| DeiT-Small, same protocol | Real and null differ by up to 8.1 pp per layer |
| Scan-start token (largest update in 100% of images in L2B3, L3B1) | Patching it alone: ≤ 0.004 pp |

---

## Statistics

- **Effects:** all paired against the unmodified model on the same images.
- **Confidence intervals:** paired bootstrap, 2,000 resamples, seed 42.
- **Tests:** McNemar χ² with continuity correction, exact when b + c < 25.
- **Multiple comparisons:** Bonferroni within each family (0.05/12 per block, 0.05/54 for the freezing grid) or Holm where stated.
- **Content effect C:** null accuracy change − real accuracy change; C > 0 means the real ablation is more damaging than a size-matched perturbation.
- **Equivalence margin** ε = 0.15 pp for the pathway contrast (chosen after the data were seen).
