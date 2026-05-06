# ICD-10 Automatic Coding via DR-CAML with Semantic Compensation

> 東海大學資訊工程學系 畢業專題  
> 指導教授：[待填]  
> 學生：劉晏呈  
> 2025-2026

---

## Overview

Automated ICD-10 coding from clinical discharge summaries using **DR-CAML** (Description Regularized Convolutional Attention for Multi-Label Classification), trained on Taiwan Veterans General Hospital (台中榮總) data with validation on MIMIC-IV.

**Core contributions:**

- **Tunghai Standard Cleaning** (東海標準清洗法): F-code removal + 4-digit code truncation + MIN_FREQ filtering → reduces label space from 14,789 to 2,099
- **DR-CAML with Focal Loss**: Focal Loss + λ=0.1 Description Regularization improves Macro F1 by 6% over BCE baseline
- **Threshold calibration**: Demonstrated threshold 0.3-0.4 outperforms default 0.5 for extreme multi-label ICD coding
- **Domain gap analysis**: Quantified MIMIC-IV vs Taiwan hospital text length disparity (3,207 vs 422 median tokens)

## Key Results

| Model | Dataset | Micro F1 | Macro F1 | Notes |
|---|---|---|---|---|
| V5.1 | Taiwan Hospital | 65.40% | 21.93% | BERT [CLS], optimal TH=0.3 |
| V6 | Taiwan Hospital | **67.27%** | **38.42%** | BERT DR-CAML, λ=0.01 |
| V6OC | Taiwan Hospital | 61.81% | 28.06% | DR-CAML + Focal Loss + λ=0.1, optimal TH=0.4 |
| V6.4 | MIMIC-IV | 41.44% | 7.03% | DR-CAML on cleaned MIMIC |

## Data Pipeline

```
Raw Hospital Records (2015-2022)
    │
    ▼ V2: 14,789 labels (MIN_FREQ=1)
    │
    ▼ V4: 14,219 labels (F-codes removed)
    │
    ▼ V5: 2,099 labels (4-code truncation + MIN_FREQ=50)
    │
    ▼ V6: DR-CAML trained on V5 data
```

## Architecture: DR-CAML

```
Clinical Note ──→ Bio_ClinicalBERT ──→ H ∈ R^(seq×768)
                                            │
                                    Label-Aware Attention
                                            │
                                    V ∈ R^(num_labels×768)
                                            │
                            ┌───────────────┴───────────────┐
                            │                               │
                     out_weights · V              ||out_weights - target_emb||²
                     → BCE / Focal Loss           → MSE Regularization (λ)
```

- **Label-Aware Attention**: Each ICD code learns to attend to different parts of the clinical note
- **Description Regularization**: Model's classifier weights are pulled toward Bio_ClinicalBERT embeddings of ICD code descriptions
- **Focal Loss** (γ=2): Down-weights easy negatives, forcing model to focus on rare diseases

## Reproducibility

### Requirements

- Python 3.10+
- PyTorch 2.x, Transformers, scikit-learn
- GPU: 12GB+ VRAM (tested on RTX 8000-12Q)

### Training

```bash
cd src
python train_v6oc.py  # DR-CAML + Focal Loss, 2099 labels
```

### Evaluation

```bash
python universal_evaluator.py \
    --model_type drcaml \
    --model_weights v6oc_best_model.pth \
    --label_map label_map_v5.json \
    --val_data val_v5.json \
    --embeddings v5_label_embeddings.pt
```

## File Structure

```
├── src/                    # Training and evaluation scripts
│   ├── train_v6oc.py       # DR-CAML + Focal Loss trainer
│   ├── universal_evaluator.py
│   └── build_mf20_data.py
├── data/                   # Preprocessed datasets
├── results/                # Model checkpoints and predictions
├── docs/                   # Documentation
│   └── ABLATION_FINAL.md   # Ablation study results
└── README.md
```

## Academic References

- Mullenbach et al. (2018). "Explainable Prediction of Medical Codes from Clinical Text." *NAACL*. — CAML/DR-CAML architecture
- Lin et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*. — Focal Loss
- Li & Yu (2020). "Multi-Filter Residual Convolutional Neural Network for Text Classification." — MultiResCNN
- Vu et al. (2020). "A Label Attention Model for ICD Coding from Clinical Text." *IJCAI*. — LAAT
- PLM-ICD (2022). "PLM-ICD: Automatic ICD Coding with Pretrained Language Models." *ClinicalNLP*. — Segment pooling for long documents

## License

[待定]
