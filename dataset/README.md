## GridGraph 4x4 Generation

The GridGraph 4x4 dataset (GG4x4) can be generated using the scripts in the `dataset/` folder.

### Step 1: Generate Balanced Dataset
First, generate the raw balanced dataset with graph questions:

```bash
python create_balanced_datasets.py \
    --questions 38000 \
    --min-vertices 4 \
    --max-vertices 10 \
    --grid-size 4 \
    --enforce-planarity \
    --no-none-of-above \
    --seed 42 \
    --edge-density-mode "long_tail" \
    --min-density 0.1 \
    --max-density 0.5
```

### Step 2: Split into Train/Val/Test
Split the dataset into canonical train/validation/test splits:

```bash
python split_dataset_canonical.py \
    --input collections/balanced/data/v4-10_undirected_planar_c112_image_text \
    --output collections/graph_dataset_4x4 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42
```

### Step 3: Create Image/Text Variations
Generate the image-only and text-only variations for KD training:

```bash
python create_image_text_variants.py \
    --input collections/graph_dataset_4x4
```

This creates:
- `train/` - Image+text (for teacher)
- `train_image_only/` - Image only (for student)  
- `train_text_only/` - Text only (baseline)
- Same for `validation/` and `test/`

---

## ChartQA

ChartQA is saved under the `dataset/collections/chartqa/` folder and can be used directly.

Alternatively, ChartQA can be downloaded from its [official repository](https://github.com/vis-nlp/ChartQA) but needs to be placed in `dataset/collections/chartqa/` with the following structure:

```
chartqa/
├── train/
│   ├── train_human.json
│   ├── train_augmented.json
│   └── png/
├── val/
│   ├── val_human.json
│   ├── val_augmented.json
│   └── png/
└── test/
    ├── test_human.json
    ├── test_augmented.json
    └── png/
```


