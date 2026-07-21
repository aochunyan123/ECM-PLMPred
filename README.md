## ECM-PLMPred

Extracellular matrix (ECM) proteins are essential components of tissue architecture and participate in cell adhesion, signaling, development, and disease. Here, we provide **ECM-PLMPred**, a computational framework that combines mean-pooled ProtT5 protein language model embeddings with a multilayer perceptron (MLP) classifier for ECM protein prediction.


---

## Environment

The framework was developed and tested under the following environment:

- numpy >= 1.23
- pandas >= 1.5
- scikit-learn >= 1.2
- torch >= 2.0
- biopython >= 1.80
- tqdm >= 4.64
- transformers >= 4.30
- sentencepiece >= 0.1.99
- protobuf >= 3.20

All required packages are listed in `requirements.txt`.

## End-to-End Reproducibility

Follow the steps below to generate ProtT5 mean embeddings, train the MLP classifier, and run prediction with the packaged best model.

### Environment setup

We recommend using a virtual environment such as Conda:

```bash
git clone https://github.com/aochunyan123/ECM-PLMPred.git
cd ECM-PLMPred

conda create -n ecm-prott5 python=3.10 -y
conda activate ecm-prott5
pip install -r requirements.txt
```

The first feature-extraction run downloads `Rostlab/prot_t5_xl_uniref50`. A CUDA GPU is strongly recommended.

## Running

### 1. Extract ProtT5 mean embeddings

Input: FASTA files  
Output: pickle files in the format `{"embeddings": {sequence_id: np.ndarray}}`, where each array has shape `(D,)`.

```bash
python scripts/extract_prott5_mean_embeddings.py \
  --fasta data/training.fasta \
  --output data/train_prott5_mean_embeddings_ECM.pkl \
  --device cuda \
  --fp16

python scripts/extract_prott5_mean_embeddings.py \
  --fasta data/testing.fasta \
  --output data/test_prott5_mean_embeddings_ECM.pkl \
  --device cuda \
  --fp16
```

Proteins longer than the configured chunk length are processed in chunks and pooled over the complete sequence. Repeated `NO_ENTRY` identifiers are retained using stable `__dupN` suffixes so that no FASTA record is overwritten.

### 2. Train the model

```bash
python scripts/train_mlp.py \
  --feature_names ProtT5_mean \
  --train_pkls data/train_prott5_mean_embeddings_ECM.pkl \
  --test_pkls data/test_prott5_mean_embeddings_ECM.pkl \
  --cv_folds 5 \
  --select_metric BACC \
  --save_path models/best_mlp.pt \
  --output_prefix results/prott5_mean_mlp \
  --epochs 200 \
  --patience 30
```

The training script performs stratified five-fold cross-validation, saves the best checkpoint for each fold, evaluates each fold-trained model on the independent test set, and selects the best overall model according to validation BACC.

### 3. Predict with the packaged best model

The repository includes the saved checkpoint:

```text
models/ProtT5_mean_best_overall.pt
```

Run inference after generating the test ProtT5 mean embeddings:

```bash
python scripts/predict_with_saved_mlp.py \
  --embeddings data/test_prott5_mean_embeddings_ECM.pkl \
  --model models/ProtT5_mean_best_overall.pt \
  --out_csv results/inference_predictions.csv
```

### 4. Evaluate predictions

When labels are present in the sequence identifiers, calculate the prediction metrics as follows:

```bash
python scripts/evaluate_predictions.py \
  --predictions results/inference_predictions.csv \
  --out_csv results/inference_metrics.csv
```

Small FASTA subsets for command and format checks are available in `examples/`. 
