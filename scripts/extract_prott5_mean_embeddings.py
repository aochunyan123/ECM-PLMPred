"""Extract protein-level mean embeddings from FASTA sequences with ProtT5."""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm


def read_fasta(path):
    sequences = {}
    id_counts = defaultdict(int)
    for record in SeqIO.parse(path, "fasta"):
        base_id = str(record.id)
        id_counts[base_id] += 1
        seq_id = base_id if id_counts[base_id] == 1 else f"{base_id}__dup{id_counts[base_id]}"
        while seq_id in sequences:
            id_counts[base_id] += 1
            seq_id = f"{base_id}__dup{id_counts[base_id]}"
        sequence = str(record.seq).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"Empty sequence: {seq_id}")
        # ProtT5 convention: map rare/ambiguous residues to X.
        sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", sequence)
        sequences[seq_id] = sequence
    if not sequences:
        raise ValueError(f"No sequences found in {path}")
    duplicate_records = sum(count - 1 for count in id_counts.values())
    if duplicate_records:
        print(f"Renamed {duplicate_records} duplicate FASTA IDs with __dupN suffixes")
    return sequences


def make_chunks(sequences, max_residues):
    chunks = []
    lengths = {}
    for seq_id, sequence in sequences.items():
        lengths[seq_id] = len(sequence)
        for start in range(0, len(sequence), max_residues):
            chunks.append((seq_id, start, sequence[start : start + max_residues]))
    return chunks, lengths


def make_batches(chunks, max_tokens):
    """Create length-aware batches bounded by padded residue count."""
    chunks = sorted(chunks, key=lambda item: len(item[2]))
    batches = []
    current = []
    current_max = 0
    for item in chunks:
        new_max = max(current_max, len(item[2]) + 1)  # +1 for EOS
        if current and (len(current) + 1) * new_max > max_tokens:
            batches.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, len(item[2]) + 1)
    if current:
        batches.append(current)
    return batches


def load_model(model_name, device, fp16):
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise ImportError(
            "Install transformers and sentencepiece first: "
            "pip install transformers sentencepiece"
        ) from exc

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    if fp16:
        model = model.half()
    model = model.eval().to(device)
    return tokenizer, model


@torch.inference_mode()
def extract(sequences, model_name, device, max_tokens, max_residues, fp16):
    tokenizer, model = load_model(model_name, device, fp16)
    chunks, lengths = make_chunks(sequences, max_residues)
    batches = make_batches(chunks, max_tokens)
    vector_sums = {}
    residue_counts = defaultdict(int)

    for batch in tqdm(batches, desc="Extracting ProtT5 mean embeddings", unit="batch"):
        spaced_sequences = [" ".join(fragment) for _, _, fragment in batch]
        encoded = tokenizer(
            spaced_sequences,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        for index, (seq_id, _start, fragment) in enumerate(batch):
            token_count = int(attention_mask[index].sum().item())
            residue_token_count = token_count - 1  # exclude EOS
            if residue_token_count != len(fragment):
                raise RuntimeError(
                    f"Unexpected tokenization for {seq_id}: "
                    f"{len(fragment)} residues became {residue_token_count} residue tokens"
                )
            chunk_sum = hidden[index, :residue_token_count].float().sum(dim=0).cpu().numpy()
            if seq_id not in vector_sums:
                vector_sums[seq_id] = chunk_sum
            else:
                vector_sums[seq_id] += chunk_sum
            residue_counts[seq_id] += len(fragment)

    embeddings = {
        seq_id: np.asarray(vector_sums[seq_id] / residue_counts[seq_id], dtype=np.float32)
        for seq_id in sequences
    }
    if any(residue_counts[key] != lengths[key] for key in sequences):
        raise RuntimeError("Internal residue-count mismatch")
    return embeddings


def save_output(path, embeddings, args):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    output = {
        "embeddings": embeddings,
        "model": args.model,
        "pooling": "mean",
        "max_residues_per_chunk": args.max_residues,
        "embedding_type": "protein-level mean-pooled embedding",
    }
    with open(path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ProtT5 mean embeddings compatible with train_mlp.py."
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output embedding PKL")
    parser.add_argument("--model", default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-tokens", type=int, default=1001)
    parser.add_argument("--max-residues", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if args.max_residues < 1 or args.max_tokens < args.max_residues + 1:
        parser.error("--max-tokens must be at least --max-residues + 1 (for EOS)")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fp16 and args.device != "cuda":
        parser.error("--fp16 requires --device cuda")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    fasta_sequences = read_fasta(cli_args.fasta)
    mean_embeddings = extract(
        fasta_sequences,
        cli_args.model,
        cli_args.device,
        cli_args.max_tokens,
        cli_args.max_residues,
        cli_args.fp16,
    )
    save_output(cli_args.output, mean_embeddings, cli_args)
    print(f"Saved {len(mean_embeddings)} embeddings to {cli_args.output}")
