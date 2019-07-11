import numpy as np


def make_atom_action_labels(vocab_encoder):
    if vocab_encoder.canonical:
        offsets = vocab_encoder._canonical_encoder._atom_canonical_offsets
    else:
        offsets = vocab_encoder._atom_offsets

    lengths = np.diff(offsets)
    vocab_idx = np.repeat(np.arange(len(lengths)), lengths)
    atom_idx = np.concatenate([np.arange(x) for x in lengths])

    atom_symbol = [
        a.GetSymbol() for s, mol in vocab_encoder.vocab for a in mol.GetAtoms()
    ]

    return {
        'vocab_idx': vocab_idx,
        'atom_idx': atom_idx,
        'atom_symbol': atom_symbol
    }


def log_model_embeddings(state_dict, writer, vocab_encoder):
    atom_action_labels = make_atom_action_labels(vocab_encoder)
    atom_metadata = [tuple(x) for x in atom_action_labels]

    writer.add_embedding(state_dict['insert_atom_network.4.weight'], metadata=atom_metadata)


if __name__ == '__main__':
    main()
