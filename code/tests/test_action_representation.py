import numpy as np
import pytest

from induc_gen import molecule_edit as me, vocabulary
from induc_gen.data_utils import get_vocab
from induc_gen.molecule_models import action_representation as ar


def action_mol_to_integer(act, mol, encoder):
    num_leaves = len(me.get_leaves(mol))
    bond_ring_idx = [b.GetIdx() for b in mol.GetBonds()]

    return ar.action_to_integer(act, num_leaves, mol.GetNumAtoms(), bond_ring_idx, encoder)


def roundtrip_action(act, vocab, mol):
    encoder = ar.VocabInsertEncoder(vocab)
    result, offsets, lengths = action_mol_to_integer(act, mol, encoder)
    action_roundtrip = ar.integer_to_action(result, lengths, encoder)

    return action_roundtrip


def test_action_to_integer_roundtrip_insert_atom():
    mol = me.get_mol('CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1')
    vocab = vocabulary.Vocabulary()
    act = me.generate_random_atom_insert(mol, mol.GetAtomWithIdx(1), vocab, rng=np.random.RandomState(20))

    action_roundtrip = roundtrip_action(act, vocab, mol)

    assert list(act.to_array()) == list(action_roundtrip.to_array())


def test_action_to_integer_roundtrip_delete():
    mol = me.get_mol('CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1')
    vocab = vocabulary.Vocabulary()
    act = me.enumerate_deletion_actions(mol)[1]

    action_roundtrip = roundtrip_action(act, vocab, mol)

    assert list(act.to_array()) == list(action_roundtrip.to_array())


def test_action_canonical_actions():
    mol = me.get_mol('C1CCC1')
    atom_equiv, num_equiv = ar.compute_canonical_atom_insert_locations(mol)

    assert num_equiv == 1
    assert list(atom_equiv) == [0] * 4


def test_action_canonical_actions_nitrogen():
    mol = me.get_mol('C1CNCCNC1')
    atom_equiv, num_equiv = ar.compute_canonical_atom_insert_locations(mol)

    assert num_equiv == 4


@pytest.mark.parametrize("seed", [20, 40, 50])
def test_action_canonical_roundtrip(seed):
    mol = me.get_mol('CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1')
    vocab = vocabulary.Vocabulary()

    act = me.generate_random_atom_insert(mol, mol.GetAtomWithIdx(1), vocab, rng=np.random.RandomState(seed))

    encoder = ar.VocabInsertEncoder(vocab)
    result, offsets, lengths = action_mol_to_integer(act, mol, encoder)
    action_roundtrip = ar.integer_to_action(result, lengths, encoder)

    assert (action_mol_to_integer(act, mol, encoder)[0] ==
            action_mol_to_integer(action_roundtrip, mol, encoder)[0])
