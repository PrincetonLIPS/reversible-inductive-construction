import bz2
import os
import pickle
import pytest
import torch
import numpy as np

from genric import molecule_representation as mr
from genric import molecule_edit as me

from genric.molecule_representation import _implementation_python as imp_py

try:
    from genric.genric_extensions import molecule_representation as imp_c
except ImportError:
    imp_c = None


def test_atom_embedding():
    mol = me.get_mol('CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1')

    expected_0 = [
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
        0., 0., 0.]

    expected_8 = [
        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
        0., 0., 0.]

    result_0 = mr.atom_features(mol.GetAtomWithIdx(0))
    result_8 = mr.atom_features(mol.GetAtomWithIdx(8))

    assert expected_0 == list(result_0)
    assert expected_8 == list(result_8)


def test_bond_embedding():
    mol = me.get_mol('CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1')

    expected_0 = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
    expected_8 = [1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.]

    result_0 = mr.bond_features(mol.GetBondWithIdx(0))
    result_8 = mr.bond_features(mol.GetBondWithIdx(8))

    assert expected_0 == list(result_0)
    assert expected_8 == list(result_8)


def test_fill_atom_features():
    mol = me.get_mol('CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1')
    num_atoms = mol.GetNumAtoms()

    result_py = np.zeros((num_atoms, mr.ATOM_FDIM), dtype=np.float32)
    result_c = np.zeros((num_atoms, mr.ATOM_FDIM), dtype=np.float32)

    imp_py.fill_atom_features(result_py, mol)
    imp_c.fill_atom_features(result_c, mol)

    assert np.allclose(result_py, result_c)


def test_fill_bond_features():
    mol = me.get_mol('CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1')
    num_bonds = mol.GetNumBonds()

    result_py = np.zeros((2 * num_bonds, mr.ATOM_FDIM + mr.BOND_FDIM), dtype=np.float32)
    result_c = np.zeros((2 * num_bonds, mr.ATOM_FDIM + mr.BOND_FDIM), dtype=np.float32)

    imp_py.fill_bond_features(result_py, mol)
    imp_c.fill_bond_features(result_c, mol)

    assert np.allclose(result_py, result_c)


def get_data(request):
    filepath = os.path.join(request.fspath.dirname, 'data', 'test_molecule_representation_data.pkl.bz2')
    with bz2.open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data


@pytest.mark.xfail()
def test_molecule_representation_stereo(request):
    data = get_data(request)

    smiles = data['smiles']
    result = mr.mol2graph(smiles, use_stereo=True)
    expected = data['graph_stereo']

    assert len(expected) == len(result)

    assert torch.all(expected[0] == result[0])
    assert len(torch.nonzero(torch.any(expected[1] != result[1], dim=1))) == 0
    assert torch.all(expected[2] == result[2])
    assert torch.all(expected[3] == result[3])
    assert expected[4] == result[4]


def test_mol2graph_single(request):
    data = get_data(request)

    mol = me.get_mol(data['smiles'][0])

    result = list(mr.mol2graph_single(mol).values())
    expected = data['graph_nostereo']

    def _compare_tensor(a, b):
        return np.allclose(a, b[:a.shape[0], :])

    assert _compare_tensor(result[0], expected[0])
    assert _compare_tensor(result[1], expected[1][1:])


def test_combine_graphs(request):
    data = get_data(request)

    result = list(mr.combine_mol_graph([mr.mol2graph_single(me.get_mol(s)) for s in data['smiles']]).values())
    expected = data['graph_stereo']

    def _compare_tensor(a, b):
        return np.allclose(a, b[:a.shape[0], :])

    assert _compare_tensor(result[0], expected[0])
    # assert _compare_tensor(result[1], expected[1])  # disabled because of stereo-chemistry stuff


def test_mol2graph_single_rings_leaves():
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')
    result = mr.mol2graph_single(mol, include_leaves=True)

    assert 'leaf_ring' in result
    assert 'leaf_atom' in result

    assert result['leaf_atom'].tolist() == [0, 7, 8, 16, 25, 33]

    assert result['leaf_ring'][0][0].tolist() == [0] * 6 + [1] * 6
    assert result['leaf_ring'][0][1].tolist() == [9, 10, 11, 12, 13, 14] + [26, 27, 28, 29, 30, 31]
    assert result['leaf_ring'][1].tolist() == [pytest.approx(1 / np.sqrt(6))] * 12


def test_combine_graphs_leaf_rings_singleton_sequence():
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')
    result = mr.mol2graph_single(mol, include_leaves=True)
    result = mr.combine_mol_graph([result])

    assert 'leaf_ring' in result
    assert 'leaf_atom' in result

    assert np.all(result['leaf_ring_scope'] == np.array([[0, 2]]))


def test_mol2graph_single_rings():
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')
    result = mr.mol2graph_single(mol, include_rings=True)

    assert 'ring_bond_idx' in result
    assert 'ring_bond_order' in result

    assert len(result['ring_bond_idx']) == 27 * 2


def test_combine_graphs_bond_rings():
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')
    result = mr.mol2graph_single(mol, include_leaves=True, include_rings=True)
    result = mr.combine_mol_graph([result])

    assert 'ring_bond_idx' in result
    assert 'ring_bond_order' in result

    assert np.allclose(result['ring_scope'], np.array([[0, 27 * 2]]))


@pytest.mark.parametrize("imp", [imp_py, imp_c])
def test_atom_incidence_sparse(imp):
    from scipy import sparse
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')

    rng = np.random.RandomState(42)

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    num_bond_emb = 2 * mol.GetNumBonds()

    bond_embedding = rng.randn(num_bond_emb + 1, 4)
    bond_embedding[0, :] = 0

    result_dense = np.zeros((num_atoms, 6), dtype=np.int32)
    imp.fill_atom_bond_list(result_dense, mol, 6)

    result_sparse_idx = np.zeros((2, 2 * num_bonds), dtype=np.int32)
    result_sparse_values = np.ones(2 * num_bonds, dtype=np.float32)
    imp.fill_atom_bond_list_sparse(result_sparse_values, result_sparse_idx, mol)
    result_sparse_values = np.ones(2 * num_bonds, dtype=np.float32)

    result_sparse = sparse.coo_matrix(
        (result_sparse_values, result_sparse_idx), shape=(num_atoms, num_bond_emb))

    atom_emb_sparse = result_sparse.dot(bond_embedding[1:])
    atom_emb_dense = np.sum(
        np.take(bond_embedding, result_dense.flat, axis=0).reshape(result_dense.shape + (4,)),
        axis=1)

    assert atom_emb_sparse.shape == atom_emb_dense.shape
    assert np.allclose(atom_emb_sparse, atom_emb_dense)


@pytest.mark.parametrize("imp", [imp_py, imp_c])
def test_bond_incidence_sparse(imp):
    from scipy import sparse
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')

    rng = np.random.RandomState(42)

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    num_bond_emb = 2 * mol.GetNumBonds()

    bond_embedding = rng.randn(num_bond_emb + 1, 4)
    bond_embedding[0, :] = 0

    result_dense = np.zeros((2 * num_bonds + 1, 6), dtype=np.int32)
    imp.fill_bond_incidence_list(result_dense, mol, 6)

    result_dense_count = np.sum(result_dense != 0)

    result_sparse_count = imp.get_edge_incidence_size(mol)

    assert result_dense_count == result_sparse_count

    result_sparse_idx = np.zeros((2, result_dense_count), dtype=np.int32)
    result_sparse_values = np.ones(result_dense_count, dtype=np.float32)

    imp.fill_bond_incidence_list_sparse(result_sparse_values, result_sparse_idx, mol)

    result_sparse_values = np.ones_like(result_sparse_values)
    result_sparse = sparse.coo_matrix(
        (result_sparse_values, result_sparse_idx), shape=(num_bond_emb, num_bond_emb))

    bond_emb_sparse = result_sparse.dot(bond_embedding[1:])
    bond_emb_dense = np.sum(
        np.take(bond_embedding, result_dense.flat, axis=0).reshape(result_dense.shape + (4,)), axis=1)

    assert np.allclose(bond_emb_sparse, bond_emb_dense[1:])


def test_atom_bond_list_segment():
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')

    scopes_c = np.empty((mol.GetNumAtoms(), 2), dtype=np.int32)
    index_c = np.empty(2 * mol.GetNumBonds(), dtype=np.int32)

    imp_c.fill_atom_bond_list_segment(scopes_c, index_c, mol)

    scopes_py = np.empty((mol.GetNumAtoms(), 2), dtype=np.int32)
    index_py = np.empty(2 * mol.GetNumBonds(), dtype=np.int32)

    imp_py.fill_atom_bond_list_segment(scopes_py, index_py, mol)

    assert np.all(scopes_c == scopes_py)
    assert np.all(index_c == index_py)
