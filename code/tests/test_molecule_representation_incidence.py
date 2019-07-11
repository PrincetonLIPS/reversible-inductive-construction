import pytest
import numpy as np

from induc_gen import molecule_representation as mr
from induc_gen import molecule_edit as me


def mol_incidence_reference(mol):
    MAX_NB = 6
    in_bonds, all_bonds = [], []
    n_atoms = mol.GetNumAtoms()

    for atom in mol.GetAtoms():
        in_bonds.append([])

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        x = a1.GetIdx()
        y = a2.GetIdx()

        b = len(all_bonds)
        all_bonds.append((x, y))
        in_bonds[y].append(b)

        b = len(all_bonds)
        all_bonds.append((y, x))
        in_bonds[x].append(b)

    total_bonds = len(all_bonds)
    a_graph = -np.ones((n_atoms, MAX_NB), dtype=np.int32)
    b_graph = -np.ones((total_bonds, MAX_NB), dtype=np.int32)

    for a in range(n_atoms):
        for i, b in enumerate(in_bonds[a]):
            a_graph[a, i] = b

    for b1 in range(total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                b_graph[b1, i] = b2

    return a_graph, b_graph


def test_atom_bond_incidence_segment_reference():
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')

    scopes, index = mr.atom_bond_list_segment(mol)
    a_graph, _ = mol_incidence_reference(mol)

    assert scopes.shape[0] == a_graph.shape[0]

    for i in range(scopes.shape[0]):
        assert np.all(index[scopes[i, 0]:scopes[i, 0] + scopes[i, 1]] == a_graph[i, a_graph[i] >= 0])


def test_bond_incidence_segment_reference():
    mol = me.get_mol('O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O')

    scopes, index = mr.bond_incidence_list_segment(mol)
    _, b_graph = mol_incidence_reference(mol)

    assert scopes.shape[0] == b_graph.shape[0]

    for i in range(scopes.shape[0]):
        incidence_segment = index[scopes[i, 0]:scopes[i, 0] + scopes[i, 1]]
        incidence_graph = b_graph[i, b_graph[i] >= 0]
        assert len(incidence_segment) == len(incidence_graph)
        assert np.all(incidence_segment == incidence_graph)
