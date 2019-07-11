""" Python implementation for molecule representation utilities. """

import numpy as np
from .. import Chem


ELEM_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6


def onek_encoding_unk(x, allowable_set):
    """ Compute one-hot encoding of the given categorical value `x` """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) +
        onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
        onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3]) +
        [atom.GetIsAromatic()],
        dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.BondType.SINGLE,
             bt == Chem.BondType.DOUBLE,
             bt == Chem.BondType.TRIPLE,
             bt == Chem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    return np.array(fbond + fstereo, dtype=np.float32)


def fill_atom_features(out, mol):
    for i, atom in enumerate(mol.GetAtoms()):
        out[i] = atom_features(atom)


def fill_bond_features(out, mol):
    for i, bond in enumerate(mol.GetBonds()):
        out[2 * i, :ATOM_FDIM] = atom_features(bond.GetBeginAtom())
        out[2 * i, ATOM_FDIM:] = bond_features(bond)
        out[2 * i + 1, :ATOM_FDIM] = atom_features(bond.GetEndAtom())
        out[2 * i + 1, ATOM_FDIM:] = bond_features(bond)


def fill_atom_bond_list(out, mol, max_neighbours):
    for i, atom in enumerate(mol.GetAtoms()):
        for j, bond in enumerate(atom.GetBonds()):
            out[i, j] = 2 * bond.GetIdx() + int(bond.GetBeginAtom().GetIdx() == atom.GetIdx()) + 1


def fill_bond_incidence_list(out, mol, max_neighbours):
    def _fill_bond_incidence(i, bond, a):
        for j, bond2 in enumerate(a.GetBonds()):
            if bond2.GetIdx() == bond.GetIdx():
                continue
            out[i, j] = 2 * bond2.GetIdx() + int(bond2.GetBeginAtom().GetIdx() == a.GetIdx()) + 1

    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        _fill_bond_incidence(2 * i + 1, bond, a1)
        _fill_bond_incidence(2 * i + 2, bond, a2)

    return out


def fill_atom_bond_list_sparse(out_values, out_index, mol):
    current_idx = 0

    if out_index.shape != (2, mol.GetNumBonds() * 2):
        raise ValueError("out_index must be 2 x (2 x num_atoms) array.")

    for i, atom in enumerate(mol.GetAtoms()):
        for j, bond in enumerate(atom.GetBonds()):
            out_index[0, current_idx] = i
            out_index[1, current_idx] = 2 * bond.GetIdx() + int(bond.GetBeginAtom().GetIdx() == atom.GetIdx())
            out_values[current_idx] = 1 / np.sqrt(len(atom.GetBonds()))
            current_idx += 1

    return out_values, out_index


def fill_atom_bond_list_segment(out_scopes, out_index, mol):
    current_idx = 0
    offset = 0

    for i, atom in enumerate(mol.GetAtoms()):
        out_scopes[i, 0] = offset
        out_scopes[i, 1] = len(atom.GetBonds())
        offset += len(atom.GetBonds())

        for j, bond in enumerate(atom.GetBonds()):
            out_index[current_idx] = 2 * bond.GetIdx() + int(bond.GetBeginAtom().GetIdx() == atom.GetIdx())
            current_idx += 1

    return out_scopes, out_index


def get_edge_incidence_size(mol):
    return sum(a.GetDegree() * (a.GetDegree() - 1) for a in mol.GetAtoms())


def fill_bond_incidence_list_sparse(out_values, out_index, mol):
    current_idx = 0

    def _fill_bond_incidence(i, bond, a, value):
        nonlocal current_idx
        for bond2 in a.GetBonds():
            if bond2.GetIdx() == bond.GetIdx():
                continue

            out_index[0, current_idx] = i
            out_index[1, current_idx] = 2 * bond2.GetIdx() + int(bond2.GetBeginAtom().GetIdx() == a.GetIdx())
            out_values[current_idx] = value
            current_idx += 1

    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        degree = a1.GetDegree() + a2.GetDegree() - 2
        value = 1 / np.sqrt(degree)

        _fill_bond_incidence(2 * i, bond, a1, value)
        _fill_bond_incidence(2 * i + 1, bond, a2, value)
    return out_values, out_index


def fill_bond_incidence_list_segment(out_scopes, out_index, mol):
    current_idx = 0
    offset = 0

    def _fill_bond_incidence(j, b, a):
        nonlocal current_idx
        nonlocal offset

        out_scopes[j, 0] = offset
        out_scopes[j, 1] = len(a.GetBonds() - 1)
        offset += len(a.GetBonds() - 1)

        for bond2 in a.GetBonds():
            if bond2.GetIdx() == b.GetIdx():
                continue

            out_index[current_idx] = 2 * bond2.GetIdx() + int(bond2.GetBeginAtom().GetIdx() == a.GetIdx())
            current_idx += 1

    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        _fill_bond_incidence(2 * i, bond, a1)
        _fill_bond_incidence(2 * i + 1, bond, a2)

    return out_scopes, out_index
