""" Module to compute representations of single molecules. """

from collections import OrderedDict
import numpy as np

from .. import Chem, chemutils
from ..vocabulary import AtomTuple

try:
    from ..genric_extensions import molecule_representation as mr_native
except ImportError:
    from . import _implementation_python as mr_native

from ._implementation_python import atom_features, bond_features, ELEM_LIST, ATOM_FDIM, BOND_FDIM

MAX_NB = 6


def ring_info(mol):
    ring_bond_idx = []
    ring_bond_order = []
    num_bonds = 0

    for i, bond in enumerate(mol.GetBonds()):
        if not bond.IsInRing():
            continue

        num_bonds += 1

        bond_order = int(AtomTuple.from_atom(bond.GetBeginAtom()) < AtomTuple.from_atom(bond.GetEndAtom()))

        ring_bond_idx.append(2 * i)
        ring_bond_idx.append(2 * i + 1)
        ring_bond_order.append(bond_order)
        ring_bond_order.append(1 - bond_order)

    return np.array(ring_bond_idx, dtype=np.int32), np.array(ring_bond_order, dtype=np.int32)


def _normalize_adjacency_values(values, normalization):
    if normalization == 'sum':
        return np.ones_like(values)
    elif normalization == 'sqrt':
        return values
    elif normalization == 'mean':
        return np.square(values, out=values)
    else:
        raise ValueError("Unknown normalization type {0}".format(normalization))


def atom_bond_list(mol, normalization='sum'):
    """ Computes the atom-bond incidence list.

    This method computes the atom-bond incidence list.
    For each atom (row), the bonds it belongs to are enumerated, and recorded
    in the ith column as an index.

    Parameters
    ----------
    mol: a rdkit molecule for which to compute the list.
    normalization: 'sum', 'mean' or 'sqrt'. The normalization to apply.

    Returns
    -------
    Elements with represent a sparse matrix in COO format.
    """
    index = np.empty((2, 2 * mol.GetNumBonds()), dtype=np.int32)
    values = np.empty(2 * mol.GetNumBonds(), dtype=np.float32)
    mr_native.fill_atom_bond_list_sparse(values, index, mol)

    values = _normalize_adjacency_values(values, normalization)

    return index, values, (mol.GetNumAtoms(), 2 * mol.GetNumBonds())


def atom_bond_list_segment(mol):
    """ Computes the atom-bond incidence list in segmented form.

    This function returns the atom-bond list in segmented format.

    Parameters
    ----------
    mol: a rdkit molecule for which to compute the list.
    """
    scopes = np.empty((mol.GetNumAtoms(), 2), dtype=np.int32)
    index = np.empty(2 * mol.GetNumBonds(), dtype=np.int32)
    mr_native.fill_atom_bond_list_segment(scopes, index, mol)
    return scopes, index


def bond_incidence_list(mol, normalization='sum'):
    """ Computes the bond-bond incidence list.

    This method computes the bond-bond incidence list.
    For each ordered bond (row), the (ordered) bonds with which it shares
    an atom are enumerated, and are recorded in the ith column as a bond index.

    When recording incident bonds, only incoming bonds are recorded (that is, the
    orientation of the bond is such that the second atom is shared with the bond
    being considered).

    Parameters
    ----------
    mol: a rdkit molecule for which to compute the list.
    normalization: 'sum', 'mean' or 'sqrt'. The normalization to apply.

    Returns
    -------
    a 2-d numpy array representing the given list. It has length 2 * num_bonds + 1.
    """
    num_elements = mr_native.get_edge_incidence_size(mol)

    index = np.empty((2, num_elements), dtype=np.int32)
    values = np.empty(num_elements, dtype=np.float32)

    mr_native.fill_bond_incidence_list_sparse(values, index, mol)

    values = _normalize_adjacency_values(values, normalization)

    return index, values, (2 * mol.GetNumBonds(), 2 * mol.GetNumBonds())


def bond_incidence_list_segment(mol):
    """ Computes the bond-bond incidence list in segmented format. """
    num_elements = mr_native.get_edge_incidence_size(mol)

    scopes = np.empty((2 * mol.GetNumBonds(), 2), dtype=np.int32)
    index = np.empty(num_elements, dtype=np.int32)

    mr_native.fill_bond_incidence_list_segment(scopes, index, mol)
    return scopes, index


def atom_leaves_index(mol):
    return np.array(chemutils.get_atom_leaves(mol), dtype=np.int32)


def ring_leaves_index(mol):
    """ Computes representation for ring leaves in molecules.

    This function computes structures for ring leaf information. It is returned
    as a tuple (and additional length information).

    Each array of the tuple represents an atom in the molecule. The first array indicates
    which leaf ring the atom belongs to, whereas the second array indicates the index
    of the atom in the molecule.
    """
    leaves = chemutils.get_ring_leaves(mol)
    leaves = [np.array(x, dtype=np.int64) for x in leaves]
    leaf_length = np.array([len(x) for x in leaves], dtype=np.int32)

    idx_atom = np.concatenate([np.zeros([0], dtype=np.int64)] + leaves, axis=0)
    idx_leaf = np.repeat(np.arange(len(leaves)), leaf_length)

    idx = np.stack((idx_leaf, idx_atom))
    values = np.repeat(np.reciprocal(np.sqrt(leaf_length.astype(np.float32)), where=leaf_length != 0), leaf_length)
    shape = (len(leaves), mol.GetNumAtoms())

    return (idx, values, shape), len(leaves)


def mol2graph_single(mol, include_leaves=False, include_rings=False, normalization='sum'):
    """ Computes graph representation of given molecule.

    Parameters
    ----------
    mol: a `rdkit.Chem.Mol` representing the molecule for which to compute the representation
    include_leaves: if True, also computes index informations for the leaves in the molecule.
    include_rings: if True, also computes bond ring information for bonds belonging to rings.
    normalization: the normalization to be used in aggregating bond messages.

    Returns
    -------
    A ordered dictionary of 4 numpy tensors.

    feature_atoms: a 2-d floating point array representing the feature embedding for each atom.
    feature_bonds: a 2-d floating point array representing the features embedding for each bond.
    atom_incidence: a 2-d integer array representing the incidence between each atom and bond.
    bond_incidence: a 2-d integer array representing the incidence between bonds.
    """
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    fatoms = np.zeros((num_atoms, ATOM_FDIM), dtype=np.float32)
    fbonds = np.zeros((2 * num_bonds, ATOM_FDIM + BOND_FDIM), dtype=np.float32)

    mr_native.fill_atom_features(fatoms, mol)
    mr_native.fill_bond_features(fbonds, mol)

    agraph = atom_bond_list(mol, normalization=normalization)
    bgraph = bond_incidence_list(mol, normalization=normalization)

    result = [
        ('atom_feature', fatoms),
        ('bond_feature', fbonds),
        ('atom_incidence', agraph),
        ('bond_incidence', bgraph)]

    count = {
        'atom': num_atoms,
        'bond': num_bonds
    }

    if include_leaves:
        ring_leaf_idx, num_ring_leaves = ring_leaves_index(mol)
        result.append(('leaf_ring', ring_leaf_idx))

        count['leaf_ring'] = num_ring_leaves

        atom_leaf_idx = atom_leaves_index(mol)
        result.append(('leaf_atom', atom_leaf_idx))

        count['leaf_atom'] = len(atom_leaf_idx)

    if include_rings:
        ring_bond_idx, ring_bond_order = ring_info(mol)

        result.append(('ring_bond_idx', ring_bond_idx))
        result.append(('ring_bond_order', ring_bond_order))
        count['ring'] = len(ring_bond_idx)

    result.append(('count', count))

    return OrderedDict(result)
