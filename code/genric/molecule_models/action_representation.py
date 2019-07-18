""" Module for computing action representations.

Actions are associated in several categories. We distinguish the following.

Stop / Continue: global decision associated with the molecule.
Delete: action associated with a leaf (i.e. a set of atoms) of the molecule.
Insert Atom Fusion: action associated with an atom of the molecule, a vocabulary item,
    and a location on that vocabulary.
Insert Bond Fusion: action associated with a bond of the molecule, a vocabulary item,
    and a location on that vocabulary.

"""

import numpy as np
from .. import action, chemutils


def compute_canonical_atom_insert_locations(mol, out=None):
    """ Compute canonical molecule atom insert locations.

    Returns
    -------
    atom_to_equiv: integer array, where the ith entry corresponds to the equivalence
        class of the ith atom in the molecule.

    """
    equivalence_classes = {}

    if out is None:
        out = np.empty(mol.GetNumAtoms(), dtype=np.int32)
    else:
        if out.shape[0] < mol.GetNumAtoms():
            raise ValueError("Not enough space in output array for atoms in molecule.")

    current_class = 0

    for i in range(mol.GetNumAtoms()):
        smiles = chemutils.get_smiles(mol, rootedAtAtom=i, isomeric=False, kekule=True)

        equiv_class = equivalence_classes.get(smiles, None)

        if equiv_class is not None:
            out[i] = equiv_class
        else:
            out[i] = current_class
            equivalence_classes[smiles] = current_class
            current_class += 1

    return out, len(equivalence_classes)


class _CanonicalAtomInsertEncoder:
    """ Helper class to deal with canonical atom insert encoding.

    This helper class mostly deals with the problem when encoding actions,
    that for some items in the vocabulary, inserting at different atoms might
    lead to the same outcome due to the symmetry in the vocabulary item. As this
    is undesirable, this class attempts to compute equivalence classes on the
    atoms in a given vocabulary molecule, and produce an indexing of these equivalence
    classes (instead of the original atoms in the vocabulary).

    Attributes
    ----------
    _atom_canonical_offsets: np.ndarray
        A one-dimensional array, whose ith element indicates the location at which
        the data starts for the ith molecule
    _atom_canonical_equiv: np.ndarray
        A one-dimensional array, which has one element for each atom (of each molecule) in the
        vocabulary. Its elements corresponds to the canonical class of the atom in the given
        molecule
    _atom_canonical_reverse: np.ndarray
        A one-dimensional array, which for each canonical class in a vocabulary molecule, gives
        the index of one atom from that class.
    """
    def __init__(self, vocab, atom_offsets):
        num_vocab_elements = len(vocab)
        self._atom_offsets = atom_offsets

        self._atom_canonical_offsets = np.zeros(num_vocab_elements + 1, dtype=np.int32)
        self._atom_canonical_equiv = np.zeros(atom_offsets[-1], dtype=np.int32)

        for i, (s, m) in enumerate(vocab):
            _, self._atom_canonical_offsets[i] = compute_canonical_atom_insert_locations(
                m, self._atom_canonical_equiv[self._atom_offsets[i]:self._atom_offsets[i + 1]])

        self._atom_canonical_offsets[1:] = np.cumsum(self._atom_canonical_offsets[:-1])
        self._atom_canonical_offsets[0] = 0

        self.num_insert_atom_locations_canonical = self._atom_canonical_offsets[-1]

        self._atom_canonical_reverse = np.zeros(self.num_insert_atom_locations_canonical, dtype=np.int32)
        for i in range(num_vocab_elements):
            s = slice(self._atom_canonical_offsets[i], self._atom_canonical_offsets[i + 1])
            _, self._atom_canonical_reverse[s] = np.unique(
                self._atom_canonical_equiv[self._atom_offsets[i]:self._atom_offsets[i + 1]],
                return_index=True)

    def get_insert_atom_location(self, idx):
        vocab_idx = np.maximum(np.searchsorted(self._atom_canonical_offsets, idx, side='right') - 1, 0)
        atom_idx_canon = idx - self._atom_canonical_offsets[vocab_idx]
        vocab_atom_idx = self._atom_canonical_reverse[atom_idx_canon]
        return vocab_idx, vocab_atom_idx

    def get_insert_atom_index(self, vocab_idx, vocab_atom_idx):
        vocab_atom_canonical_idx = self._atom_canonical_equiv[self._atom_offsets[vocab_idx] + vocab_atom_idx]
        return self._atom_canonical_offsets[vocab_idx] + vocab_atom_canonical_idx


class VocabInsertEncoder:
    def __init__(self, vocab=None, canonical=False):
        """ Initialize a new vocabulary encoder for insert actions.

        This class contains several pre-computed structures to help facilitate
        converting insertion actions (both atom and bond) to linear indices.

        Parameters
        ----------
        vocab: the vocabulary to use
        canonical: if True, indicates that canonical action indices should be used.

        """
        if vocab is None:
            from ..data_utils import get_vocab
            vocab = get_vocab()

        self.vocab = vocab
        self.canonical = canonical

        num_vocab_elements = len(vocab)
        self._atom_offsets = np.zeros(num_vocab_elements + 1, dtype=np.int32)

        self._bond_offsets = np.zeros(num_vocab_elements, dtype=np.int32)

        # For now, we store lengths instead of offsets.

        for i, (s, m) in enumerate(vocab):
            self._atom_offsets[i] = m.GetNumAtoms()
            self._bond_offsets[i] = m.GetNumBonds() if m.GetNumAtoms() >= 2 else 0

        self.num_insert_atom_locations = np.sum(self._atom_offsets)
        self.num_insert_bond_locations = np.sum(self._bond_offsets)

        self._atom_offsets[1:] = np.cumsum(self._atom_offsets[:-1])
        self._bond_offsets[1:] = np.cumsum(self._bond_offsets[:-1])
        self._atom_offsets[0] = 0
        self._bond_offsets[0] = 0

        if self.canonical:
            self._canonical_encoder = _CanonicalAtomInsertEncoder(vocab, self._atom_offsets)
        else:
            self._canonical_encoder = None

    def get_insert_atom_index(self, vocab_idx, vocab_atom_idx):
        """ Gets the index corresponding to the given atom insertion action.

        Parameters
        ----------
        vocab_idx: the index of the vocab element on which to insert.
        vocab_atom_idx: the index of the atom in the vocab element on which to insert.
        """
        if self.canonical:
            return self._canonical_encoder.get_insert_atom_index(vocab_idx, vocab_atom_idx)
        else:
            return self._atom_offsets[vocab_idx] + vocab_atom_idx

    def get_num_atom_insert_locations(self):
        if self.canonical:
            return self._canonical_encoder.num_insert_atom_locations_canonical
        else:
            return self.num_insert_atom_locations

    def get_insert_bond_index(self, vocab_idx, vocab_bond_idx):
        return self._bond_offsets[vocab_idx] + vocab_bond_idx

    def get_insert_bond_location(self, idx):
        vocab_idx = np.maximum(np.searchsorted(self._bond_offsets, idx, side='right') - 1, 0)
        vocab_bond_idx = idx - self._bond_offsets[vocab_idx]
        return vocab_idx, vocab_bond_idx

    def get_insert_atom_location(self, idx):
        if self.canonical:
            return self._canonical_encoder.get_insert_atom_location(idx)
        else:
            vocab_idx = np.maximum(np.searchsorted(self._atom_offsets, idx, side='right') - 1, 0)
            vocab_atom_idx = idx - self._atom_offsets[vocab_idx]

        return vocab_idx, vocab_atom_idx


def compute_action_lengths(mol, vocab_encoder):
    """ Compute the lengths assigned to each action space on the given molecule. """
    ring_bonds = [b.GetIdx() for b in mol.GetBonds() if b.IsInRing()]
    num_ring_bonds = len(ring_bonds)
    num_leaves = len(chemutils.get_leaves(mol))

    lengths = np.array([
        1, num_leaves,
        mol.GetNumAtoms() * vocab_encoder.get_num_atom_insert_locations(),
        2 * num_ring_bonds * vocab_encoder.num_insert_bond_locations], dtype=np.int32)

    return lengths


def compute_action_lengths_split(mol, vocab_encoder, mode):
    if mode == 'insert':
        ring_bonds = [b.GetIdx() for b in mol.GetBonds() if b.IsInRing()]
        num_ring_bonds = len(ring_bonds)

        return np.array([
            1,
            mol.GetNumAtoms() * vocab_encoder.get_num_atom_insert_locations(),
            2 * num_ring_bonds * vocab_encoder.num_insert_bond_locations])
    elif mode == 'delete':
        num_leaves = len(chemutils.get_leaves(mol))

        return np.array([1, num_leaves])


def action_to_integer(act, num_leaves, num_atoms, ring_bond_idx, vocab_encoder: VocabInsertEncoder):
    """ Encodes the given action as an integer for input to the model.

    Parameters
    ----------
    act: the action to encode.
    num_leaves: the number of leaves in the molecule.
    num_atoms: the number of atoms in the molecule.
    num_ring_bonds: the number of bonds in rings in the molecule.
    ring_bond_idx: array of indices of bonds in rings in the molecule.
    vocab_encoder: encoder to use

    Returns
    -------
    result: an integer representing the action
    offsets: auxiliary information from the molecule used to decode the action.
    """
    num_ring_bonds = len(ring_bond_idx)

    lengths = np.array([
        1, num_leaves,
        num_atoms * vocab_encoder.get_num_atom_insert_locations(),
        2 * num_ring_bonds * vocab_encoder.num_insert_bond_locations], dtype=np.int32)

    offsets = np.zeros_like(lengths)
    np.cumsum(lengths[:-1], out=offsets[1:])

    if isinstance(act, action.Stop):
        result = offsets[0]
    elif isinstance(act, action.Delete):
        result = offsets[1] + act.leaf_idx
    elif isinstance(act, action.InsertAtomFusion):
        stride = vocab_encoder.get_num_atom_insert_locations()
        insert_index = vocab_encoder.get_insert_atom_index(act.vocab_idx, act.vocab_atom_idx)
        result = offsets[2] + act.atom_idx * stride + insert_index
    elif isinstance(act, action.InsertBondFusion):
        stride = vocab_encoder.num_insert_bond_locations
        insert_index = vocab_encoder.get_insert_bond_index(act.vocab_idx, act.vocab_bond_idx)
        location_index = np.searchsorted(ring_bond_idx, act.bond_idx)
        location_index = 2 * location_index
        result = offsets[3] + location_index * stride + insert_index
    else:
        raise ValueError("Unknown action type.")

    return result, offsets, lengths


def _action_to_integer_split_insert(act, num_atoms, ring_bond_idx, encoder):
    num_ring_bonds = len(ring_bond_idx)

    lengths = np.array([
        1, num_atoms * encoder.get_num_atom_insert_locations(),
        2 * num_ring_bonds * encoder.num_insert_bond_locations], dtype=np.int32)

    offsets = np.zeros_like(lengths)
    np.cumsum(lengths[:-1], out=offsets[1:])

    if isinstance(act, action.Stop):
        result = offsets[0]
    elif isinstance(act, action.InsertAtomFusion):
        stride = encoder.get_num_atom_insert_locations()
        insert_index = encoder.get_insert_atom_index(act.vocab_idx, act.vocab_atom_idx)
        result = offsets[1] + act.atom_idx * stride + insert_index
    elif isinstance(act, action.InsertBondFusion):
        stride = encoder.num_insert_bond_locations
        insert_index = encoder.get_insert_bond_index(act.vocab_idx, act.vocab_bond_idx)
        location_index = np.searchsorted(ring_bond_idx, act.bond_idx)
        location_index = 2 * location_index
        result = offsets[2] + location_index * stride + insert_index
    else:
        raise ValueError("Action with invalid type for insert mode. Action was {0}".format(act))

    return result, offsets, lengths


def _action_to_integer_split_delete(act, num_leaves, encoder):
    lengths = np.array([1, num_leaves], dtype=np.int32)
    offsets = np.array([0, 1], dtype=np.int32)

    if isinstance(act, action.Switch):
        result = offsets[0]
    elif isinstance(act, action.Delete):
        result = offsets[1] + act.leaf_idx
    else:
        raise ValueError("Action with invalid type for delete mode. Action was {0}".format(act))

    return result, offsets, lengths


def action_to_integer_split(act, num_leaves, num_atoms, ring_bond_idx, vocab_encoder, mode=None):
    """ Converts an action to its integer representation with a split mode.

    This method uses a split representation, where there are two overlapping representations
    for insert mode vs. delete mode operation.
    """
    if mode == "insert" or isinstance(act, (action.Stop, action.InsertAtomFusion, action.InsertBondFusion)):
        return _action_to_integer_split_insert(act, num_atoms, ring_bond_idx, vocab_encoder)
    elif mode == "delete" or isinstance(act, (action.Switch, action.Delete)):
        return _action_to_integer_split_delete(act, num_leaves, vocab_encoder)
    else:
        raise ValueError("Invalid mode, must be insert or delete.")


def integer_to_action(value, lengths, encoder):
    """ Decodes an integer value to an action representation.

    Parameter
    ---------
    value: The integer representing the action, as created by `action_to_integer`.
    lengths: An array which contains the number of actions in each action space. Provided by `action_to_integer`.
    encoder: A vocab encoder to use to encode / decode vocabulary action indices.

    Returns
    -------
    action: an instance of `action.Action` representing the action to be executed.
    """
    if value == 0:
        return action.Stop()

    # Try to decode Delete action
    value -= lengths[0]

    if value < lengths[1]:
        return action.Delete(value)

    value -= lengths[1]

    # Try to decode atom insert action
    if value < lengths[2]:
        location_idx, vocab_idx = divmod(value, encoder.get_num_atom_insert_locations())
        vocab_idx, vocab_atom_idx = encoder.get_insert_atom_location(vocab_idx)

        return action.InsertAtomFusion(location_idx, vocab_idx, vocab_atom_idx)

    value -= lengths[2]

    # Try to decode bond fusion action
    if value < lengths[3]:
        location_idx, vocab_idx = divmod(value, encoder.num_insert_bond_locations)
        vocab_idx, vocab_bond_idx = encoder.get_insert_bond_location(vocab_idx)

        # TODO: Fix-up bond in order.
        location_idx, bond_in_order = divmod(location_idx, 2)

        return action.InsertBondFusion(location_idx, vocab_idx, vocab_bond_idx, bond_in_order)


def _integer_to_action_split_delete(value, lengths):
    if value == 0:
        return action.Switch()

    value -= lengths[0]

    if value < lengths[1]:
        return action.Delete(value)

    raise ValueError("value is incompatible with given lengths specification.")


def _integer_to_action_split_insert(value, lengths, encoder):
    if value == 0:
        return action.Stop()

    value -= lengths[0]

    if value < lengths[1]:
        location_idx, vocab_idx = divmod(value, encoder.get_num_atom_insert_locations())
        vocab_idx, vocab_atom_idx = encoder.get_insert_atom_location(vocab_idx)

        return action.InsertAtomFusion(location_idx, vocab_idx, vocab_atom_idx)

    value -= lengths[1]

    if value < lengths[2]:
        location_idx, vocab_idx = divmod(value, encoder.num_insert_bond_locations)
        vocab_idx, vocab_bond_idx = encoder.get_insert_bond_location(vocab_idx)

        location_idx, bond_in_order = divmod(location_idx, 2)

        return action.InsertBondFusion(location_idx, vocab_idx, vocab_bond_idx, bond_in_order)

    raise ValueError("value is incompatible with given lengths specification.")


def integer_to_action_split(value, lengths, encoder, mode):
    """ Decodes an integer value to an action representation.

    This function is used to decode integer encodings for split
    models.
    """
    if mode == 'insert':
        return _integer_to_action_split_insert(value, lengths, encoder)
    elif mode == 'delete':
        return _integer_to_action_split_delete(value, lengths)
    else:
        raise ValueError("Invalid mode. Must be insert or delete.")


def integer_to_insert_atom_location(idx, offsets, encoder):
    return ((idx - offsets[..., 2]) // encoder.get_num_atom_insert_locations()).astype(np.int32)


def integer_to_insert_atom_vocab(idx, offsets, encoder):
    vocab_idx, _ = encoder.get_insert_atom_location((idx - offsets[..., 2]) % encoder.get_num_atom_insert_locations())
    return vocab_idx
