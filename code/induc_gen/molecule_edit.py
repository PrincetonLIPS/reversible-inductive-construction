from __future__ import print_function
import logging
import numpy as np

from . import Chem
from . import action, data_utils, chemutils
from .chemutils import get_mol, get_smiles, get_leaves

SPECIAL_ATOMS = ['S', 'P']


def comp_mols(mol_a, mol_b):
    def clean_mol(mol):
        return chemutils.sanitize(chemutils.copy_edit_mol(mol))
    mols = [clean_mol(mol_a), clean_mol(mol_b)]
    smiles = []
    for mol in mols:
        Chem.SanitizeMol(mol)
        smiles.append(get_smiles(mol, kekule=False, isomeric=False))
    return smiles[0] == smiles[1]


def my_explicit_valence(atom):
    count = 0
    for b in atom.GetBonds():
        count += b.GetBondTypeAsDouble()
    return count


def my_implicit_valence(atom):
    return atom.GetTotalValence() - my_explicit_valence(atom)


def atom_match(ctr_atom, nei_atom, overlap=0):
    basic_match = (ctr_atom.GetSymbol() == nei_atom.GetSymbol() and
                   ctr_atom.GetFormalCharge() == nei_atom.GetFormalCharge())

    if basic_match and ctr_atom.GetSymbol() in SPECIAL_ATOMS:
        return True

    if basic_match:
        if ctr_atom.GetImplicitValence() + overlap >= nei_atom.GetExplicitValence():
            return True
        elif my_implicit_valence(ctr_atom) + overlap >= my_explicit_valence(nei_atom):
            return True
    return False


def bond_match(ctr_bond, nei_bond):
    if ctr_bond.GetBondType() != nei_bond.GetBondType():
        return False
    c1, c2 = ctr_bond.GetBeginAtom(), ctr_bond.GetEndAtom()
    n1, n2 = nei_bond.GetBeginAtom(), nei_bond.GetEndAtom()
    overlap = int(ctr_bond.GetBondTypeAsDouble())
    return ((atom_match(c1, n1, overlap) and atom_match(c2, n2, overlap)) or
            (atom_match(c1, n2, overlap) and atom_match(c2, n1, overlap)))


def legal_at_atom(mol, atom, vocab):
    """ Computes the set of potential substructures that can be attached
    to the specified location.

    Parameters
    ----------
    mol: the molecule onto which to attach the substructure
    atom: the atom in the molecule to which to attach the substructure
    vocab: a list of (smile string, molecule) tuples representing the potential substructures.

    Returns
    -------
    valid_vocab: a list of tuples (molecule, list of indices), where each molecule
    is taken from the given `vocab`, and the corresponding list of indices represents
    the indices of the atoms in the substructure which can be attached.
    """
    try:
        # enable vocabulary caching of legality checking.
        return vocab.legal_at_atom(atom)
    except AttributeError:
        pass

    free_slots = atom.GetImplicitValence()  # Implicit valence is the number of free slots that atom can accomodate
    if atom.GetExplicitValence() != my_explicit_valence(atom):
        free_slots = my_implicit_valence(atom)

    if free_slots == 0 and atom.GetSymbol() not in SPECIAL_ATOMS:
        return []

    valid_vocab = []
    for s, c in vocab:
        match_atoms = [a.GetIdx() for a in c.GetAtoms() if atom_match(atom, a)]
        if len(match_atoms) > 0:
            valid_vocab.append((c, match_atoms))

    return valid_vocab


# This function is to attach a substructure to a particular bond in the molecule
def legal_at_bond(mol, bond, vocab):
    """ Cmoputes the set of potential substructures that can be attached to
    the specified bond.

    Parameters
    ----------
    mol: the molecule onto which to attach the substructure
    bond: the bond of the molecule where to attach the substructure
    vocab: a list of (smile string, molecule) tuples representing the potential substructures.

    Returns
    -------
    valid_vocab: a list of tuples (smile string, list of indices), where each smile
    string is taken from the given `vocab`, and the corresponding list of indices
    represents the indices of the bonds in the substructure which can be attached.
    """
    try:
        # enable vocabulary caching of legality checking.
        return vocab.legal_at_bond(bond)
    except AttributeError:
        pass

    valid_vocab = []
    assert bond.IsInRing()  # Only ring and ring has bond overlap

    for s, c in vocab:
        if c.GetNumAtoms() <= 2:
            continue  # Only rings can attach
        match_bonds = [b.GetIdx() for b in c.GetBonds() if bond_match(bond, b)]
        if len(match_bonds) > 0:
            valid_vocab.append((s, match_bonds))

    return valid_vocab


def find_vocab_index(smiles, vocab):
    """ Finds the index of the given smiles string in the vocabulary """
    for i, (s, m) in enumerate(vocab):
        if s == smiles:
            return i

    # Didn't find. Try relaxed matching.
    logging.debug("Vocab item not found. Attempting relaxed matching.")
    for i, (s, m) in enumerate(vocab):
        if comp_mols(m, get_mol(smiles)):
            return i

    logging.debug('Failed')
    raise ValueError("Smiles string not found in vocab.")


def get_insertion_target(target_mol, substruct_idxs, result, vocab=None):
    """ Computes the reverse insertion action.

    Parameters
    ----------
    target_mol: Input to the deletion procedure
    substruct_idxs: dictionary with two entries
        remaining: list of indices in `target_mol` corresponding to substructure part
                    of the deletion process but not deleted.
        deleted: list of indices in `target_mol` corresponding to subtructure actually removed.
    result: Output of the deletion procedure
    vocab: Vocabulary to use.
    """
    remaining_idxs = substruct_idxs['remaining']
    deleted_idxs = substruct_idxs['deleted']

    # Identify deleted substructure
    del_smiles = get_smiles(chemutils.get_clique_mol(target_mol, remaining_idxs + deleted_idxs))

    if vocab is None:
        vocab = data_utils.get_vocab()

    vocab_idx = find_vocab_index(del_smiles, vocab)

    if len(remaining_idxs) == 1:
        target_atom = target_mol.GetAtomWithIdx(remaining_idxs[0])
        # Handle the case when the inverse should be an atom fusion
        for atom in result.GetAtoms():
            # Skip atom if it doesn't match the remaining atom
            if not chemutils.atom_equal(target_atom, atom):
                continue
            legal = legal_at_atom(result, atom, [vocab[vocab_idx]])
            if not legal:
                continue
            (_, matched_atom_idxs), = legal

            for ma_idx in matched_atom_idxs:
                act = action.InsertAtomFusion(atom.GetIdx(), vocab_idx, ma_idx)
                try:
                    potential_mol = compute_insert_atom(result, act, vocab)
                except:
                    continue
                if comp_mols(target_mol, potential_mol):
                    return act
        raise ValueError("No inverse found for atom fusion.")

    if len(remaining_idxs) == 2:
        target_bond_atoms = [target_mol.GetAtomWithIdx(i) for i in remaining_idxs]
        # Handle the case when the inverse should be a bond fusion
        for bond in result.GetBonds():
            if not bond.IsInRing():
                continue
            # Skip bond if it doesn't match the target bond
            if not chemutils.atom_pair_equal_bond(target_bond_atoms, bond):
                continue
            legal = legal_at_bond(result, bond, [vocab[vocab_idx]])
            if not legal:
                continue
            (_, matched_bond_idxs), = legal

            for mb_idx in matched_bond_idxs:
                for bond_in_order in [True, False]:
                    act = action.InsertBondFusion(bond.GetIdx(), vocab_idx, mb_idx, bond_in_order)
                    try:
                        potential_mol = compute_insert_bond(result, act, vocab)
                    except:
                        continue
                    if comp_mols(target_mol, potential_mol):
                        return act
        raise ValueError("No inverse found for bond fusion.")
    raise ValueError("Remaining indices expected to be length 1 or 2")


def enumerate_deletion_actions(molecule, leaves=None):
    """ Enumerate all possible deletion actions on the given molecule.

    Parameters
    ----------
    molecule: the molecule for which to enumerate the actions.
    leaves: if not None, the tree structure correspoding to the molecule.
    """
    if leaves is None:
        leaves = get_leaves(molecule)
    return [action.Delete(leaf_idx) for leaf_idx, _ in enumerate(leaves)]


def compute_deletion(molecule, act: action.Delete, return_inverse=False, leaves=None, vocab=None):
    """ Computes the result of executing the given deletion action on the molecule.

    Parameters
    ----------
    molecule: the molecule on which to perform the action
    act: an instance of `action.Delete` representing the deletion to perform
    return_inverse: if True, also returns the action which corresponds to the inverse of `action`
    leaves: if not None, the list of leaves corresponding to the molecule. Used for performance.
    vocab: if not None, the vocabulary to be used to compute the inverse action.

    Returns
    -------
    A new molecule representing the result of the given deletion action.
    """
    result_mol = Chem.RWMol(molecule)

    if leaves is None:
        leaves = get_leaves(molecule)

    leaf_idx = act.leaf_idx
    leaf = leaves[leaf_idx]

    if not isinstance(leaf, tuple):
        # Deleting an atom
        result_mol.RemoveAtom(leaf)
        if return_inverse:
            remaining_idx = molecule.GetAtomWithIdx(leaf).GetBonds()[0].GetOtherAtomIdx(leaf)
            substruct_idxs = {'remaining': [remaining_idx], 'deleted': [leaf]}
    else:
        # Deleting a ring
        leaf = list(leaf)
        remainders = []
        for atom_idx in leaf:
            # Don't remove atoms attached to atoms not part of the ring
            nei_idxs = [nei.GetIdx() for nei in result_mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
            if not set(nei_idxs).issubset(leaf):
                remainders.append(result_mol.GetAtomWithIdx(atom_idx))

        deleted = leaf.copy()
        remaining_idxs = [r.GetIdx() for r in remainders]
        for r in remaining_idxs:
            deleted.remove(r)

        deleted.sort(reverse=True)  # delete atoms in reverse order
        for atom_idx in deleted:
            result_mol.RemoveAtom(atom_idx)

        for atom in remainders:
            remove_aroma = True
            for bond in atom.GetBonds():
                if bond.GetIsAromatic():
                    remove_aroma = False
            if remove_aroma:
                atom.SetIsAromatic(False)
        if return_inverse:
            substruct_idxs = {'remaining': remaining_idxs, 'deleted': deleted}

    result_mol = chemutils.sanitize(result_mol)

    if return_inverse:
        inverse_action = get_insertion_target(molecule, substruct_idxs, result_mol, vocab)
        return result_mol, inverse_action
    else:
        return result_mol


# This function deletes a random leaf from mol unless a specific leaf is specified by act (of type action.Delete)
def delete_random_leaf(mol, rng=np.random, act=None, return_action=False, return_inverse=False, vocab=None):
    leaves = get_leaves(mol)

    if act is None:
        act = rng.choice(enumerate_deletion_actions(mol, leaves))

    if return_inverse:
        result, inverse_action = compute_deletion(mol, act, return_inverse, leaves=leaves, vocab=vocab)
    else:
        result = compute_deletion(mol, act, return_inverse, leaves=leaves, vocab=vocab)

    result = [result]

    if return_action:
        result += [act]

    if return_inverse:
        result += [inverse_action]

    return result[0] if len(result) == 1 else result


# This function finds the leaf in the input molecule that should be the target for deletion
def get_deletion_target(target_mol, mol_with_insert):
    """ Computes the deletion action which should be applied to `mol_with_insert` to
    obtain `target_mol`.

    Parameters
    ----------
    target_mol: the molecule which should be obtained after deleting.
    mol_with_insert: the molecule from which to delete.
    """

    leaves = get_leaves(mol_with_insert)
    num_target_atoms = target_mol.GetNumAtoms()
    num_current_atoms = mol_with_insert.GetNumAtoms()

    # number of atoms which should be deleted.
    num_atoms_diff = num_current_atoms - num_target_atoms
    assert num_atoms_diff >= 1

    for leaf_idx, leaf in enumerate(leaves):
        leaf_is_ring = isinstance(leaf, tuple)

        if leaf_is_ring:
            if len(leaf) - num_atoms_diff not in (1, 2):
                # skip over incompatible ring leaves
                # deleting a ring leaf will remove all but 1 or 2 atoms from the leaf.
                continue
        else:
            if num_atoms_diff != 1:
                # Skip atom leaves if sizes differ by more than 1.
                continue

        act = action.Delete(leaf_idx)
        potential_mol = compute_deletion(mol_with_insert, act, leaves=leaves)

        if potential_mol.GetNumAtoms() != num_target_atoms:
            continue

        if comp_mols(target_mol, potential_mol):
            return act

    raise ValueError('Could not invert insertion. No matching deletion found.')


def compute_insert_atom(molecule, act: action.InsertAtomFusion, vocab, return_inverse=False):
    """ Computes the result of executing the specified atom insertion operation
    on the given molecule.

    Parameters
    ----------
    molecule: the molecule on which to insert.
    act: an action object describing the insertion operation to perform.
    vocab: the vocabulary of structures from which to select the inserted structure.
    return_inverse: if True, also computes and returns the inverse action.
    """
    atom = molecule.GetAtomWithIdx(act.atom_idx)
    inserted_structure = get_mol(vocab[act.vocab_idx][0])

    start_idx = molecule.GetNumAtoms()
    result_mol = Chem.RWMol(Chem.CombineMols(molecule, inserted_structure))
    vocab_atom_idx = act.vocab_atom_idx

    match_idx = start_idx + vocab_atom_idx
    transfer_bonds(result_mol, result_mol.GetAtomWithIdx(atom.GetIdx()), result_mol.GetAtomWithIdx(match_idx))
    result_mol.RemoveAtom(match_idx)

    result_mol = chemutils.sanitize(result_mol)
    if not result_mol:
        raise ValueError("Error in sanitizing the resulting molecule.")

    if return_inverse:
        inverse_action = get_deletion_target(molecule, result_mol)
        return result_mol, inverse_action
    else:
        return result_mol


def generate_random_atom_insert(molecule, atom, vocab, rng=np.random):
    """ Generate a random atom insertion action.

    Parameters
    ----------
    molecule: the molecule on which to perform insertion.
    atom: the atom of the molecule at which to attach.
    vocab: the vocabulary of structure to attach.
    rng: optional random number generator.

    Returns
    -------
    A random instance of `action.InsertAtomFusion`.
    """
    attach_ring = rng.uniform() < 0.5
    candidates = legal_at_atom(molecule, atom, vocab)
    candidates = [(mol, locations) for mol, locations in candidates
                  if mol.GetAtomWithIdx(0).IsInRing() == attach_ring]

    if not candidates:
        raise ValueError("No valid candidate insertion location.")

    while True:
        c, match_atoms = candidates[rng.choice(len(candidates))]  # match_atoms is idxs
        if c.GetNumAtoms() > 1:  # not inserting single atom node
            break
    vocab_idx = find_vocab_index(get_smiles(c), vocab)
    vocab_atom_idx = int(rng.choice(match_atoms))

    return action.InsertAtomFusion(atom.GetIdx(), vocab_idx, vocab_atom_idx)


def insert_at_atom(mol, atom, vocab, rng=np.random, act=None, return_action=False, return_inverse=False):
    """ Performs an atom fusion on the specified molecule unless a specific atom fusion
    is specified by `act`.

    Parameters
    ----------
    mol: the molecule on which to insert.
    atom: the atom of `mol` at which to insert.
    vocab: vocabulary of structures for insertion.
    rng: random state generator.
    act: if not None, executes the given action instead of generating one randomly.
    return_action: if True, also returns the executed action.
    return_inverse: if True, also returns the inverse of the executed action.
    """
    if not act:
        act = generate_random_atom_insert(mol, atom, vocab, rng)

    if return_inverse:
        result, inverse = compute_insert_atom(mol, act, vocab, return_inverse=return_inverse)
    else:
        result = compute_insert_atom(mol, act, vocab)

    result = [result]

    if return_action:
        result += [act]

    if return_inverse:
        result += [inverse]

    return result[0] if len(result) == 1 else result


def transfer_bonds(rw_mol, ctr_atom, nei_atom):
    for b in nei_atom.GetBonds():
        if nei_atom.GetIdx() == b.GetBeginAtomIdx():
            other_idx = b.GetEndAtomIdx()
        else:
            other_idx = b.GetBeginAtomIdx()
        rw_mol.AddBond(ctr_atom.GetIdx(), other_idx, b.GetBondType())


def generate_random_bond_insert(molecule, bond, vocab, rng=np.random):
    """ Generate a random bond insertion action.

    Parameters
    ----------
    molecule: the molecule on which to insert
    bond: the bond at which to insert
    vocab: a vocabulary of potential structures to insert
    rng: random number generator.

    Returns
    -------
    A random `action.InsertBondFusion`.

    Raises
    ------
    ValueError: If there are no valid insertions at the given bond.
    """
    cands = legal_at_bond(molecule, bond, vocab)
    if not cands:
        raise ValueError("No valid candidate insertion location.")
    c_smile, match_bonds = cands[rng.choice(len(cands))]
    vocab_bond_idx = int(rng.choice(match_bonds))

    inserted_structure = get_mol(c_smile)
    start_bond_idx = molecule.GetNumBonds()
    result_mol = Chem.RWMol(Chem.CombineMols(molecule, inserted_structure))
    match_bond_idx = start_bond_idx + vocab_bond_idx
    # Check how to align the matching bond (randomly choose if both work)
    overlap = int(bond.GetBondTypeAsDouble())
    match_bond = result_mol.GetBondWithIdx(match_bond_idx)
    a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()

    match_bond_atoms = [match_bond.GetBeginAtom(), match_bond.GetEndAtom()]

    def compute_bond_in_order(n1, n2):
        return (a1.GetIdx() - a2.GetIdx()) * (n1.GetIdx() - n2.GetIdx()) > 0

    potential_n1s = [
        n1 for n1 in match_bond_atoms
        if atom_match(a1, n1, overlap) and
        atom_match(a2, match_bond.GetOtherAtom(n1), overlap)]
    n1 = rng.choice(potential_n1s)
    n2 = match_bond.GetOtherAtom(n1)
    bond_in_order = compute_bond_in_order(n1, n2)

    vocab_idx = [i for i, s in enumerate(vocab) if s[0] == c_smile][0]

    return action.InsertBondFusion(bond.GetIdx(), vocab_idx, vocab_bond_idx, bond_in_order)


def compute_insert_bond(molecule, act: action.InsertBondFusion, vocab, return_inverse=False, yes=False):
    """ Computes the result of inserting a structure at a bond on the given molecule.

    Parameters
    ----------
    molecule: the molecule on which to insert.
    act: A `action.InsertBondFusion` describing the action to be executed.
    vocab: the vocabulary of substructures to attach.
    """
    c_smile = vocab[act.vocab_idx][0]
    bond = molecule.GetBondWithIdx(act.bond_idx)

    c = get_mol(c_smile)
    start_bond_idx = molecule.GetNumBonds()
    result_mol = Chem.RWMol(Chem.CombineMols(molecule, c))
    vocab_bond_idx = act.vocab_bond_idx

    match_bond_idx = start_bond_idx + vocab_bond_idx
    # Check how to align the matching bond (randomly choose if both work)
    overlap = int(bond.GetBondTypeAsDouble())
    match_bond = result_mol.GetBondWithIdx(match_bond_idx)
    a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()

    match_bond_atoms = [match_bond.GetBeginAtom(), match_bond.GetEndAtom()]

    def compute_bond_in_order(n1, n2):
        return (a1.GetIdx() - a2.GetIdx()) * (n1.GetIdx() - n2.GetIdx()) > 0

    n1, n2 = match_bond_atoms

    if act.bond_in_order != compute_bond_in_order(n1, n2):
        n1, n2 = n2, n1

    result_mol.RemoveBond(n1.GetIdx(), n2.GetIdx())
    transfer_bonds(result_mol, a1, n1)
    transfer_bonds(result_mol, a2, n2)
    result_mol.RemoveAtom(n1.GetIdx())
    result_mol.RemoveAtom(n2.GetIdx())

    result_mol = chemutils.sanitize(result_mol)
    if not result_mol:
        raise ValueError("Error in sanitizing the resulting molecule.")

    if return_inverse:
        inverse_action = get_deletion_target(molecule, result_mol)
        return result_mol, inverse_action
    else:
        return result_mol


def insert_at_bond(mol, bond, vocab, rng=np.random, act=None, return_action=False, return_inverse=False):
    """ Performs a bond fusion on the specified molecule, choosing randomly unless
    a specific fusion action is specified by `act`.
    """
    if not act:
        act = generate_random_bond_insert(mol, bond, vocab, rng)

    if return_inverse:
        result_mol, inverse_action = compute_insert_bond(mol, act, vocab, return_inverse=True)
    else:
        result_mol = compute_insert_bond(mol, act, vocab)

    result = [result_mol]

    if return_action:
        result += [act]

    if return_inverse:
        result += [inverse_action]

    return result[0] if len(result) == 1 else result


def compute_insert(molecule, act: action.Insert, vocab, return_inverse=False):
    """ Computes the result of executing the given insertion action.

    Parameters
    ----------
    molecule: the molecule on which to perform the insert.
    act: an action representing the insertion to be performed.
    vocab: list representing the vocabulary.
    return_inverse: if True, also computes the inverse action.
    """
    if isinstance(act, action.InsertAtomFusion):
        return compute_insert_atom(molecule, act, vocab, return_inverse=return_inverse)
    elif isinstance(act, action.InsertBondFusion):
        return compute_insert_bond(molecule, act, vocab, return_inverse=return_inverse)
    else:
        raise ValueError("Expected action to be either instance of InsertAtomFusion or InsertBondFusion")


def insert_random_node(mol, vocab, rng=np.random, act=None, return_action=False, return_inverse=False):
    """ Inserts a node from the vocab onto the molecule randomly, unless
    a specific insertion is specified by `act`.
    """
    if act:
        results = compute_insert(mol, act, vocab, return_inverse=return_inverse)
    if not act:
        def generate_random_insert():
            a = rng.choice(list(mol.GetAtoms()))
            if a.IsInRing() and rng.uniform() < 0.25:  # Bond attach w/ 25% prob.
                bond = rng.choice([b for b in a.GetBonds() if b.IsInRing()])
                return generate_random_bond_insert(mol, bond, vocab, rng=rng)
            else:
                return generate_random_atom_insert(mol, a, vocab, rng=rng)

        max_insert_attempts = 20

        for _ in range(max_insert_attempts):
            try:
                act = generate_random_insert()
                results = compute_insert(mol, act, vocab, return_inverse=return_inverse)
                break
            except ValueError as e:
                pass
        else:
            raise ValueError("Could not generate valid insertion after {0} attempts".format(max_insert_attempts))

    if return_inverse:
        result_mol, inverse = results
    else:
        result_mol = results

    result = [result_mol]
    if return_action:
        result += [act]
    if return_inverse:
        result += [inverse]

    return result[0] if len(result) == 1 else result


def compute_action(mol, act, vocab=None):
    if isinstance(act, action.Stop):
        return mol
    elif isinstance(act, action.Insert):
        return compute_insert(mol, act, vocab)
    else:
        return compute_deletion(mol, act)
