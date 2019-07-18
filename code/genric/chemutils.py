# Some functions below adapted from https://github.com/wengong-jin/icml18-jtnn.

import logging

from . import Chem

MST_MAX_WEIGHT = 100
MAX_NCAND = 2000


try:
    from .genric_extensions.molecule_edit import copy_edit_mol
except ImportError:
    def copy_edit_mol(mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
        return new_mol


def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def get_mol(smiles, sanitize=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise ValueError("Invalid smiles string.")
    Chem.Kekulize(mol)
    return mol


def get_mol_2D(s):
    s = Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
    mol = Chem.MolFromSmiles(s)
    Chem.Kekulize(mol)
    return mol


def get_smiles_2D(mol):
    return get_smiles(mol, isomeric=False)


def get_smiles(mol, isomeric=True, kekule=True, **kwargs):
    return Chem.MolToSmiles(mol, kekuleSmiles=kekule, isomericSmiles=isomeric, **kwargs)


def decode_stereo(smiles2D):
    from Chem import EnumerateStereoisomers, StereoEnumerationOptions
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms()
               if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D


def sanitize(mol):
    if isinstance(mol, Chem.RWMol):
        mol = mol.GetMol()
    try:
        mol = copy_edit_mol(mol)
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        logging.debug("Failed. Returning None.\n")
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def clean_sulfur_hs(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'S':
            atom.SetNumExplicitHs(0)


def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol


def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()


# Bond type not considered because all aromatic (so SINGLE matches DOUBLE)
def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])


def atom_pair_equal_bond(atom_pair, bond):
    b1 = atom_pair
    b2 = (bond.GetBeginAtom(), bond.GetEndAtom())
    return ((atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])) or
            (atom_equal(b1[0], b2[1]) and atom_equal(b1[1], b2[0])))


def get_atom_leaves(mol):
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]


def get_ring_leaves(mol):
    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append(set([a1, a2]))

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]

    # Check for bridged compounds
    for i in range(len(rings) - 1):
        for j in range(i + 1, len(rings)):
            if len(rings[i].intersection(rings[j])) > 2:
                rings[i] = rings[i].union(rings[j])
                rings[j] = set()
    rings = [r for r in rings if len(r) > 0]

    clusters.extend(rings)

    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        # Check if only one neighbor or if multiple neighbors all intersect
        if len(inters) == 1 or (len(inters) > 1 and len(set.intersection(*inters)) > 0):
            leaf_rings.append(tuple(r))

    return leaf_rings


def get_leaves(mol):
    """ Computes leaves of the junction tree representation of the given molecule. """
    leaf_atoms = get_atom_leaves(mol)
    leaf_rings = get_ring_leaves(mol)

    return leaf_atoms + leaf_rings
