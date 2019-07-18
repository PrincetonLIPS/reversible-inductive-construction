""" Implementation for vocabulary with precomputed structures for faster searches and checks. """

from . import molecule_edit as me
from . import data_utils
from .chemutils import get_mol
from typing import NamedTuple


class AtomTuple(NamedTuple):
    symbol: str
    formal_charge: int
    implicit_valence: int
    explicit_valence: int
    my_implicit_valence: int
    my_explicit_valence: int

    @staticmethod
    def from_atom(atom):
        return AtomTuple(
            atom.GetSymbol(),
            atom.GetFormalCharge(),
            atom.GetImplicitValence(),
            atom.GetExplicitValence(),
            me.my_implicit_valence(atom),
            me.my_explicit_valence(atom))


class BondTuple(NamedTuple):
    bond_type: int
    overlap: float
    atom1: AtomTuple
    atom2: AtomTuple

    @staticmethod
    def from_bond(bond):
        a1 = AtomTuple.from_atom(bond.GetBeginAtom())
        a2 = AtomTuple.from_atom(bond.GetEndAtom())

        return BondTuple(
            bond.GetBondType(),
            bond.GetBondTypeAsDouble(),
            min(a1, a2), max(a1, a2))


class Vocabulary:
    """ Vocabulary class with cached legality checks.

    This class stores the vocabulary, and additionally caches legality checks
    performed to enable large speed-ups when editing many molecules.
    """
    _average_eps = 3e-3

    def __init__(self, vocab=None):
        if vocab is None:
            vocab = data_utils.get_vocab()
        elif isinstance(vocab, str):
            vocab = data_utils.get_vocab(vocab_name=vocab)

        self.vocab = vocab
        self._cache_legal_atom = {}
        self._cache_legal_bond = {}

        self.cache_atom_hit_rate = 0.
        self.cache_bond_hit_rate = 0.

    def __iter__(self):
        return iter(self.vocab)

    def __getitem__(self, key):
        return self.vocab[key]

    def __len__(self):
        return len(self.vocab)

    def legal_at_atom(self, atom):
        self.cache_atom_hit_rate *= (1 - Vocabulary._average_eps)

        free_slots = atom.GetImplicitValence()
        if atom.GetExplicitValence() != me.my_explicit_valence(atom):
            free_slots = me.my_implicit_valence(atom)

        if free_slots == 0 and atom.GetSymbol() not in me.SPECIAL_ATOMS:
            self.cache_atom_hit_rate += Vocabulary._average_eps
            return []

        atom_tuple = AtomTuple.from_atom(atom)
        result = self._cache_legal_atom.get(atom_tuple, None)

        if result is not None:
            # return cached result if it exists
            self.cache_atom_hit_rate += Vocabulary._average_eps
            return result

        result = []

        for s, c in self.vocab:
            match_atoms = [a.GetIdx() for a in c.GetAtoms()
                           if me.atom_match(atom, a)]
            if len(match_atoms) > 0:
                result.append((c, match_atoms))

        self._cache_legal_atom[atom_tuple] = result
        return result

    def legal_at_bond(self, bond):
        self.cache_bond_hit_rate *= (1 - Vocabulary._average_eps)
        if not bond.IsInRing():
            raise ValueError("bond was not in ring.")

        bond_tuple = BondTuple.from_bond(bond)
        result = self._cache_legal_bond.get(bond_tuple, None)

        if result is not None:
            # return cached result if it exists
            self.cache_bond_hit_rate += Vocabulary._average_eps
            return result

        result = []
        for s, c in self.vocab:
            if c.GetNumAtoms() <= 2:
                # only rings can attach by bond
                continue

            match_bonds = [b.GetIdx() for b in c.GetBonds() if me.bond_match(bond, b)]
            if match_bonds:
                result.append((s, match_bonds))

        self._cache_legal_bond[bond_tuple] = result
        return result

    def get_vocab_index(self, mol):
        pass
