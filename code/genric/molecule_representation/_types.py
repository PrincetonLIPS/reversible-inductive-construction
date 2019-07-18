import typing

FloatSeq = typing.Sequence[float]
IntSeq = typing.Sequence[int]


class AtomInfo(typing.NamedTuple):
    atom_feature: FloatSeq
    atom_incidence: IntSeq
    atom_scope: IntSeq


class BondInfo(typing.NamedTuple):
    bond_feature: FloatSeq
    bond_incidence: IntSeq
    bond_scope: IntSeq


class ScopedTuple(typing.NamedTuple):
    feature: FloatSeq
    scope: IntSeq


class LeafInfo(typing.NamedTuple):
    leaf_atom_info: ScopedTuple
    leaf_ring_info: ScopedTuple

    @staticmethod
    def from_sequence(seq):
        return LeafInfo(ScopedTuple(*seq[0]), ScopedTuple(*seq[1]))


class RingInfo(typing.NamedTuple):
    ring_bond_idx: IntSeq
    ring_bond_order: IntSeq
    ring_scope: IntSeq

    @staticmethod
    def from_sequence(seq):
        if seq is None:
            return None

        return RingInfo(*seq)


class GraphInfo(typing.NamedTuple):
    atom_info: AtomInfo
    bond_info: BondInfo
    leaf_info: LeafInfo
    ring_info: RingInfo

    @staticmethod
    def from_sequence(seq):
        it = iter(seq)

        atom_info = AtomInfo(*next(it))
        bond_info = BondInfo(*next(it))
        leaf_info = LeafInfo.from_sequence(next(it, None))
        ring_info = RingInfo.from_sequence(next(it, None))

        return GraphInfo(atom_info, bond_info, leaf_info, ring_info)
