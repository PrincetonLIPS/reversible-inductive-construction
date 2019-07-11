import typing


class StructureRep(typing.NamedTuple):
    feature: typing.Any
    incidence: typing.Any
    scope: typing.Any

    @staticmethod
    def from_sequence(seq):
        return StructureRep(*seq)

    def apply(self, fn):
        return StructureRep._make(map(fn, self))


class ScopedIndex(typing.NamedTuple):
    index: typing.Any
    scope: typing.Any

    @staticmethod
    def from_sequence(seq):
        return ScopedIndex(*seq)

    def apply(self, fn):
        return ScopedIndex._make(map(fn, self))


class LamanActionScopes(typing.NamedTuple):
    h1: typing.Any
    h2: typing.Any
    rev_h1: typing.Any
    rev_h2: typing.Any

    @staticmethod
    def from_sequence(seq):
        return LamanActionScopes(*seq)

    def apply(self, fn):
        return LamanActionScopes._make(map(fn, self))


class LamanRep(typing.NamedTuple):
    """ Representation of a batch of laman graphs.

    This class holds the encoded properties of a batch of laman graphs.
    Topological properties (vertex and edge incidence, and vertex and edge features)
    are held in the `vertex` and `edge` field. The `reverse_h1` and `reverse_h2` fields
    contain index information indicating the location of valid moves of the specified type.

    """
    vertex: StructureRep
    edge: StructureRep
    reverse_h1: ScopedIndex
    reverse_h2: ScopedIndex
    action_scopes: LamanActionScopes

    @staticmethod
    def from_sequence(seq):
        return LamanRep(
            StructureRep.from_sequence(seq[0]),
            StructureRep.from_sequence(seq[1]),
            ScopedIndex.from_sequence(seq[2]),
            ScopedIndex.from_sequence(seq[3]),
            LamanActionScopes.from_sequence(seq[4]))

    def apply(self, fn):
        return LamanRep(
            self.vertex.apply(fn),
            self.edge.apply(fn),
            self.reverse_h1.apply(fn),
            self.reverse_h2.apply(fn),
            self.action_scopes.apply(fn))
