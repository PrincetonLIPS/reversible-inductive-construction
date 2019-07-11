import numpy as np
from ._data import LamanRep, StructureRep, ScopedIndex, LamanActionScopes


def _combine_incidence_sparse(all_incidence, offsets_i, offsets_j):
    indices = []
    values = []
    total_shape = np.zeros(2, dtype=np.int32)

    for (index, value, shape), off_i, off_j in zip(all_incidence, offsets_i, offsets_j):
        indices.append(index + np.expand_dims(np.array([off_i, off_j], dtype=np.int32), -1))
        values.append(value)
        total_shape += np.array(shape, dtype=np.int32)

    return np.concatenate(indices, axis=1), np.concatenate(values), tuple(total_shape)


def _compute_scopes(lengths):
    offsets = np.concatenate([np.zeros((1,), dtype=np.int32), np.cumsum(lengths, 0)[:-1]], 0)
    return np.stack([offsets, lengths], 1)


def _offsets_from_counts(counts):
    return np.concatenate([np.zeros((1,), dtype=np.int32), np.cumsum(counts, 0)[:-1]], 0)


def _make_action_scopes(num_vertex, num_edge):
    pass


def combine_graph_reps(graph_reps) -> LamanRep:
    vertex_feature = np.concatenate([g['vertex_feature'] for g in graph_reps], axis=0)
    edge_feature = np.concatenate([g['edge_feature'] for g in graph_reps], axis=0)

    num_vertex = np.array([g['vertex_feature'].shape[0] for g in graph_reps])
    num_edge = np.array([g['edge_feature'].shape[0] for g in graph_reps])

    vertex_offsets = _offsets_from_counts(num_vertex)
    edge_offsets = _offsets_from_counts(num_edge)

    vertex_scopes = _compute_scopes(num_vertex)
    edge_scopes = _compute_scopes(num_edge)

    vertex_incidence = _combine_incidence_sparse(
        [g['vertex_incidence'] for g in graph_reps],
        vertex_offsets, edge_offsets)

    edge_incidence = _combine_incidence_sparse(
        [g['edge_incidence'] for g in graph_reps],
        edge_offsets, edge_offsets)

    reverse_h1_location = np.concatenate(
        [g['reverse_h1_location'] + offset for g, offset in zip(graph_reps, vertex_offsets)], axis=0)
    reverse_h1_scopes = _compute_scopes([len(g['reverse_h1_location']) for g in graph_reps])

    reverse_h2_location = np.concatenate(
        [g['reverse_h2_location'] + offset for g, offset in zip(graph_reps, vertex_offsets)], axis=0)
    reverse_h2_scopes = _compute_scopes([len(g['reverse_h2_location']) for g in graph_reps])

    h1_scope = _compute_scopes(num_vertex * (num_vertex - 1) // 2)
    h2_scope = _compute_scopes(num_vertex * (num_edge // 2))

    return LamanRep(
        StructureRep(vertex_feature, vertex_incidence, vertex_scopes),
        StructureRep(edge_feature, edge_incidence, edge_scopes),
        ScopedIndex(reverse_h1_location, reverse_h1_scopes),
        ScopedIndex(reverse_h2_location, reverse_h2_scopes),
        LamanActionScopes(h1_scope, h2_scope, reverse_h1_scopes, reverse_h2_scopes))
