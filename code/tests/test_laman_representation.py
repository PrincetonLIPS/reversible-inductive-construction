import pytest
from induc_gen.laman import representation
from induc_gen.laman.representation import _representation
from functools import lru_cache


@lru_cache()
def _get_data():
    import pickle
    import gzip
    return pickle.load(open('../data/laman/demo_dataset.obj', 'rb'))


def test_edge_incidence_list():
    graph = _get_data()[0]
    result = _representation.get_edge_incidence_list(graph)


def test_single_representation():
    graph = _get_data()[0]
    result = representation.graph_to_rep(graph)

    assert result['vertex_feature'].shape[0] == graph.number_of_nodes()


def test_aggregation():
    graphs = _get_data()[:10]

    reps = list(map(representation.graph_to_rep, graphs))
    result = representation.combine_graph_reps(reps)

    assert result.vertex.scope.shape[0] == 10
