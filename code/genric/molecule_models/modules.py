import torch
import torch.nn

from ..torch_ext import repeat_interleave


def get_scopes_from_graph(graph):
    return {
        'delete_leaf_atom': graph.leaf_info.leaf_atom_info.scope,
        'delete_leaf_ring': graph.leaf_info.leaf_ring_info.scope,
        'insert_atom': graph.atom_info.atom_scope,
        'insert_bond': graph.ring_info.ring_scope
    }


class FullyConnectedUnit(torch.nn.Module):
    def __init__(self, num_inputs, selector, scope_fn=None):
        super(FullyConnectedUnit, self).__init__()
        self.fc = torch.nn.Linear(num_inputs, 1)
        self.selector = selector
        self.scope_fn = scope_fn

    def forward(self, embedding, graph):
        x = self.selector(embedding, graph)

        if x.shape[0] == 0:
            return x.new_zeros([1])

        return self.fc(x).squeeze(1)


class PartialLogitNetwork(torch.nn.Module):
    def __init__(self, message_size, embedding_size, output_size, hidden_size,
                 feature_fn, scope_fn):
        super(PartialLogitNetwork, self).__init__()

        self.classification_network = torch.nn.Sequential(
            torch.nn.Linear(message_size + embedding_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size))

        self.feature_fn = feature_fn
        self.scope_fn = scope_fn
        self.output_size = output_size

    def forward(self, embedding, graph):
        scope = self.scope_fn(graph)
        feature = self.feature_fn(embedding, graph)

        if feature.shape[0] == 0:
            return feature.new_zeros([self.output_size])

        global_embedding = repeat_interleave(
            embedding.embedding, scope, dim=0, out_length=feature.shape[0])
        feature_and_global = torch.cat((feature, global_embedding), dim=1)
        return self.classification_network(feature_and_global)


class SingleDeviceDistributedParallel(torch.nn.parallel.distributed.DistributedDataParallel):
    def __init__(self, module, device_id):
        super(SingleDeviceDistributedParallel, self).__init__(module, [device_id])

    def forward(self, *inputs, **kwargs):
        self._sync_params()
        output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled():
            self.reducer.prepare_for_backward([])

        return output
