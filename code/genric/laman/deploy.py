import torch
import numpy as np
import pickle
import networkx as nx
from tqdm import tqdm, trange

from . import laman_edit as edit, representation, action, joint_network, _utils
from ._data import LamanSamplerConfig
from ..model import classification


def _decode_action_try(a, graph):
    return representation.decode_action(a, graph)


def get_model_output(model, graph):
    graph_rep = representation.combine_graph_reps([representation.graph_to_rep(graph)])
    graph_rep = _utils.cast_numpy_rec(graph_rep)
    logits_and_scopes = model(graph_rep)

    _, sampled_actions = classification.multi_classification_prediction(logits_and_scopes, num_samples=5)
    return [_decode_action_try(a, graph) for a in sampled_actions[0]]


def run_single_transition(model, graph, config: LamanSamplerConfig, rng=None):
    if rng is None:
        rng = np.random

    graph = edit.apply_random_corruptions(graph, config.expected_corruption_steps, rng)
    graph_previous = graph

    graph_seq = [graph]
    act_seq = []

    for _ in range(config.max_denoising_steps):
        acts = get_model_output(model, graph)
        for act in acts:
            if act is None:
                tqdm.write("Skipping invalid None action")
                continue

            if isinstance(act, action.Stop):
                break

            try:
                graph = edit.compute_action(graph, act)
                break
            except ValueError:
                tqdm.write("Failed to apply action.")
        else:
            raise ValueError("Repeatedly failed to apply action.")

        act_seq.append(act)
        graph_seq.append(graph)

        if isinstance(act, action.Stop):
            break
        if config.use_revisit and any(nx.is_isomorphic(graph, g) for g in graph_seq[:-1]):
            break
    else:
        tqdm.write("Stop not sampled after 20 steps")

    return graph_seq, act_seq


def run_chain(model, init_graph, config: LamanSamplerConfig):
    torch.autograd.set_grad_enabled(False)
    rng = np.random.RandomState(seed=42)

    graph = init_graph

    samples = []

    for t in trange(config.num_steps):
        graph_seq, act_seq = run_single_transition(model, graph, rng=rng, config=config)
        graph = graph_seq[-1]
        samples.append(graph)

    return samples


def get_initial_graph(initial_data_path=None):
    if initial_data_path is None:
        initial_data_path = '../data/laman/low_decomp_dataset_sample.pkl'

    with open(initial_data_path, 'rb') as f:
        data = pickle.load(f)
        return np.random.choice(data)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--use-revisit', action='store_true')
    parser.add_argument('--num-transitions', type=int, default=1000)
    parser.add_argument('--initial-data-path')

    args = parser.parse_args()

    state_dict = torch.load(args.model_path, map_location='cpu')
    config = joint_network.JointClassificationNetworkConfig(
        5, message_size=256)
    model = joint_network.JointClassificationNetwork(config)

    model.load_state_dict(torch.load(
        args.model_path, map_location='cpu'))

    model = model.eval()

    init_graph = get_initial_graph(args.initial_data_path)
    sampler_config = LamanSamplerConfig(
        5, args.use_revisit, args.num_transitions)
    result = run_chain(model, init_graph, sampler_config)

    with open(args.output_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
