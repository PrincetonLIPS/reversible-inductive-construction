import argparse
import numpy as np
import time
from datetime import datetime
import os
from copy import deepcopy
import functools
import typing
import pickle

from .. import Chem
from .. import chemutils
import rdkit
from rdkit.Chem import Draw
import rdkit.RDLogger
import torch

from ..chemutils import get_mol, get_smiles, get_smiles_2D
from .. import vocabulary
from .. import data_utils
from .. import action, molecule_representation as mr, model as mo
from .. import molecule_edit as me
from ..molecule_models import joint_network
from ..molecule_models import action_representation as ar

from ..molecule_models._train_utils import replace_sparse_tensor, load_cuda_async, cast_numpy_to_torch
from ._data import SampleResult


SIZE_CAP = 25


class GibbsSampler:
    """Runs a Gibbs chain that alternates between a provided corruption distribution and reconstruction model.
    """
    def __init__(self, model, expected_corruption_steps=5, action_encoder=None, device=None):
        """
        Parameters
        ----------
        model: a Pytorch network that takes as input a molecular graph (x_tilde) and returns logits
            over all possible actions.`
        expected_corruption_steps: the expected length of the corruption sequence,
            used to determine the geometric distribution parameter.
        action_encoder: used to specify the vocab size and possible actions
            (of default type action_representation.VocabInsertEncoder)
        """
        self.model = model
        self.expected_corruption_steps = expected_corruption_steps
        self.vocab = vocabulary.Vocabulary()

        if action_encoder is None:
            action_encoder = ar.VocabInsertEncoder(canonical=True)

        self.action_encoder = action_encoder
        self.device = device

    def corrupter(self, mol, rng=np.random, return_seq=False):
        """Corrupts the input (of type rdkit Mol) via the default random insert & delete operations in molecule_edit.py.
        """

        seq = [mol]

        acts = []

        ori_mol = deepcopy(mol)
        number_of_steps = rng.geometric(1 / (1 + self.expected_corruption_steps)) - 1
        for _ in range(number_of_steps):
            if rng.uniform() < 0.5 and len(me.get_leaves(mol)) >= 2:
                mol, this_act = me.delete_random_leaf(mol, rng=rng, return_action=True)
            else:
                mol, this_act = me.insert_random_node(mol, self.vocab, rng=rng, return_action=True)

            seq.append(mol)
            acts.append(this_act)

        # Size cap
        if mol.GetNumAtoms() > SIZE_CAP:
            return self.corrupter(ori_mol, rng=rng, return_seq=return_seq)

        # Avoid splits (rare)
        if '.' in get_smiles_2D(mol):
            return self.corrupter(ori_mol, rng=rng, return_seq=return_seq)

        if not return_seq:
            return mol
        else:
            return mol, seq

    def _reconstruct_single_step(self, x_tilde):
        """ Runs a single step of the reconstruction process.

        Parameters
        ----------
        x_tilde: the input molecule to the reconstructor

        Returns
        -------
        A tuple of two elements.
        mol: Either the one-step action applied to the denoiser, if it was valid,
            or None if the sampled actions were invalid.
        act: The action that was sampled for the molecule.
        """
        x_tilde_graph = mr.combine_mol_graph(
            [mr.mol2graph_single(x_tilde, include_leaves=True, include_rings=True, normalization='sqrt')],
            return_namedtuple=True,
            cast_tensor=cast_numpy_to_torch)

        x_tilde_graph = load_cuda_async(x_tilde_graph, device=self.device)
        x_tilde_graph = mr.GraphInfo.from_sequence(x_tilde_graph)
        x_tilde_graph = replace_sparse_tensor(x_tilde_graph)

        logits_and_scopes = self.model(x_tilde_graph)
        predictions, cand_act_idxs = mo.classification.multi_classification_prediction(
            logits_and_scopes, predict=True, num_samples=5)

        for i, act_idx in enumerate(cand_act_idxs.cpu()[0]):
            act_idx = act_idx.item()
            # Get corresponding action object and try executing it
            lengths = ar.compute_action_lengths(x_tilde, self.action_encoder)
            act = ar.integer_to_action(act_idx, lengths, self.action_encoder)
            try:
                result = me.compute_action(x_tilde, act, vocab=self.vocab)
                break
            except ValueError:
                pass
        else:
            # No valid action sampled.
            result = None

        return result, act

    def reconstruct(self, actual_x_tilde, return_seq=False):
        """ Runs the reconstructor on the given molecule.

        Parameters
        ----------
        actual_x_tilde: the corrupted molecule
        return_seq: if True, returns the denoising sequence, otherwise,
            only return the last denoised value.
        """
        # Reconstruct
        x = None

        if return_seq:
            seq = [actual_x_tilde]

        x_tilde = deepcopy(actual_x_tilde)
        num_steps_taken = 0
        visited_smiles = {get_smiles_2D(actual_x_tilde): 0}
        is_revisit = False

        while True:
            x_tilde, act = self._reconstruct_single_step(x_tilde)

            if x_tilde is None:
                print('Did not sample valid action. Returning to previous mol.')
                break

            num_steps_taken += 1
            this_smiles = get_smiles_2D(x_tilde)
            is_revisit = False

            if this_smiles in visited_smiles:
                # print('Revisited on step %i' % visited_smiles[this_smiles])
                is_revisit = True
            else:
                visited_smiles[this_smiles] = num_steps_taken

            if is_revisit or isinstance(act, action.Stop):
                if x_tilde.GetNumAtoms() > SIZE_CAP:
                    # print('Mol too large. Returning to previous mol.')
                    pass
                elif '.' in get_smiles_2D(x_tilde):
                    # Avoid splits (rare). Leaving this as return to previous
                    # print('Broke mol. Returning to previous mol.')
                    pass
                else:
                    x = x_tilde
                break

        if not return_seq:
            return x, num_steps_taken, is_revisit
        else:
            return x, seq, num_steps_taken, is_revisit

    def _apply_corrupter(self, x, rng, check_substructure):
        while True:
            try:
                actual_x_tilde, seq = self.corrupter(x, rng=rng, return_seq=True)
                if check_substructure(actual_x_tilde):
                    break
            except ValueError:
                print('Corruption failed. Retrying corruption.')
                pass
        return actual_x_tilde, len(seq)

    def run_chain(self, init_smiles=None, num_transitions=1000, sample_freq=1, seed=None, substructure=None):
        """
        Parameters
        ----------
        init_smiles: the SMILES string with which to initialize the chain.
            If not provided, a random string from the ZINC validation set will be used.
        num_transitions: total number of chain transitions to run.
        sample_freq: frequency to print chain's state.
        seed: seed for numpy.random.
        """
        if not seed:
            seed = np.random.randint(2**31 - 1)
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_grad_enabled(False)

        # Initialize chain
        if substructure is not None:
            init_smiles = substructure
        elif not init_smiles:
            path = '../data/zinc/train.txt'
            with open(path, 'r') as f:
                data = [line.strip("\r\n ").split()[0] for line in f]
            init_smiles = rng.choice(data)

        init_smiles = get_smiles_2D(get_mol(init_smiles))
        x = get_mol(init_smiles)

        if substructure is not None:
            for atom in x.GetAtoms():
                atom.SetAtomMapNum(42)
            num_marked = x.GetNumAtoms()

            def check_sub_intact(mol):
                num_here = len([atom for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 42])
                if num_here == num_marked:
                    return True
                else:
                    return False
        else:
            def check_sub_intact(mol):
                return True

        # Run chain
        collected_x_tilde = []
        collected_x = [init_smiles]
        print('init_x: %s' % init_smiles)
        num_steps_reconstruct_chain = []
        num_steps_corrupt_chain = []
        revisit_chain = []
        transition_attempts_chain = []

        for t in range(num_transitions):
            transition_attempts = 0

            while True:
                actual_x_tilde, num_steps_corrupt = self._apply_corrupter(x, rng, check_sub_intact)

                # Reconstruct
                for _ in range(10):
                    # Attempt 10 possible reconstruction transitions from the given corruption.
                    transition_attempts += 1
                    potential_x, num_steps_reconstruct, revisit = self.reconstruct(actual_x_tilde)
                    if potential_x is not None and check_sub_intact(potential_x):
                        # If the proposed x is valid, record it, and move to next transition.
                        x = potential_x
                        break
                else:
                    # If none of the proposed reconstructions are valid after 10 steps,
                    # we retry the entire transition (including sampling the corruption).
                    continue

                # Break out of the loop to validate a single transition.
                break

            if (t + 1) % sample_freq == 0:
                # Print current state
                collected_x_tilde.append(get_smiles_2D(actual_x_tilde))
                collected_x.append(get_smiles_2D(x))
                num_steps_corrupt_chain.append(num_steps_corrupt)
                num_steps_reconstruct_chain.append(num_steps_reconstruct)
                revisit_chain.append(revisit)
                transition_attempts_chain.append(transition_attempts)
                print('Iteration: %i' % (t + 1))
                print('x_tilde: %s, x: %s' % (get_smiles_2D(actual_x_tilde), get_smiles_2D(x)))

        return SampleResult(
            seed, self.expected_corruption_steps, collected_x, collected_x_tilde,
            num_steps_corrupt_chain, num_steps_reconstruct_chain, transition_attempts_chain,
            revisit_chain, {})


def save_result(result: SampleResult, parameters=None):
    if parameters is None:
        parameters = {}

    path = parameters.get('output_path')

    if path is None:
        savedir = parameters.get('save_dir')
    else:
        savedir = os.path.dirname(path)

    if savedir is None:
        savedir = '../output/'

    if path is None:
        for i in range(1000):
            path = os.path.join(savedir, 'result_{0}.pkl'.format(i))
            if not os.path.exists(path):
                break
        else:
            raise ValueError("All paths exist.")

    os.makedirs(savedir, exist_ok=True)
    result = result._replace(meta={
        **result.meta,
        'model_path': parameters.get('model_path', "Model path unknown.")
    })
    print('Saving result in path {0}'.format(os.path.abspath(path)))
    with open(path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    # Set rdkit logging level
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--expected_corruption_steps', default=5, type=int)
    parser.add_argument('--num_transitions', default=1000, type=int)
    parser.add_argument('--sample_freq', default=1, type=int)
    parser.add_argument('--substructure', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError('Please specify a model path.')

    # Load model
    action_encoder = ar.VocabInsertEncoder(canonical=True)
    config = joint_network.JointClassificationNetworkConfiguration(
        action_encoder.get_num_atom_insert_locations(),
        action_encoder.num_insert_bond_locations,
        hidden_size=384)
    model = joint_network.JointClassificationNetwork(1, config)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    device = torch.device(args.device)
    model = model.to(device=device)
    model.eval()

    # Run chain
    sampler = GibbsSampler(model, args.expected_corruption_steps, action_encoder, device)
    result = sampler.run_chain(seed=args.seed, num_transitions=args.num_transitions, sample_freq=args.sample_freq, substructure=args.substructure)
    save_result(result, vars(args))


if __name__ == '__main__':
    main()
