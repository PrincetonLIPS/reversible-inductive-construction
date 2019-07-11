import torch
import numpy as np
from ..molecule_models import vae_network, action_representation as ar, _train_utils
from .. import corruption_dataset, molecule_representation as mr


def embed_dataset(loader, model: vae_network.JointVaeNetwork, progress='tqdm'):
    means = []
    log_variances = []

    if progress == 'tqdm':
        import tqdm
        loader = tqdm.tqdm(loader)

    for batch in loader:
        batch = _train_utils.load_cuda_async(batch)

        with torch.autograd.no_grad():
            mean, log_variance = model.recognition(batch['graph'])

        means.append(mean.cpu().numpy())
        log_variances.append(log_variance.cpu().numpy())

    means = np.concatenate(means, axis=0)
    log_variances = np.concatenate(log_variances, axis=0)
    return means, log_variances


def make_model(action_encoder=None):
    if action_encoder is None:
        action_encoder = ar.VocabInsertEncoder(canonical=True)

    config_recog = vae_network.MoleculeRecognitionNetworkConfig(
        hidden_size=384,
        embedding_size=384,
        depth=5)

    config = vae_network.JointClassificationNetworkConfiguration(
        action_encoder.get_num_atom_insert_locations(),
        action_encoder.num_insert_bond_locations,
        hidden_size=384,
        depth=5)

    model = vae_network.JointVaeNetwork(config_recog, config)
    return model


def _dataset_transform(mol, _):
    return {
        'graph': mr.mol2graph_single(mol, include_leaves=True, include_rings=True, normalization='sqrt')
    }


def embed_dataset_files(model_path, data_path, output_path):
    model = make_model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.cuda()
    model.eval()

    dataset = corruption_dataset.BaseDataset(data_path, transform=_dataset_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, collate_fn=_train_utils.collate, num_workers=4,
        pin_memory=True)

    means, log_variances = embed_dataset(data_loader, model)

    np.savez(output_path, means=means, log_variances=log_variances)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path')
    parser.add_argument('--data-path')
    parser.add_argument('--output-path')

    args = parser.parse_args()

    embed_dataset_files(args.model_path, args.data_path, args.output_path)


if __name__ == '__main__':
    main()
