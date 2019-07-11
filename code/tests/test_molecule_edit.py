from induc_gen import molecule_edit as me, chemutils, data_utils, action
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

from IPython import embed


def test_deletion():
    initial_smile = 'CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1'

    test_targets = [
        'CCC(NC(=O)c1c(C2CC2)ncs1)C(N)=O',
        'CCC(CN)NC(=O)c1c(C2CC2)ncs1',
        'CC(CN)NC(=O)c1c(C2CC2)ncs1',
        'CC(CN)NCc1c(C2CC2)ncs1']

    mol = me.get_mol(initial_smile)
    for i in range(4):
        target_mol = me.get_mol(test_targets[i])
        rng = np.random.RandomState(i)
        mol = me.delete_random_leaf(mol, rng)
        assert me.comp_mols(target_mol, mol)


# def test_insertion():
#     mol = me.get_mol('CCCC')
#     test_targets = [
#         'CCCC=[N-]',
#         'CCCC(=[N-])N',
#         '[N-]=C(N)CCC1C=NC=CCN1',
#         '[N-]=C(N)CCC1(N)C=NC=CCN1']

#     vocab = data_utils.get_vocab()

#     for i in range(4):
#         target_mol = me.get_mol(test_targets[i])
#         rng = np.random.RandomState(i)
#         mol = me.insert_random_node(mol, vocab, rng)
#         assert me.comp_mols(target_mol, mol)


def test_deterministic_deletion():
    init_smiles = ['CC1=CC(C2=C(C#N)N3N=C(C4CC4)SC3=N2)=CC=C1Cl']
    leaf_idxs = [3]
    test_targets = ['CC1=NN2C(C#N)=C(C3=CC=C(Cl)C(C)=C3)N=C2S1']

    for smiles, leaf_idx, target in zip(init_smiles, leaf_idxs, test_targets):
        mol = me.get_mol(smiles)
        new_mol = me.delete_random_leaf(mol, act=action.Delete(leaf_idx))
        target_mol = me.get_mol(target)
        assert me.comp_mols(target_mol, new_mol)


def test_bond_fusion_deterministic():
    smile = 'CC1=CC(C2=C(C#N)N3N=C(C4CC4)SC3=N2)=CC=C1Cl'
    vocab = data_utils.get_vocab()
    rng = np.random.RandomState(7)

    mol = me.get_mol(smile)
    bond = mol.GetBondWithIdx(11)
    result, act = me.insert_at_bond(mol, bond, vocab, rng=rng, return_action=True)

    result_deterministic = me.compute_insert_bond(mol, act, vocab)

    assert me.get_smiles(result) == me.get_smiles(result_deterministic)


def test_atom_insert_deterministic():
    smile = 'CC1=CC(C2=C(C#N)N3N=C(C4CC4)SC3=N2)=CC=C1Cl'
    vocab = data_utils.get_vocab()
    rng = np.random.RandomState(7)

    mol = me.get_mol(smile)
    atom = mol.GetAtomWithIdx(13)
    result, act = me.insert_at_atom(mol, atom, vocab, rng=rng, return_action=True)
    result_deterministic = me.compute_insert_atom(mol, act, vocab)

    assert me.get_smiles(result) == me.get_smiles(result_deterministic)


# def test_insertion_deterministic():
#     init_mol = me.get_mol('CCCC')
#     test_targets = [
#         'CCCC=[N-]',
#         'CCCC(=[N-])N',
#         '[N-]=C(N)CCC1C=NC=CCN1',
#         '[N-]=C(N)CCC1(N)C=NC=CCN1']

#     vocab = data_utils.get_vocab()
#     actions = []

#     mol = init_mol

#     for i in range(4):
#         rng = np.random.RandomState(i)
#         mol, act = me.insert_random_node(mol, vocab, rng, return_action=True)
#         actions.append(act)
#         assert me.comp_mols(me.get_mol(test_targets[i]), mol)

#     mol = init_mol
#     for i in range(4):
#         mol = me.insert_random_node(mol, vocab, act=actions[i])
#         assert me.comp_mols(me.get_mol(test_targets[i]), mol)


def test_insert_atom_inverse():
    smile = 'CC1=CC(C2=C(C#N)N3N=C(C4CC4)SC3=N2)=CC=C1Cl'
    vocab = data_utils.get_vocab()
    rng = np.random.RandomState(7)

    mol = me.get_mol(smile)
    atom = mol.GetAtomWithIdx(13)
    result, act, inverse = me.insert_at_atom(mol, atom, vocab, rng=rng,
                                             return_action=True, return_inverse=True)
    result = chemutils.sanitize(result)

    mol_roundtrip = me.compute_deletion(result, inverse)

    assert me.comp_mols(mol, mol_roundtrip)


def test_insert_bond_inverse():
    smile = 'CC1=CC(C2=C(C#N)N3N=C(C4CC4)SC3=N2)=CC=C1Cl'
    vocab = data_utils.get_vocab()
    rng = np.random.RandomState(7)

    mol = me.get_mol(smile)
    bond = mol.GetBondWithIdx(11)
    result, act, inverse = me.insert_at_bond(mol, bond, vocab, rng=rng,
                                             return_action=True, return_inverse=True)

    result = chemutils.sanitize(result)

    mol_roundtrip = me.compute_deletion(result, inverse)

    assert me.comp_mols(mol, mol_roundtrip)


def test_delete_inverse():
    smile = 'CC1=CC(C2=C(C#N)N3N=C(C4CC4)SC3=N2)=CC=C1Cl'
    vocab = data_utils.get_vocab()
    rng = np.random.RandomState(7)

    mol = me.get_mol(smile)
    result, act, inverse = me.delete_random_leaf(mol, rng=rng, return_action=True, return_inverse=True)

    mol_roundtrip = me.insert_random_node(result, vocab, act=inverse)

    assert me.comp_mols(mol, mol_roundtrip)


def test_delete_inverse_kekulize():
    smile = 'C1=CC(=CC=C1)C4=C(C#N)[N]3N=C(C2CC2)SC3=N4'

    vocab = data_utils.get_vocab()
    act = action.Delete(leaf_idx=1)

    mol = me.get_mol(smile)
    result, inverse = me.compute_deletion(mol, act, return_inverse=True)
    mol_roundtrip = me.compute_insert(result, inverse, vocab)

    assert me.comp_mols(mol, mol_roundtrip)


def test_delete_inverse_bond():
    smile = 'C1=CC(=CC=C1)C3=C(C#N)[N]2N=CSC2=N3'
    vocab = data_utils.get_vocab()
    act = action.Delete(leaf_idx=2)

    mol = me.get_mol(smile)
    result, inverse = me.compute_deletion(mol, act, return_inverse=True)
    mol_roundtrip = me.compute_insert(result, inverse, vocab)

    assert me.comp_mols(mol, mol_roundtrip)


def test_multi_delete_deterministic_and_inverses():
    vocab = data_utils.get_vocab()
    rng = np.random.RandomState(7)

    init_smiles = [
        'CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1',
        'COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1',
        'C=CCn1c(S[C@H](C)c2nc3sc(C)c(C)c3c(=O)[nH]2)nnc1C1CC1',
        'C[NH+](C/C=C/c1ccco1)CCC(F)(F)F',
        'COc1ccc(N2C(=O)C(=O)N(CN3CCC(c4nc5ccccc5s4)CC3)C2=O)cc1',
        'Cc1ccc([C@@H](C)[NH2+][C@H](C)C(=O)Nc2ccccc2F)cc1',
        'O=c1cc(C[NH2+]Cc2cccc(Cl)c2)nc(N2CCCC2)[nH]1',
        'O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O',
        'O=C(Nc1cccc(S(=O)(=O)N2CCCCC2)c1)c1cc(F)c(F)cc1Cl',
        'CC(C)Cc1nnc(NC(=O)C(=O)NCCC2CCCCC2)s1',
        'C[C@H](NC(=O)[C@@H](O)c1ccccc1)c1nnc2ccccn12',
        'O=S(=O)(Nc1cc(F)ccc1F)c1ccc(Cl)cc1F',
        'CSc1cc(C(=O)N2c3ccccc3NC(=O)C[C@@H]2C)ccn1',
        'CCCN1C(=O)c2[nH]nc(-c3cc(Cl)ccc3O)c2[C@H]1c1ccc(C)cc1',
        'CC[S@@](=O)[C@@H]1CCC[C@H](NC(=O)N(Cc2cccs2)C2CC2)C1',
        'C[C@@H](c1ccco1)[NH+](Cc1ncc(-c2ccccc2)o1)C1CC1',
        'COc1ccc(Cc2nnc(SCC(=O)N3CCC[C@@H](C)C3)o2)cc1',
        'O=C(/C=C/c1ccc2c(c1)OCO2)NC[C@@H]1C[NH+]2CCN1CC2',
        'COc1ccccc1/C=C/C=C(\\C#N)C(=O)Nc1ccc(C(=O)N(C)C)cc1',
        'Cc1cccc(NC(=S)N2CC[NH+](C)CC2)c1C']

    for idx, smiles in enumerate(init_smiles):
        mol = me.get_mol(smiles)

        result, act, inverse = me.delete_random_leaf(mol, rng=rng, return_action=True, return_inverse=True)
        result_deterministic = me.compute_deletion(mol, act=act)
        assert me.comp_mols(result, result_deterministic)

        mol_roundtrip = me.compute_insert(result, inverse, vocab)
        assert me.comp_mols(mol, mol_roundtrip)

        # 2nd deletion
        result2, act2, inverse2 = me.delete_random_leaf(result, rng=rng, return_action=True, return_inverse=True)
        result2_deterministic = me.compute_deletion(result, act=act2)
        assert me.comp_mols(result2, result2_deterministic)

        mol_roundtrip2 = me.compute_insert(result2, inverse2, vocab)
        assert me.comp_mols(result, mol_roundtrip2)


def test_multi_insert_deterministic_and_inverses():
    vocab = data_utils.get_vocab()
    rng = np.random.RandomState(7)

    init_smiles = [
        'CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1',
        'COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1',
        'C=CCn1c(S[C@H](C)c2nc3sc(C)c(C)c3c(=O)[nH]2)nnc1C1CC1',
        'C[NH+](C/C=C/c1ccco1)CCC(F)(F)F',
        'COc1ccc(N2C(=O)C(=O)N(CN3CCC(c4nc5ccccc5s4)CC3)C2=O)cc1',
        'Cc1ccc([C@@H](C)[NH2+][C@H](C)C(=O)Nc2ccccc2F)cc1',
        'O=c1cc(C[NH2+]Cc2cccc(Cl)c2)nc(N2CCCC2)[nH]1',
        'O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O',
        'O=C(Nc1cccc(S(=O)(=O)N2CCCCC2)c1)c1cc(F)c(F)cc1Cl',
        'CC(C)Cc1nnc(NC(=O)C(=O)NCCC2CCCCC2)s1',
        'C[C@H](NC(=O)[C@@H](O)c1ccccc1)c1nnc2ccccn12',
        'O=S(=O)(Nc1cc(F)ccc1F)c1ccc(Cl)cc1F',
        'CSc1cc(C(=O)N2c3ccccc3NC(=O)C[C@@H]2C)ccn1',
        'CCCN1C(=O)c2[nH]nc(-c3cc(Cl)ccc3O)c2[C@H]1c1ccc(C)cc1',
        'CC[S@@](=O)[C@@H]1CCC[C@H](NC(=O)N(Cc2cccs2)C2CC2)C1',
        'C[C@@H](c1ccco1)[NH+](Cc1ncc(-c2ccccc2)o1)C1CC1',
        'COc1ccc(Cc2nnc(SCC(=O)N3CCC[C@@H](C)C3)o2)cc1',
        'O=C(/C=C/c1ccc2c(c1)OCO2)NC[C@@H]1C[NH+]2CCN1CC2',
        'COc1ccccc1/C=C/C=C(\\C#N)C(=O)Nc1ccc(C(=O)N(C)C)cc1',
        'Cc1cccc(NC(=S)N2CC[NH+](C)CC2)c1C']

    for idx, smiles in enumerate(init_smiles):
        mol = me.get_mol(smiles)

        result, act, inverse = me.insert_random_node(mol, vocab, rng=rng, return_action=True, return_inverse=True)

        result_deterministic = me.compute_insert(mol, act, vocab)
        assert me.comp_mols(result, result_deterministic)

        mol_roundtrip = me.delete_random_leaf(result, act=inverse)
        assert me.comp_mols(mol, mol_roundtrip)

        # 2nd insertion
        result2, act2, inverse2 = me.insert_random_node(result, vocab, rng=rng, return_action=True, return_inverse=True)

        result2_deterministic = me.compute_insert(result, act2, vocab)
        assert me.comp_mols(result2, result2_deterministic)

        mol_roundtrip2 = me.delete_random_leaf(result2, act=inverse2)
        assert me.comp_mols(result, mol_roundtrip2)
