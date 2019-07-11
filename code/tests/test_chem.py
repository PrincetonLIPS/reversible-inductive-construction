from induc_gen import Chem
from rdkit import Chem as RDChem
import pytest


def test_smiles_roundtrip():
    smiles = 'CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1'

    mol = Chem.MolFromSmiles(smiles)
    mol_rd = RDChem.MolFromSmiles(smiles)

    assert mol.GetNumAtoms() == mol_rd.GetNumAtoms()


@pytest.mark.xfail(reason="stereo chemistry bugs")
def test_smiles_bond_properties():
    smiles = 'CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1'

    mol = Chem.MolFromSmiles(smiles)
    mol_rd = RDChem.MolFromSmiles(smiles)

    bond = mol.GetBondWithIdx(11)
    bond_rd = mol_rd.GetBondWithIdx(11)

    assert int(bond.GetStereo()) == int(bond_rd.GetStereo())
    assert bond.IsInRing() == bond_rd.IsInRing()

    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    bond_types_rd = [RDChem.BondType.SINGLE, RDChem.BondType.DOUBLE, RDChem.BondType.TRIPLE, RDChem.BondType.AROMATIC]

    assert bond_types.index(bond.GetBondType()) == bond_types_rd.index(bond_rd.GetBondType())


@pytest.mark.parametrize("atom_idx", [1, 4])
def test_smiles_atom_features(atom_idx):
    smiles = 'COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1'

    mol = Chem.MolFromSmiles(smiles)
    mol_rd = RDChem.MolFromSmiles(smiles)

    atom = mol.GetAtomWithIdx(atom_idx)
    atom_rd = mol_rd.GetAtomWithIdx(atom_idx)

    assert atom.GetSymbol() == atom_rd.GetSymbol()
    assert atom.GetDegree() == atom_rd.GetDegree()
    assert atom.GetFormalCharge() == atom_rd.GetFormalCharge()
    assert int(atom.GetChiralTag()) == int(atom_rd.GetChiralTag())
    assert atom.GetIsAromatic() == atom_rd.GetIsAromatic()


def test_smiles_pickle():
    import pickle
    smiles = 'CCCCCCC1=NN2C(=N)/C(=C\\c3cc(C)n(-c4ccc(C)cc4C)c3C)C(=O)N=C2S1'

    mol = Chem.MolFromSmiles(smiles)

    mol2 = pickle.loads(pickle.dumps(mol, protocol=pickle.HIGHEST_PROTOCOL))

    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(mol2)


def test_get_leaves():
    from induc_gen import molecule_edit as me
    smiles = 'O=C(Cn1nc(C(=O)[O-])c2ccccc2c1=O)Nc1ccc2c(c1)C(=O)c1ccccc1C2=O'
    mol = Chem.MolFromSmiles(smiles)

    result = me.get_leaves(mol)

    assert set(result) == set([0, 7, 8, 16, 25, 33, (9, 10, 11, 12, 13, 14), (26, 27, 28, 29, 30, 31)])
