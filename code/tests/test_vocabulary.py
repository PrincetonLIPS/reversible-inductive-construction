from genric import vocabulary, molecule_edit as me


test_smiles = [
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


def test_vocabulary_legal_at_atom():
    vocab = vocabulary.Vocabulary()
    test_mol = [me.get_mol(s) for s in test_smiles]

    for mol in test_mol:
        for atom in mol.GetAtoms():
            assert vocab.legal_at_atom(atom) == me.legal_at_atom(mol, atom, vocab.vocab)


def test_vocabulary_legal_at_bond():
    vocab = vocabulary.Vocabulary()
    test_mol = [me.get_mol(s) for s in test_smiles]

    for mol in test_mol:
        for bond in mol.GetBonds():
            if not bond.IsInRing():
                continue

            assert vocab.legal_at_bond(bond) == me.legal_at_bond(mol, bond, vocab.vocab)
