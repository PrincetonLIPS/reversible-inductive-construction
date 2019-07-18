""" Chem utilities from rdkit exposed through custom wrappers.

This module will attempt to import our custom extension. If it fails,
it falls back to the `rdkit` implementation.

"""


try:
    from .genric_extensions import MolToSmiles, MolFromSmiles, MolFragmentToSmiles, Kekulize, CombineMols, GetSymmSSSR, SanitizeMol
    from .genric_extensions import Mol, RWMol, Atom, Bond, BondType

    from .genric_extensions import enable_log, disable_log
except ImportError:
    from rdkit.Chem import MolToSmiles, MolFromSmiles, MolFragmentToSmiles, Kekulize, CombineMols, GetSymmSSSR, SanitizeMol
    from rdkit.Chem import Mol, RWMol, Atom, Bond, BondType

    from rdkit.RDLogger import EnableLog as enable_log, DisableLog as disable_log
