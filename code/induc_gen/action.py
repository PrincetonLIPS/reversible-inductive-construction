""" This module contains classes which describes the inductive moves used by the corrupter and denoiser """

import abc
import enum
import numpy as np

from . import data_utils


class ActionType(enum.IntEnum):
    Stop = 0
    Delete = 1
    InsertBondFusion = 2
    InsertAtomFusion = 3
    Switch = 4


class Action(abc.ABC):
    @property
    @abc.abstractproperty
    def action_type(self):
        pass

    @abc.abstractmethod
    def to_array(self):
        x = np.zeros(6, dtype=np.int32)
        x[0] = int(self.action_type)
        return x

    @staticmethod
    @abc.abstractmethod
    def from_array(arr):
        action_type = arr[0]

        if action_type == ActionType.Stop:
            return Stop.from_array(arr)
        elif action_type == ActionType.Delete:
            return Delete.from_array(arr)
        elif action_type == ActionType.InsertBondFusion:
            return InsertBondFusion.from_array(arr)
        elif action_type == ActionType.InsertAtomFusion:
            return InsertAtomFusion.from_array(arr)
        elif action_type == ActionType.Switch:
            return Switch.from_array(arr)

        raise ValueError("Unknown action type")


class Continue(Action):
    pass


class Stop(Action):
    @property
    def action_type(self):
        return ActionType.Stop

    def __repr__(self):
        return "Stop()"

    @staticmethod
    def from_array(arr):
        return Stop()

    def to_array(self):
        return super(Stop, self).to_array()


class Switch(Action):
    @property
    def action_type(self):
        return ActionType.Switch

    def __repr__(self):
        return "Switch()"

    @staticmethod
    def from_array(arr):
        return Switch()

    def to_array(self):
        return super(Switch, self).to_array()


class Delete(Continue):
    """ Deletion action.

    The deletion is represented by the index of the leaf to be deleted
    in the junction tree of the molecule.

    """
    def __init__(self, leaf_idx):
        self.leaf_idx = int(leaf_idx)

    def __repr__(self):
        return "Delete(leaf={0})".format(self.leaf_idx)

    @property
    def action_type(self):
        return ActionType.Delete

    def to_array(self):
        x = super(Delete, self).to_array()
        x[1] = self.leaf_idx
        return x

    @staticmethod
    def from_array(arr):
        return Delete(arr[1])


class Insert(Continue):
    pass


class InsertBondFusion(Insert):
    def __init__(self, bond_idx, vocab_idx, vocab_bond_idx, bond_in_order, stereo=None):
        self.bond_idx = int(bond_idx)
        self.vocab_idx = int(vocab_idx)
        self.vocab_bond_idx = int(vocab_bond_idx)
        self.bond_in_order = bond_in_order
        self.stereo = stereo

    def __repr__(self):
        return "InsertBondFusion(bond={0}, vocab={1}, vocab_bond={2}, inorder={3}, stereo={4})".format(
            self.bond_idx, self.vocab_idx, self.vocab_bond_idx, self.bond_in_order, self.stereo)

    @property
    def action_type(self):
        return ActionType.InsertBondFusion

    @staticmethod
    def from_array(arr):
        return InsertBondFusion(*arr[1:5], arr[5] if arr[5] > 0 else None)

    def to_array(self):
        x = super(InsertBondFusion, self).to_array()
        x[1:5] = (self.bond_idx, self.vocab_idx, self.vocab_bond_idx, self.bond_in_order)
        x[5] = self.stereo if self.stereo is not None else -1
        return x


class InsertAtomFusion(Insert):
    def __init__(self, atom_idx, vocab_idx, vocab_atom_idx, stereo=None):
        self.atom_idx = int(atom_idx)
        self.vocab_idx = int(vocab_idx)
        self.vocab_atom_idx = int(vocab_atom_idx)
        self.stereo = stereo

    def __repr__(self):
        return "InsertAtomFusion(atom={0}, vocab={1}, vocab_atom={2}, stereo={3})".format(
            self.atom_idx, self.vocab_idx, self.vocab_atom_idx, self.stereo)

    @property
    def action_type(self):
        return ActionType.InsertAtomFusion

    @staticmethod
    def from_array(arr):
        return InsertAtomFusion(arr[1], arr[2], arr[3], arr[5] if arr[5] > 0 else None)

    def to_array(self):
        x = super(InsertAtomFusion, self).to_array()
        x[1:4] = (self.atom_idx, self.vocab_idx, self.vocab_atom_idx)
        x[5] = self.stereo if self.stereo is not None else -1
        return x
