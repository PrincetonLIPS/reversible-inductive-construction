#include "module.h"

#include <algorithm>
#include <iterator>

#include <GraphMol/AtomIterators.h>
#include <GraphMol/BondIterators.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>

#include <vector>
#include <set>

namespace py = pybind11;

namespace {

std::shared_ptr<RDKit::RWMol> copy_edit_mol(RDKit::ROMol &mol) {
    auto result = std::make_shared<RDKit::RWMol>();

    for (auto it = mol.beginAtoms(), e = mol.endAtoms(); it != e; ++it) {
        auto atom = new RDKit::Atom((*it)->getAtomicNum());
        atom->setFormalCharge((*it)->getFormalCharge());

        result->addAtom(atom, true, true);
    }

    for (auto it = mol.beginBonds(), e = mol.endBonds(); it != e; ++it) {
        auto a1_idx = (*it)->getBeginAtomIdx();
        auto a2_idx = (*it)->getEndAtomIdx();
        result->addBond(a1_idx, a2_idx, (*it)->getBondType());
    }

    result->updatePropertyCache();
    return result;
}

} // namespace


void genric::register_molecule_edit(py::module &m) {
    m.doc() = "Helper functions for molecule edit functionality.";
    m.def("copy_edit_mol", &copy_edit_mol, py::call_guard<py::gil_scoped_release>());
}