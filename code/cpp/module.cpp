#include <GraphMol/GraphMol.h>
#include <RDGeneral/Invariant.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wrap/wrappers.h"
#include "module.h"

namespace py = pybind11;

using Atom = RDKit::Atom;
using Bond = RDKit::Bond;
using ROMol = RDKit::ROMol;
using RWMol = RDKit::RWMol;


PYBIND11_MODULE(genric_extensions, m) {
    m.doc() = "RDKit bindings and extensions for induc-gen";

    genric::register_atom(m);
    genric::register_bond(m);
    genric::register_mol(m);
    genric::register_molops(m);
    genric::register_utilities(m);

    auto m_rep = m.def_submodule("molecule_representation");
    genric::register_molecule_representation(m_rep);

    auto m_edit = m.def_submodule("molecule_edit");
    genric::register_molecule_edit(m_edit);

}
