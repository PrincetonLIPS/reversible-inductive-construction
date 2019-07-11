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


PYBIND11_MODULE(induc_gen_extensions, m) {
    m.doc() = "RDKit bindings and extensions for induc-gen";

    induc_gen::register_atom(m);
    induc_gen::register_bond(m);
    induc_gen::register_mol(m);
    induc_gen::register_molops(m);
    induc_gen::register_utilities(m);

    auto m_rep = m.def_submodule("molecule_representation");
    induc_gen::register_molecule_representation(m_rep);

    auto m_edit = m.def_submodule("molecule_edit");
    induc_gen::register_molecule_edit(m_edit);

}
