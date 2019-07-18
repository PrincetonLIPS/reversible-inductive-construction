#include "wrappers.h"

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <boost/optional.hpp>

#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include <pybind11/stl.h>

namespace py = pybind11;

using Atom = RDKit::Atom;
using Bond = RDKit::Bond;
using ROMol = RDKit::ROMol;
using RWMol = RDKit::RWMol;

namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};
} // namespace detail
} // namespace pybind11

namespace {
std::shared_ptr<RWMol> MolFromSmiles(py::object ismiles, bool sanitize, py::dict replDict) {
    std::map<std::string, std::string> replacements;

    for (auto const &it : replDict) {
        replacements[py::str(it.first)] = py::str(it.second);
    }

    RDKit::RWMol *newM;
    std::string smiles = py::str(ismiles);

    try {
        newM = RDKit::SmilesToMol(smiles, 0, sanitize, &replacements);
    } catch (...) {
        newM = nullptr;
    }

    newM->updatePropertyCache();

    return std::shared_ptr<RWMol>(newM);
}

std::string MolFragmentToSmiles(const ROMol &mol, py::list atomsToUse,
                                boost::optional<std::vector<int>> bondsToUse,
                                boost::optional<std::vector<std::string>> atomSymbols,
                                boost::optional<std::vector<std::string>> bondSymbols,
                                bool doIsomericSmiles, bool doKekule, int rootedAtAtom,
                                bool canonical, bool allBondsExplicit, bool allHsExplicit) {
    std::vector<int> atoms = py::cast<std::vector<int>>(atomsToUse);

    return RDKit::MolFragmentToSmiles(mol, atoms, bondsToUse.get_ptr(), atomSymbols.get_ptr(),
                                      bondSymbols.get_ptr(), doIsomericSmiles, doKekule,
                                      rootedAtAtom, canonical, allBondsExplicit, allHsExplicit);
}

std::vector<std::vector<int>> getSymmSssr(ROMol &mol) {
    std::vector<std::vector<int>> result;
    RDKit::MolOps::symmetrizeSSSR(mol, result);
    return result;
}

RDKit::MolOps::SanitizeFlags sanitizeMol(RWMol &mol, RDKit::MolOps::SanitizeFlags sanitizeOps) {
    unsigned int failedOperations;
    RDKit::MolOps::sanitizeMol(mol, failedOperations, sanitizeOps);
    return static_cast<RDKit::MolOps::SanitizeFlags>(failedOperations);
}
} // namespace

void genric::register_molops(py::module &m) {
    py::enum_<RDKit::MolOps::SanitizeFlags>(m, "SanitizeFlags", py::arithmetic())
        .value("SANITIZE_NONE", RDKit::MolOps::SANITIZE_NONE)
        .value("SANITIZE_CLEANUP", RDKit::MolOps::SANITIZE_CLEANUP)
        .value("SANITIZE_PROPERTIES", RDKit::MolOps::SANITIZE_PROPERTIES)
        .value("SANITIZE_SYMMRINGS", RDKit::MolOps::SANITIZE_SYMMRINGS)
        .value("SANITIZE_KEKULIZE", RDKit::MolOps::SANITIZE_KEKULIZE)
        .value("SANITIZE_FINDRADICALS", RDKit::MolOps::SANITIZE_FINDRADICALS)
        .value("SANITIZE_SETAROMATICITY", RDKit::MolOps::SANITIZE_SETAROMATICITY)
        .value("SANITIZE_SETCONJUGATION", RDKit::MolOps::SANITIZE_SETCONJUGATION)
        .value("SANITIZE_SETHYBRIDIZATION", RDKit::MolOps::SANITIZE_SETHYBRIDIZATION)
        .value("SANITIZE_CLEANUPCHIRALITY", RDKit::MolOps::SANITIZE_CLEANUPCHIRALITY)
        .value("SANITIZE_ADJUSTHS", RDKit::MolOps::SANITIZE_ADJUSTHS)
        .value("SANITIZE_ALL", RDKit::MolOps::SANITIZE_ALL)
        .export_values();

    m.def("MolFromSmiles", &MolFromSmiles, py::arg("SMILES"), py::arg("sanitize") = true,
          py::arg("replacements") = py::dict());

    m.def("MolToSmiles", &RDKit::MolToSmiles, py::arg("mol"), py::arg("isomericSmiles") = true,
          py::arg("kekuleSmiles") = false, py::arg("rootedAtAtom") = -1,
          py::arg("canonical") = true, py::arg("allBondsExplicit") = false,
          py::arg("allHsExplicit") = false, py::arg("doRandom") = false);

    m.def("MolFragmentToSmiles", &MolFragmentToSmiles, py::arg("mol"), py::arg("atomsToUse"),
          py::arg("bondsToUse") = py::none(), py::arg("atomSymbols") = py::none(),
          py::arg("bondSymbols") = py::none(), py::arg("isomericSmiles") = false,
          py::arg("kekuleSmiles") = false, py::arg("rootedAtAtom") = -1,
          py::arg("canonical") = true, py::arg("allBondsExplicit") = false,
          py::arg("allHsExplicit") = false);

    m.def("Kekulize", RDKit::MolOps::Kekulize, py::arg("mol"),
          py::arg("clearAromaticFlags") = false, py::arg("maxBackTracks") = 100);

    py::class_<RDGeom::Point3D>(m, "Point3D")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &RDGeom::Point3D::x)
        .def_readwrite("y", &RDGeom::Point3D::y)
        .def_readwrite("z", &RDGeom::Point3D::z);

    m.def("CombineMols", RDKit::combineMols, py::arg("mol1"), py::arg("mol2"),
          py::arg("offset") = RDGeom::Point3D(0, 0, 0));

    m.def("GetSymmSSSR", &getSymmSssr);
    m.def("SanitizeMol", &sanitizeMol, py::arg("mol"),
          py::arg("sanitizeOps") = RDKit::MolOps::SANITIZE_ALL);
}