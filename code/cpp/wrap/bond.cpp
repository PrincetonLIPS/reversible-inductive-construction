#include "wrappers.h"

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using Atom = RDKit::Atom;
using Bond = RDKit::Bond;

bool genric::BondIsInRing(const Bond *bond) {
    if (!bond->getOwningMol().getRingInfo()->isInitialized()) {
        RDKit::MolOps::findSSSR(bond->getOwningMol());
    }
    return bond->getOwningMol().getRingInfo()->numBondRings(bond->getIdx()) != 0;
}

void genric::register_bond(pybind11::module &m) {
    py::enum_<RDKit::Bond::BondType>(m, "BondType")
        .value("UNSPECIFIED", Bond::BondType::UNSPECIFIED)
        .value("SINGLE", Bond::BondType::SINGLE)
        .value("DOUBLE", Bond::BondType::DOUBLE)
        .value("TRIPLE", Bond::BondType::TRIPLE)
        .value("QUADRUPLE", Bond::BondType::QUADRUPLE)
        .value("QUINTUPLE", Bond::BondType::QUINTUPLE)
        .value("HEXTUPLE", Bond::BondType::HEXTUPLE)
        .value("ONEANDAHALF", Bond::BondType::ONEANDAHALF)
        .value("TWOANDAHALF", Bond::BondType::TWOANDAHALF)
        .value("THREEANDAHALF", Bond::BondType::THREEANDAHALF)
        .value("FOURANDAHALF", Bond::BondType::FOURANDAHALF)
        .value("FIVEANDAHALF", Bond::BondType::FIVEANDAHALF)
        .value("AROMATIC", Bond::BondType::AROMATIC)
        .value("IONIC", Bond::BondType::IONIC)
        .value("HYDROGEN", Bond::BondType::HYDROGEN)
        .value("THREECENTER", Bond::BondType::THREECENTER)
        .value("DATIVEONE", Bond::BondType::DATIVEONE)
        .value("DATIVE", Bond::BondType::DATIVE)
        .value("DATIVEL", Bond::BondType::DATIVEL)
        .value("DATIVER", Bond::BondType::DATIVER)
        .value("OTHER", Bond::BondType::OTHER)
        .value("ZERO", Bond::BondType::ZERO)
        .export_values();

    py::enum_<RDKit::Bond::BondStereo>(m, "BondStereo")
        .value("STEREONONE", Bond::BondStereo::STEREONONE)
        .value("STEREOANY", Bond::BondStereo::STEREOANY)
        .value("STEREOZ", Bond::BondStereo::STEREOZ)
        .value("STEREOE", Bond::BondStereo::STEREOE)
        .value("STEREOCIS", Bond::BondStereo::STEREOCIS)
        .value("STEREOTRANS", Bond::BondStereo::STEREOTRANS)
        .export_values();

    py::class_<RDKit::Bond>(m, "Bond")
        .def("GetOwningMol", &Bond::getOwningMol, py::return_value_policy::reference_internal)
        .def("GetBondType", &Bond::getBondType)
        .def("GetBondTypeAsDouble", &Bond::getBondTypeAsDouble)
        .def("GetBondDir", &Bond::getBondDir)
        .def("GetStereo", &Bond::getStereo)
        .def("GetIdx", &Bond::getIdx)
        .def("GetBeginAtomIdx", &Bond::getBeginAtomIdx)
        .def("GetEndAtomIdx", &Bond::getEndAtomIdx)
        .def("GetBeginAtom", &Bond::getBeginAtom, py::return_value_policy::reference_internal)
        .def("GetEndAtom", &Bond::getEndAtom, py::return_value_policy::reference_internal)
        .def("GetIsAromatic", &Bond::getIsAromatic)
        .def("Match", (bool (Bond::*)(const Bond *) const) & Bond::Match)
        .def("IsInRing", &BondIsInRing)
        .def("GetOtherAtomIdx", &Bond::getOtherAtomIdx)
        .def("GetOtherAtom", &Bond::getOtherAtom, py::return_value_policy::reference_internal);
}