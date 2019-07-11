#include "wrappers.h"

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using Atom = RDKit::Atom;

namespace {

py::tuple AtomGetNeighbors(Atom *atom) {
    py::list res;
    const RDKit::ROMol *parent = &atom->getOwningMol();
    RDKit::ROMol::ADJ_ITER begin, end;
    boost::tie(begin, end) = parent->getAtomNeighbors(atom);

    while (begin != end) {
        res.append(py::cast(parent->getAtomWithIdx(*begin)));
        begin++;
    }
    return py::tuple(res);
}

py::tuple AtomGetBonds(Atom *atom) {
    py::list res;
    const RDKit::ROMol *parent = &atom->getOwningMol();
    RDKit::ROMol::OEDGE_ITER begin, end;
    boost::tie(begin, end) = parent->getAtomBonds(atom);
    while (begin != end) {
        const RDKit::Bond *tmpB = (*parent)[*begin];
        res.append(py::cast(tmpB));
        begin++;
    }
    return py::tuple(res);
}

bool AtomIsInRing(const Atom *atom) {
    if (!atom->getOwningMol().getRingInfo()->isInitialized()) {
        RDKit::MolOps::findSSSR(atom->getOwningMol());
    }
    return atom->getOwningMol().getRingInfo()->numAtomRings(atom->getIdx()) != 0;
}

bool AtomIsInRingSize(const Atom *atom, int size) {
    if (!atom->getOwningMol().getRingInfo()->isInitialized()) {
        RDKit::MolOps::findSSSR(atom->getOwningMol());
    }
    return atom->getOwningMol().getRingInfo()->isAtomInRingOfSize(atom->getIdx(), size);
}

} // namespace

void induc_gen::register_atom(py::module &m) {
    py::enum_<RDKit::Atom::ChiralType>(m, "ChiralType")
        .value("CHI_UNSPECIFIED", Atom::CHI_UNSPECIFIED)
        .value("CHI_TETRAHEDRAL_CW", Atom::CHI_TETRAHEDRAL_CW)
        .value("CHI_TETRAHEDRAL_CCW", Atom::CHI_TETRAHEDRAL_CCW)
        .value("CHI_OTHER", Atom::CHI_OTHER)
        .export_values();

    py::class_<RDKit::Atom>(m, "Atom")
        .def(py::init<std::string>())
        .def(py::init<unsigned int>())
        .def("GetAtomicNum", &Atom::getAtomicNum, "Returns the atomic number.")
        .def("SetAtomicNum", &Atom::setAtomicNum,
             "Sets the atomic number, takes an integer value as an argument")
        .def("GetSymbol", &Atom::getSymbol, "Returns the atomic symbol (a string)\n")
        .def("GetIdx", &Atom::getIdx, "Returns the atom's index (ordering in the molecule)\n")
        .def("GetDegree", &Atom::getDegree,
             "Returns the degree of the atom in the molecule.\n\n"
             "  The degree of an atom is defined to be its number of\n"
             "  directly-bonded neighbors.\n"
             "  The degree is independent of bond orders, but is dependent\n"
             "    on whether or not Hs are explicit in the graph.\n")
        .def("GetTotalDegree", &Atom::getTotalDegree,
             "Returns the degree of the atom in the molecule including Hs.\n\n"
             "  The degree of an atom is defined to be its number of\n"
             "  directly-bonded neighbors.\n"
             "  The degree is independent of bond orders.\n")
        .def("GetTotalNumHs", &Atom::getTotalNumHs,
             (py::arg("self"), py::arg("includeNeighbors") = false),
             "Returns the total number of Hs (explicit and implicit) on the "
             "atom.\n\n"
             "  ARGUMENTS:\n\n"
             "    - includeNeighbors: (optional) toggles inclusion of "
             "neighboring H atoms in the sum.\n"
             "      Defaults to 0.\n")
        .def("GetNumImplicitHs", &Atom::getNumImplicitHs,
             "Returns the total number of implicit Hs on the atom.\n")
        .def("GetExplicitValence", &Atom::getExplicitValence,
             "Returns the explicit valence of the atom.\n")
        .def("GetImplicitValence", &Atom::getImplicitValence,
             "Returns the number of implicit Hs on the atom.\n")
        .def("GetTotalValence", &Atom::getTotalValence,
             "Returns the total valence (explicit + implicit) of the atom.\n\n")

        .def("GetFormalCharge", &Atom::getFormalCharge)
        .def("SetFormalCharge", &Atom::setFormalCharge)

        .def("IsInRing", &AtomIsInRing)
        .def("IsInRingSize", &AtomIsInRingSize)

        .def("SetNoImplicit", &Atom::setNoImplicit,
             "Sets a marker on the atom that *disallows* implicit Hs.\n"
             "  This holds even if the atom would otherwise have implicit Hs "
             "added.\n")
        .def("GetNoImplicit", &Atom::getNoImplicit,
             "Returns whether or not the atom is *allowed* to have implicit "
             "Hs.\n")
        .def("SetNumExplicitHs", &Atom::setNumExplicitHs)
        .def("GetNumExplicitHs", &Atom::getNumExplicitHs)
        .def("SetIsAromatic", &Atom::setIsAromatic)
        .def("GetIsAromatic", &Atom::getIsAromatic)
        .def("GetMass", &Atom::getMass)
        .def("SetIsotope", &Atom::setIsotope)
        .def("GetIsotope", &Atom::getIsotope)
        .def("SetNumRadicalElectrons", &Atom::setNumRadicalElectrons)
        .def("GetNumRadicalElectrons", &Atom::getNumRadicalElectrons)

        .def("SetChiralTag", &Atom::setChiralTag)
        .def("InvertChirality", &Atom::invertChirality)
        .def("GetChiralTag", &Atom::getChiralTag)

        .def("SetHybridization", &Atom::setHybridization,
             "Sets the hybridization of the atom.\n"
             "  The argument should be a HybridizationType\n")
        .def("GetHybridization", &Atom::getHybridization, "Returns the atom's hybridization.\n")

        .def("SetAtomMapNum", &Atom::setAtomMapNum, py::arg("mapno"), py::arg("strict") = false,
             "Sets the atom map number. A value of zero clears the map.")
        .def("GetAtomMapNum", &Atom::getAtomMapNum, "Gets the atom map number.")

        .def("GetOwningMol", &Atom::getOwningMol, "Returns the Mol that owns this atom.\n",
             py::return_value_policy::reference_internal)

        .def("GetNeighbors", AtomGetNeighbors,
             "Returns a read-only sequence of the atom's neighbors\n")
        .def("GetBonds", AtomGetBonds, "Returns a read-only sequence of the atom's bonds\n")

        .def("Match", (bool (Atom::*)(const Atom *) const) & Atom::Match,
             "Returns whether or not this atom matches another Atom.\n\n"
             "  Each Atom (or query Atom) has a query function which is\n"
             "  used for this type of matching.\n\n"
             "  ARGUMENTS:\n"
             "    - other: the other Atom to which to compare\n");
}
