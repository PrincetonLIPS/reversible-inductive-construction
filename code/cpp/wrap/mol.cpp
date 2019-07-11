#include "wrappers.h"

#include <GraphMol/AtomIterators.h>
#include <GraphMol/BondIterators.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolPickler.h>

#include <pybind11/stl.h>

namespace py = pybind11;

using Atom = RDKit::Atom;
using Bond = RDKit::Bond;
using ROMol = RDKit::ROMol;
using RWMol = RDKit::RWMol;

namespace {
template <typename IterFunc, typename LenFunc> struct ReadonlyIterSeq {
    std::shared_ptr<RDKit::ROMol> mol;
    IterFunc get_iter;
    LenFunc get_len;

    ReadonlyIterSeq(std::shared_ptr<RDKit::ROMol> mol) : mol(std::move(mol)) {}

    py::iterator dunder_iter() { return get_iter(*mol); }
    int dunder_len() { return get_len(*mol); }

    static void add_binding(pybind11::module &m, const char *name) {
        py::class_<ReadonlyIterSeq>(m, name)
            .def("__iter__", &ReadonlyIterSeq::dunder_iter)
            .def("__len__", &ReadonlyIterSeq::dunder_len);
    }
};

struct BondFunc {
    struct Iter {
        py::iterator operator()(RDKit::ROMol const &mol) {
            return py::make_iterator(mol.beginBonds(), mol.endBonds(),
                                     py::return_value_policy::reference_internal);
        }
    };

    struct Len {
        unsigned int operator()(RDKit::ROMol const &mol) { return mol.getNumBonds(); }
    };
};

typedef ReadonlyIterSeq<BondFunc::Iter, BondFunc::Len> BondIterSeq;

struct AtomFunc {
    struct Iter {
        py::iterator operator()(RDKit::ROMol const &mol) {
            return py::make_iterator(mol.beginAtoms(), mol.endAtoms(),
                                     py::return_value_policy::reference_internal);
        }
    };

    struct Len {
        unsigned int operator()(RDKit::ROMol const &mol) { return mol.getNumAtoms(); }
    };
};

typedef ReadonlyIterSeq<AtomFunc::Iter, AtomFunc::Len> AtomIterSeq;
} // namespace

void induc_gen::register_mol(py::module &m) {
    BondIterSeq::add_binding(m, "BondIterSeq");
    AtomIterSeq::add_binding(m, "AtomIterSeq");

    auto romol =
        py::class_<RDKit::ROMol, std::shared_ptr<RDKit::ROMol>>(m, "Mol")
            .def(py::init<const std::string &>())
            .def(py::init<const ROMol &>())
            .def(py::init<const ROMol &, bool>())
            .def(py::init<const ROMol &, bool, int>())
            .def("GetNumAtoms", &ROMol::getNumAtoms, (py::arg("onlyExplicit") = true),
                 "Returns the number of atoms in the molecule.\n\n"
                 "  ARGUMENTS:\n"
                 "    - onlyExplicit: (optional) include only explicit atoms ")
            .def("GetNumHeavyAtoms", &ROMol::getNumHeavyAtoms,
                 "Returns the number of heavy atoms (atomic number >1) in the "
                 "molecule.\n\n")
            .def("GetNumBonds", &ROMol::getNumBonds, py::arg("onlyHeavy") = true)
            .def("GetAtomWithIdx", (Atom * (ROMol::*)(unsigned int)) & ROMol::getAtomWithIdx,
                 py::return_value_policy::reference_internal,
                 "Returns a particular Atom.\n\n"
                 "  ARGUMENTS:\n"
                 "    - idx: which Atom to return\n\n"
                 "  NOTE: atom indices start at 0\n")
            .def("GetBondWithIdx", (Bond * (ROMol::*)(unsigned int)) & ROMol::getBondWithIdx,
                 py::return_value_policy::reference_internal)
            .def("GetBonds",
                 [](std::shared_ptr<ROMol> mol) { return BondIterSeq(std::move(mol)); })
            .def("GetAtoms",
                 [](std::shared_ptr<ROMol> mol) { return AtomIterSeq(std::move(mol)); })
            .def(py::pickle(
                [](const ROMol &self) {
                    std::string res;
                    RDKit::MolPickler::pickleMol(self, res);
                    return py::bytes(res);
                },
                [](py::bytes b) {
                    return std::make_shared<ROMol>(b);
                }));

    py::class_<RDKit::RWMol, std::shared_ptr<RDKit::RWMol>>(m, "RWMol", romol)
        .def(py::init<const ROMol &, bool, int>(), py::arg("other"), py::arg("quickCopy") = false,
             py::arg("conformerId") = -1)
        .def("AddAtom", (unsigned int (RWMol::*)(Atom *, bool, bool)) & RWMol::addAtom,
             py::arg("atom"), py::arg("updateLabel") = true, py::arg("takeOwnership") = false)
        .def("RemoveAtom", (void (RWMol::*)(unsigned int)) & RDKit::RWMol::removeAtom)
        .def("AddBond",
             (unsigned int (RWMol::*)(unsigned int, unsigned int, Bond::BondType)) &
                 RWMol::addBond,
             py::arg("beginAtomIdx"), py::arg("endAtomIdx"), py::arg("order") = Bond::UNSPECIFIED)
        .def("GetMol", [](RWMol const &mol) { return std::make_shared<ROMol>(mol); })
        .def("RemoveBond", &RWMol::removeBond)
        .def(py::pickle(
            [](const RWMol &self) {
                std::string res;
                RDKit::MolPickler::pickleMol(self, res);
                return py::bytes(res);
            },
            [](py::bytes b) {
                return std::make_shared<RWMol>(static_cast<std::string>(b));
            }
        ));
}
