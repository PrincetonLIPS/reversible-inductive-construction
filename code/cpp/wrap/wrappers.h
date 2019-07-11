#pragma once

#include <pybind11/pybind11.h>

namespace RDKit {
    class Bond;
}

namespace induc_gen {
    void register_atom(pybind11::module &m);
    void register_bond(pybind11::module &m);
    void register_mol(pybind11::module &m);
    void register_molops(pybind11::module &m);
    void register_utilities(pybind11::module &m);


    bool BondIsInRing(const RDKit::Bond *bond);
}
