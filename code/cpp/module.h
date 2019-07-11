#pragma once

#include <pybind11/pybind11.h>

namespace induc_gen {
void register_molecule_representation(pybind11::module &m);
void register_molecule_edit(pybind11::module &m);
}