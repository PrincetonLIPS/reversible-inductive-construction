#include "module.h"
#include "wrap/wrappers.h"

#include <pybind11/buffer_info.h>
#include <pybind11/pybind11.h>

#include <GraphMol/Atom.h>
#include <GraphMol/AtomIterators.h>
#include <GraphMol/Bond.h>
#include <GraphMol/BondIterators.h>
#include <GraphMol/GraphMol.h>
#include <cstdint>
#include <exception>

namespace py = pybind11;

namespace {

const int NumAtomSymbols = 23;
const int AtomFeatureDimension = NumAtomSymbols + 6 + 5 + 4 + 1;
const int BondFeatureDimension = 5 + 6;

int GetAtomSymbolOneHotIndex(int atomic_num) {
    switch (atomic_num) {
    case 6:
        return 0; // "C"
    case 7:
        return 1; // "N"
    case 8:
        return 2; // "O"
    case 16:
        return 3; // "S"
    case 9:
        return 4; // "F"
    case 14:
        return 5; // "Si"
    case 15:
        return 6; // "P"
    case 17:
        return 7; // "Cl"
    case 35:
        return 8; // "Br"
    case 12:
        return 9; // "Mg"
    case 11:
        return 10; // "Na"
    case 20:
        return 11; // "Ca"
    case 26:
        return 12; // "Fe"
    case 13:
        return 13; // "Al"
    case 53:
        return 14; // "I"
    case 5:
        return 15; // "B"
    case 19:
        return 16; // "K"
    case 34:
        return 17; // "Se"
    case 30:
        return 18; // "Zn"
    case 1:
        return 19; // "H"
    case 29:
        return 20; // "Cu"
    case 25:
        return 21; // "Mn"
    default:
        return 22; // "unknown"
    }
}

int GetAtomFormalChargeIndex(int formalCharge) {
    switch (formalCharge) {
    case -1:
        return 0;
    case -2:
        return 1;
    case 1:
        return 2;
    case 2:
        return 3;
    case 0:
    default:
        return 4;
    }
}

template <typename T> void atom_features(T *data, RDKit::Atom const &atom) {
    int offset = 0;

    data[GetAtomSymbolOneHotIndex(atom.getAtomicNum())] = 1;
    offset += NumAtomSymbols;

    data[std::min(atom.getDegree(), 5u) + offset] = 1;
    offset += 6;

    data[GetAtomFormalChargeIndex(atom.getFormalCharge()) + offset] = 1;
    offset += 5;

    data[static_cast<int>(atom.getChiralTag()) + offset] = 1;
    offset += 4;

    data[offset] = atom.getIsAromatic() ? 1 : 0;
};

void fill_atom_features(py::buffer &view, RDKit::ROMol &mol) {
    auto info = view.request(true);

    if (info.ndim != 2) {
        throw std::runtime_error("Buffer must be 2-dimensional");
    }

    if (info.shape[1] < AtomFeatureDimension || info.shape[0] < mol.getNumAtoms()) {
        throw std::runtime_error("Buffer too small to fit specified molecule!");
    }

    if (info.strides[1] != info.itemsize) {
        throw std::runtime_error("Buffer must have C-contiguous layout");
    }

    if (info.format != py::format_descriptor<float>::format()) {
        throw std::runtime_error("Buffer must be of float type");
    }

    auto strides_i = info.strides[0] / info.itemsize;
    auto ptr = static_cast<float *>(info.ptr);

    int i = 0;

    for (auto it1 = mol.beginAtoms(), e1 = mol.endAtoms(); it1 != e1; ++i, ++it1) {
        atom_features(ptr + i * strides_i, **it1);
    }
}

// Writes the bond type one-hot vector to the given data array.
template <typename T> void bond_type_feature(T *data, RDKit::Bond::BondType type) {
    switch (type) {
    case RDKit::Bond::BondType::SINGLE:
        data[0] = 1;
        return;
    case RDKit::Bond::BondType::DOUBLE:
        data[1] = 1;
        return;
    case RDKit::Bond::BondType::TRIPLE:
        data[2] = 1;
        return;
    case RDKit::Bond::BondType::AROMATIC:
        data[3] = 1;
        return;
    default:
        return;
    }
}

// writes the bond features to the given data array.
template <typename T> void bond_features(T *data, RDKit::Bond const &bond) {
    int offset = 0;

    bond_type_feature(data + offset, bond.getBondType());
    offset += 4;

    data[offset] = induc_gen::BondIsInRing(&bond);
    offset += 1;

    data[offset + std::min(static_cast<int>(bond.getStereo()), 5)] = 1;
}

void fill_bond_features(py::buffer &view, RDKit::ROMol &mol) {
    auto info = view.request(true);

    if (info.ndim != 2) {
        throw std::runtime_error("Buffer must be 2-dimensional");
    }

    if (info.shape[1] < AtomFeatureDimension + BondFeatureDimension ||
        info.shape[0] < 2 * mol.getNumBonds()) {
        throw std::runtime_error("Buffer too small to fit specified molecule!");
    }

    if (info.strides[1] != info.itemsize) {
        throw std::runtime_error("Buffer must have C-contiguous layout");
    }

    if (info.format != py::format_descriptor<float>::format()) {
        throw std::runtime_error("Buffer must be of float type");
    }

    {
        py::gil_scoped_release release;

        int i = 0;

        auto strides_i = info.strides[0] / info.itemsize;
        auto ptr = static_cast<float *>(info.ptr);

        for (auto it = mol.beginBonds(), e = mol.endBonds(); it != e; ++i, ++it) {
            // fill in bonds in both directions.
            atom_features(ptr + (2 * i) * strides_i, *(*it)->getBeginAtom());
            bond_features(ptr + (2 * i) * strides_i + AtomFeatureDimension, **it);

            atom_features(ptr + (2 * i + 1) * strides_i, *(*it)->getEndAtom());
            bond_features(ptr + (2 * i + 1) * strides_i + AtomFeatureDimension, **it);
        }
    }
}

bool check_format_i32(std::string const &format) { return format == "i" || format == "l"; }

void fill_atom_bond_list(py::buffer &view, RDKit::ROMol &mol, int max_neighbours) {
    auto info = view.request(true);

    if (!check_format_i32(info.format) || info.ndim != 2) {
        throw std::runtime_error("Incompatible buffer format!");
    }

    if (info.shape[0] < mol.getNumAtoms() || info.shape[1] < max_neighbours) {
        throw std::runtime_error("Buffer too small to fit specified molecule!");
    }

    {
        // release GIL once we acquire buffer pointer
        py::gil_scoped_release release;

        int i = 0;
        auto view_ptr = static_cast<std::int32_t *>(info.ptr);
        auto strides_i = info.strides[0] / info.itemsize;
        auto strides_j = info.strides[1] / info.itemsize;

        for (auto it1 = mol.beginAtoms(), e1 = mol.endAtoms(); it1 != e1; ++i, ++it1) {
            RDKit::ROMol::OEDGE_ITER bond_begin, bond_end;
            std::tie(bond_begin, bond_end) = mol.getAtomBonds(*it1);

            for (int j = 0; bond_begin != bond_end; ++j, ++bond_begin) {
                const RDKit::Bond *bond = mol[*bond_begin];

                view_ptr[i * strides_i + j * strides_j] =
                    2 * bond->getIdx() + (bond->getBeginAtomIdx() == (*it1)->getIdx()) + 1;
            }
        }
    }
}

void fill_bond_incidence_list(py::buffer &view, RDKit::ROMol &mol, int max_neighbours) {
    auto info = view.request(true);

    if (!check_format_i32(info.format) || info.ndim != 2) {
        throw std::runtime_error("Incompatible buffer format!");
    }

    if (info.shape[0] < 2 * mol.getNumBonds() + 1 || info.shape[1] < max_neighbours) {
        throw std::runtime_error("Buffer too small to fit specified molecule!");
    }

    auto view_ptr = static_cast<std::int32_t *>(info.ptr);
    auto strides_i = info.strides[0] / info.itemsize;
    auto strides_j = info.strides[1] / info.itemsize;

    int i = 0;

    {
        // release GIL once we acquire buffer pointer
        py::gil_scoped_release release;

        for (auto it1 = mol.beginBonds(), e1 = mol.endBonds(); it1 != e1; ++i, ++it1) {
            auto bond = *it1;

            auto fill_bond_incidence = [&](int index, const RDKit::Atom *atom) {
                RDKit::ROMol::OEDGE_ITER bond_begin, bond_end;
                std::tie(bond_begin, bond_end) = mol.getAtomBonds(atom);

                for (int j = 0; bond_begin != bond_end; ++j, ++bond_begin) {
                    const RDKit::Bond *bond2 = mol[*bond_begin];

                    if (bond2->getIdx() == bond->getIdx()) {
                        continue;
                    }

                    view_ptr[index * strides_i + j * strides_j] =
                        2 * bond2->getIdx() + (bond2->getBeginAtomIdx() == atom->getIdx()) + 1;
                }
            };

            fill_bond_incidence(2 * i + 1, bond->getBeginAtom());
            fill_bond_incidence(2 * i + 2, bond->getEndAtom());
        }
    }
}

void fill_atom_bond_list_sparse(py::buffer values, py::buffer index, const RDKit::ROMol &mol) {
    auto info_values = values.request(true);
    auto info_index = index.request(true);

    auto num_elements = mol.getNumBonds() * 2;

    if ((info_values.format != py::format_descriptor<float>::format()) ||
        (info_values.shape[0] != num_elements)) {
        throw std::runtime_error(
            "Invalid values buffer. Must be of type float and length 2 * number of bonds.");
    }

    if (!check_format_i32(info_index.format) || info_index.ndim != 2) {
        throw std::runtime_error("Invalid index buffer. Must be of type int32 and dimension 2.");
    }

    if (info_index.shape[0] != 2 || info_index.shape[1] != num_elements) {
        throw std::runtime_error(
            "Invalid shape for index buffer. Must be of size 2 x (2 * num_bonds).");
    }

    auto values_ptr = static_cast<float *>(info_values.ptr);
    auto values_stride = info_values.strides[0] / info_values.itemsize;

    auto index_ptr = static_cast<int32_t *>(info_index.ptr);
    auto index_offset_0 = info_index.strides[0] / info_index.itemsize;
    auto index_stride = info_index.strides[1] / info_index.itemsize;

    {
        py::gil_scoped_release release_gil;

        int i = 0;
        int current_index = 0;

        for (auto it1 = mol.beginAtoms(), e1 = mol.endAtoms(); it1 != e1; ++i, ++it1) {
            RDKit::ROMol::OEDGE_ITER bond_begin, bond_end;
            std::tie(bond_begin, bond_end) = mol.getAtomBonds(*it1);

            auto value = 1 / sqrt((*it1)->getDegree());

            for (int j = 0; bond_begin != bond_end; ++j, ++bond_begin) {
                const RDKit::Bond *bond = mol[*bond_begin];

                index_ptr[current_index * index_stride] = i;
                index_ptr[index_offset_0 + current_index * index_stride] =
                    2 * bond->getIdx() + (bond->getBeginAtomIdx() == (*it1)->getIdx());
                values_ptr[current_index] = value;
                current_index += 1;
            }
        }
    }
}

int get_edge_incidence_size(RDKit::ROMol const &mol) {
    int result = 0;

    for (auto it1 = mol.beginAtoms(), it2 = mol.endAtoms(); it1 != it2; ++it1) {
        auto degree = (*it1)->getDegree();
        result += degree * (degree - 1);
    }

    return result;
}

void fill_bond_incidence_list_sparse(py::buffer values, py::buffer index,
                                     const RDKit::ROMol &mol) {
    auto info_values = values.request(true);
    auto info_index = index.request(true);

    if ((info_values.format != py::format_descriptor<float>::format()) || info_values.ndim != 1) {
        throw std::runtime_error("Values must be one-dimensional floating-point array.");
    }

    if (!check_format_i32(info_index.format) || info_index.ndim != 2) {
        throw std::runtime_error("index must be two-dimensional integer array.");
    }

    auto values_ptr = static_cast<float *>(info_values.ptr);
    auto values_stride = info_values.strides[0] / info_values.itemsize;

    auto index_ptr = static_cast<int32_t *>(info_index.ptr);
    auto index_offset_0 = info_index.strides[0] / info_index.itemsize;
    auto index_stride = info_index.strides[1] / info_index.itemsize;

    {
        // release GIL once we acquire buffer pointer
        py::gil_scoped_release release;
        int i = 0;
        int current_index = 0;

        for (auto it1 = mol.beginBonds(), e1 = mol.endBonds(); it1 != e1; ++i, ++it1) {
            auto bond = *it1;

            auto a1 = bond->getBeginAtom();
            auto a2 = bond->getEndAtom();

            float degree = a1->getDegree() + a2->getDegree() - 2;
            float value = 1 / sqrt(degree);

            auto fill_bond_incidence = [&](int index, const RDKit::Atom *atom) {
                RDKit::ROMol::OEDGE_ITER bond_begin, bond_end;
                std::tie(bond_begin, bond_end) = mol.getAtomBonds(atom);

                for (int j = 0; bond_begin != bond_end; ++j, ++bond_begin) {
                    const RDKit::Bond *bond2 = mol[*bond_begin];

                    if (bond2->getIdx() == bond->getIdx()) {
                        continue;
                    }

                    index_ptr[current_index * index_stride] = index;
                    index_ptr[index_offset_0 + current_index * index_stride] =
                        2 * bond2->getIdx() + (bond2->getBeginAtomIdx() == atom->getIdx());

                    values_ptr[current_index] = value;
                    current_index += 1;
                }
            };

            fill_bond_incidence(2 * i, a1);
            fill_bond_incidence(2 * i + 1, a2);
        }
    }
}

void fill_atom_bond_list_segment(py::buffer scopes, py::buffer index, const RDKit::ROMol &mol) {
    auto info_scopes = scopes.request(true);
    auto info_index = index.request(true);

    if (!check_format_i32(info_scopes.format) || info_scopes.ndim != 2) {
        throw std::runtime_error("scopes must be two-dimensional integer array.");
    }

    if (!check_format_i32(info_index.format) || info_index.ndim != 1) {
        throw std::runtime_error("index must be one-dimensional integer array.");
    }

    if (info_scopes.shape[0] != mol.getNumAtoms()) {
        throw std::runtime_error("Scopes must be of length number of atoms.");
    }

    if (info_index.shape[0] != 2 * mol.getNumBonds()) {
        throw std::runtime_error("Index must be of length twice number of bonds.");
    }

    auto index_ptr = static_cast<int32_t *>(info_index.ptr);
    auto index_stride = info_index.strides[0] / info_index.itemsize;

    auto scopes_ptr = static_cast<int32_t *>(info_scopes.ptr);
    auto scopes_stride = info_scopes.strides[0] / info_scopes.itemsize;
    auto scopes_offset_1 = info_scopes.strides[1] / info_scopes.itemsize;

    {
        py::gil_scoped_release release;

        int offset = 0;
        int i = 0;
        int current_index = 0;

        for (auto it1 = mol.beginAtoms(), e1 = mol.endAtoms(); it1 != e1; ++i, ++it1) {
            RDKit::ROMol::OEDGE_ITER bond_begin, bond_end;
            std::tie(bond_begin, bond_end) = mol.getAtomBonds(*it1);

            auto value = 1 / sqrt((*it1)->getDegree());

            auto degree = (*it1)->getDegree();
            scopes_ptr[i * scopes_stride] = offset;
            scopes_ptr[i * scopes_stride + scopes_offset_1] = degree;
            offset += degree;

            for (int j = 0; bond_begin != bond_end; ++j, ++bond_begin) {
                const RDKit::Bond *bond = mol[*bond_begin];

                index_ptr[current_index * index_stride] =
                    2 * bond->getIdx() + (bond->getBeginAtomIdx() == (*it1)->getIdx());
                current_index += 1;
            }
        }
    }
}

void fill_bond_incidence_list_segment(py::buffer scopes, py::buffer index,
                                      const RDKit::ROMol &mol) {
    auto info_scopes = scopes.request(true);
    auto info_index = index.request(true);

    if (!check_format_i32(info_scopes.format) || info_scopes.ndim != 2) {
        throw std::runtime_error("scopes must be two-dimensional integer array.");
    }

    if (!check_format_i32(info_index.format) || info_index.ndim != 1) {
        throw std::runtime_error("index must be one-dimensional integer array.");
    }

    if (info_scopes.shape[0] != 2 * mol.getNumBonds()) {
        throw std::runtime_error("scopes must be of length 2 * number of bonds.");
    }

    auto index_ptr = static_cast<int32_t *>(info_index.ptr);
    auto index_stride = info_index.strides[0] / info_index.itemsize;

    auto scopes_ptr = static_cast<int32_t *>(info_scopes.ptr);
    auto scopes_stride = info_scopes.strides[0] / info_scopes.itemsize;
    auto scopes_offset_1 = info_scopes.strides[1] / info_scopes.itemsize;

    {
        // release GIL once we acquire buffer pointer
        py::gil_scoped_release release;
        int i = 0;
        int current_index = 0;
        int offset = 0;

        for (auto it1 = mol.beginBonds(), e1 = mol.endBonds(); it1 != e1; ++i, ++it1) {
            auto bond = *it1;

            auto a1 = bond->getBeginAtom();
            auto a2 = bond->getEndAtom();

            float degree = a1->getDegree() + a2->getDegree() - 2;
            float value = 1 / sqrt(degree);

            auto fill_bond_incidence = [&](int index, const RDKit::Atom *atom) {
                RDKit::ROMol::OEDGE_ITER bond_begin, bond_end;
                std::tie(bond_begin, bond_end) = mol.getAtomBonds(atom);
                auto num_bonds = 0;

                for (int j = 0; bond_begin != bond_end; ++j, ++bond_begin) {
                    const RDKit::Bond *bond2 = mol[*bond_begin];

                    if (bond2->getIdx() == bond->getIdx()) {
                        continue;
                    }

                    index_ptr[current_index * index_stride] =
                        2 * bond2->getIdx() + (bond2->getBeginAtomIdx() == atom->getIdx());
                    current_index += 1;
                    num_bonds += 1;
                }

                scopes_ptr[index * scopes_stride] = offset;
                scopes_ptr[index * scopes_stride + scopes_offset_1] = num_bonds;
                offset += num_bonds;
            };

            fill_bond_incidence(2 * i, a1);
            fill_bond_incidence(2 * i + 1, a2);
        }
    }
}

} // namespace

namespace induc_gen {

void register_molecule_representation(py::module &m) {
    m.doc() = "Helpers for computing molecule representations";
    m.def("fill_atom_bond_list", &fill_atom_bond_list,
          "Fills the given buffer with atom-bond incidence information.");

    m.def("fill_bond_incidence_list", &fill_bond_incidence_list,
          "Fills the given buffer with bond-bond incidence information.");

    m.def("fill_atom_features", &fill_atom_features,
          "Fills the given buffer with atom features. \n\n"
          "The buffer must be a C-contiguous two-dimensional float buffer with number of rows at "
          "least the number of molecules mol, and number of columns at least "
          "`AtomFeatureDimension`.");

    m.def(
        "fill_bond_features", &fill_bond_features,
        "Fills the given buffer with bond features. \n\n"
        "The buffer must be a C-contiguous two-dimensional float buffer with number of columns at "
        "least `AtomFeatureDimension + BondFeatureDimension`, and number of rows at least "
        "twice the number of bonds plus one.");

    m.def("fill_atom_bond_list_sparse", &fill_atom_bond_list_sparse, py::arg("values"),
          py::arg("index"), py::arg("mol"),
          "Fills the given values and index buffers with the atom-bond incidence information."
          "The two buffers represent a sparse weighted adjacency matrix in COO format.");

    m.def("fill_bond_incidence_list_sparse", &fill_bond_incidence_list_sparse, py::arg("values"),
          py::arg("index"), py::arg("mol"),
          "Fills the given values and index buffers with the bond-bond incidence information."
          "The two buffers represent a sparse weighted adjacency matrix in COO format.");

    m.def("get_edge_incidence_size", &get_edge_incidence_size,
          "Computes the number of non-zero entries in the bond incidence adjacency graph.");

    m.attr("AtomFeatureDimension") = py::int_(AtomFeatureDimension);
    m.attr("BondFeatureDimension") = py::int_(BondFeatureDimension);

    m.def("fill_atom_bond_list_segment", &fill_atom_bond_list_segment, py::arg("scopes"),
          py::arg("index"), py::arg("mol"));

    m.def("fill_bond_incidence_list_segment", &fill_bond_incidence_list_segment, py::arg("scopes"),
          py::arg("index"), py::arg("mol"));
}
} // namespace induc_gen