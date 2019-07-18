#include "wrappers.h"

#include <RDGeneral/Invariant.h>
#include <RDGeneral/RDLog.h>

namespace py = pybind11;

namespace {

void enable_log(std::string spec) { boost::logging::enable_logs(spec); }

void disable_log(std::string spec) { boost::logging::disable_logs(spec); }

} // namespace

void genric::register_utilities(py::module &m) {
    static py::exception<Invar::Invariant> invariant_exc(m, "RDInvariantError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const Invar::Invariant &e) {
            invariant_exc(e.toString().c_str());
        }
    });

    m.def("enable_log", &enable_log);
    m.def("disable_log", &disable_log);
}