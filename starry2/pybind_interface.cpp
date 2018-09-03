#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include "utils.h"
#include "pybind_interface.h"
#include "docstrings.h"
#include "kepler.h"
namespace py = pybind11;

PYBIND11_MODULE(_starry2, m) {

    using utils::Matrix;
    using utils::Vector;
    using utils::Multi;
    using pybind_interface::bindMap;
    using pybind_interface::bindBody;
    using pybind_interface::bindPrimary;
    using pybind_interface::bindSecondary;
    using pybind_interface::bindSystem;
    using namespace pybind11::literals;

    // The four supported Map types
    using T1 = Vector<double>;
    using T2 = Vector<Multi>;
    using T3 = Matrix<double>;
    using T4 = Matrix<Multi>;

    // Add the top-level documentation
    py::options options;
    options.disable_function_signatures();
    m.doc() = docstrings::starry::doc;

    // Declare the `Map` class (not user-facing)
    auto Map1 = bindMap<T1>(m, "Map1");
    auto Map2 = bindMap<T2>(m, "Map2");
    auto Map3 = bindMap<T3>(m, "Map3");
    auto Map4 = bindMap<T4>(m, "Map4");

    // User-facing class factory
    m.def("Map", [Map1, Map2, Map3, Map4]
                 (const int lmax=2, const int nwav=1, const bool multi=false) {
        if ((nwav == 1) && (!multi))
            return Map1(lmax, nwav);
        else if ((nwav == 1) && (multi))
            return Map2(lmax, nwav);
        else if ((nwav > 1) && (!multi))
            return Map3(lmax, nwav);
        else if ((nwav > 1) && (multi))
            return Map4(lmax, nwav);
        else
            throw errors::ValueError("Invalid argument(s) to `Map`.");
    }, docstrings::Map::doc, "lmax"_a=2, "nwav"_a=1, "multi"_a=false);

    // Declare the `kepler` module
    auto mk = m.def_submodule("kepler");

    // Declare the `Body` class (not user-facing)
    auto Body1 = bindBody<T1>(mk, Map1, "Body1");
    auto Body2 = bindBody<T2>(mk, Map2, "Body2");
    auto Body3 = bindBody<T3>(mk, Map3, "Body3");
    auto Body4 = bindBody<T4>(mk, Map4, "Body4");

    // Declare the `Primary` class (not user-facing)
    auto Primary1 = bindPrimary<T1>(mk, Body1, "Primary1");
    auto Primary2 = bindPrimary<T2>(mk, Body2, "Primary2");
    auto Primary3 = bindPrimary<T3>(mk, Body3, "Primary3");
    auto Primary4 = bindPrimary<T4>(mk, Body4, "Primary4");

    // User-facing class factory
    mk.def("Primary", [Primary1, Primary2, Primary3, Primary4]
            (const int lmax=2, const int nwav=1, const bool multi=false) {
        if ((nwav == 1) && (!multi))
            return Primary1(lmax, nwav);
        else if ((nwav == 1) && (multi))
            return Primary2(lmax, nwav);
        else if ((nwav > 1) && (!multi))
            return Primary3(lmax, nwav);
        else if ((nwav > 1) && (multi))
            return Primary4(lmax, nwav);
        else
            throw errors::ValueError("Invalid argument(s) to `Primary`.");
    }, docstrings::Primary::doc, "lmax"_a=2, "nwav"_a=1, "multi"_a=false);

    // Declare the `Secondary` class (not user-facing)
    auto Secondary1 = bindSecondary<T1>(mk, Body1, "Secondary1");
    auto Secondary2 = bindSecondary<T2>(mk, Body2, "Secondary2");
    auto Secondary3 = bindSecondary<T3>(mk, Body3, "Secondary3");
    auto Secondary4 = bindSecondary<T4>(mk, Body4, "Secondary4");

    // User-facing class factory
    mk.def("Secondary", [Secondary1, Secondary2, Secondary3, Secondary4]
            (const int lmax=2, const int nwav=1, const bool multi=false) {
        if ((nwav == 1) && (!multi))
            return Secondary1(lmax, nwav);
        else if ((nwav == 1) && (multi))
            return Secondary2(lmax, nwav);
        else if ((nwav > 1) && (!multi))
            return Secondary3(lmax, nwav);
        else if ((nwav > 1) && (multi))
            return Secondary4(lmax, nwav);
        else
            throw errors::ValueError("Invalid argument(s) to `Secondary`.");
    }, docstrings::Secondary::doc, "lmax"_a=2, "nwav"_a=1, "multi"_a=false);

    // Declare the `System` class (not user-facing)
    auto System1 = bindSystem<T1>(mk, "System1");
    auto System2 = bindSystem<T2>(mk, "System2");
    auto System3 = bindSystem<T3>(mk, "System3");
    auto System4 = bindSystem<T4>(mk, "System4");

    // User-facing class factories (one per type)
    // Note: We could probably template these!
    mk.def("System", [System1] (kepler::Primary<T1>& primary,
                                py::args secondaries) {
        if (secondaries.size() == 1) {
            kepler::Secondary<T1>& sec =
                py::cast<kepler::Secondary<T1>&>(secondaries[0]);
            return System1(&primary, &sec);
        } else {
            std::vector<kepler::Secondary<T1>*> sec;
            for (size_t n = 0; n < secondaries.size(); ++n)
                sec.push_back(py::cast<kepler::Secondary<T1>*>(secondaries[n]));
            return System1(&primary, sec);
        }
    }, docstrings::System::doc, "primary"_a);

    mk.def("System", [System2] (kepler::Primary<T2>& primary,
                                py::args secondaries) {
        if (secondaries.size() == 1) {
            kepler::Secondary<T2>& sec =
                py::cast<kepler::Secondary<T2>&>(secondaries[0]);
            return System2(&primary, &sec);
        } else {
            std::vector<kepler::Secondary<T2>*> sec;
            for (size_t n = 0; n < secondaries.size(); ++n)
                sec.push_back(py::cast<kepler::Secondary<T2>*>(secondaries[n]));
            return System2(&primary, sec);
        }
    }, docstrings::System::doc, "primary"_a);

    mk.def("System", [System3] (kepler::Primary<T3>& primary,
                                py::args secondaries) {
        if (secondaries.size() == 1) {
            kepler::Secondary<T3>& sec =
                py::cast<kepler::Secondary<T3>&>(secondaries[0]);
            return System3(&primary, &sec);
        } else {
            std::vector<kepler::Secondary<T3>*> sec;
            for (size_t n = 0; n < secondaries.size(); ++n)
                sec.push_back(py::cast<kepler::Secondary<T3>*>(secondaries[n]));
            return System3(&primary, sec);
        }
    }, docstrings::System::doc, "primary"_a);

    mk.def("System", [System4] (kepler::Primary<T4>& primary,
                                py::args secondaries) {
        if (secondaries.size() == 1) {
            kepler::Secondary<T4>& sec =
                py::cast<kepler::Secondary<T4>&>(secondaries[0]);
            return System4(&primary, &sec);
        } else {
            std::vector<kepler::Secondary<T4>*> sec;
            for (size_t n = 0; n < secondaries.size(); ++n)
                sec.push_back(py::cast<kepler::Secondary<T4>*>(secondaries[n]));
            return System4(&primary, sec);
        }
    }, docstrings::System::doc, "primary"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
