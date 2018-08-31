#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <stdlib.h>
#include "utils.h"
#include "pybind_interface.h"
#include "docstrings.h"
namespace py = pybind11;

PYBIND11_MODULE(_starry2, m) {

    using utils::Matrix;
    using utils::Vector;
    using utils::VectorT;
    using utils::Multi;
    using pybind_interface::bindMap;
    using pybind_interface::bindBody;
    using pybind_interface::bindPrimary;
    using pybind_interface::bindSecondary;
    using namespace pybind11::literals;

    // Add the top-level documentation
    py::options options;
    options.disable_function_signatures();
    m.doc() = docstrings::starry::doc;

    // Declare the Map class
    auto MapDoubleMono = bindMap<Vector<double>>(m, "MapDoubleMono");
    auto MapMultiMono = bindMap<Vector<Multi>>(m, "MapMultiMono");
    auto MapDoubleSpectral = bindMap<Matrix<double>>(m, "MapDoubleSpectral");
    auto MapMultiSpectral = bindMap<Matrix<Multi>>(m, "MapMultiSpectral");
    m.def("Map",
          [MapDoubleMono, MapMultiMono, MapDoubleSpectral, MapMultiSpectral]
          (const int lmax=2, const int nwav=1, const bool multi=false) {
        if ((nwav == 1) && (!multi)) {
            return MapDoubleMono(lmax, nwav);
        } else if ((nwav == 1) && (multi)) {
            return MapMultiMono(lmax, nwav);
        } else if ((nwav > 1) && (!multi)) {
            return MapDoubleSpectral(lmax, nwav);
        } else if ((nwav > 1) && (multi)) {
            return MapMultiSpectral(lmax, nwav);
        } else {
            throw errors::ValueError("Invalid argument(s) to `Map`.");
        }
    }, docstrings::Map::doc, "lmax"_a=2, "nwav"_a=1, "multi"_a=false);

    // Declare the `kepler` module
    auto mkepler = m.def_submodule("kepler");

    // Declare the Body class (not user-facing)
    auto BodyDoubleMono = bindBody<Vector<double>>(m, "BodyDoubleMono");
    auto BodyMultiMono = bindBody<Vector<Multi>>(m, "BodyMultiMono");
    auto BodyDoubleSpectral = bindBody<Matrix<double>>(m, "BodyDoubleSpectral");
    auto BodyMultiSpectral = bindBody<Matrix<Multi>>(m, "BodyMultiSpectral");

    // Declare the Primary class
    auto PrimaryDoubleMono = bindPrimary<Vector<double>>(mkepler,
        BodyDoubleMono, "PrimaryDoubleMono");
    auto PrimaryMultiMono = bindPrimary<Vector<Multi>>(mkepler,
        BodyMultiMono, "PrimaryMultiMono");
    auto PrimaryDoubleSpectral = bindPrimary<Matrix<double>>(mkepler,
        BodyDoubleSpectral, "PrimaryDoubleSpectral");
    auto PrimaryMultiSpectral = bindPrimary<Matrix<Multi>>(mkepler,
        BodyMultiSpectral, "PrimaryMultiSpectral");
    mkepler.def("Primary",
          [PrimaryDoubleMono, PrimaryMultiMono,
           PrimaryDoubleSpectral, PrimaryMultiSpectral]
          (const int lmax=2, const int nwav=1, const bool multi=false) {
        if ((nwav == 1) && (!multi)) {
            return PrimaryDoubleMono(lmax, nwav);
        } else if ((nwav == 1) && (multi)) {
            return PrimaryMultiMono(lmax, nwav);
        } else if ((nwav > 1) && (!multi)) {
            return PrimaryDoubleSpectral(lmax, nwav);
        } else if ((nwav > 1) && (multi)) {
            return PrimaryMultiSpectral(lmax, nwav);
        } else {
            throw errors::ValueError("Invalid argument(s) to `Primary`.");
        }
    }, "lmax"_a=2, "nwav"_a=1, "multi"_a=false);

    // Declare the Secondary class
    auto SecondaryDoubleMono = bindSecondary<Vector<double>>(mkepler,
        BodyDoubleMono, "SecondaryDoubleMono");
    auto SecondaryMultiMono = bindSecondary<Vector<Multi>>(mkepler,
        BodyMultiMono, "SecondaryMultiMono");
    auto SecondaryDoubleSpectral = bindSecondary<Matrix<double>>(mkepler,
        BodyDoubleSpectral, "SecondaryDoubleSpectral");
    auto SecondaryMultiSpectral = bindSecondary<Matrix<Multi>>(mkepler,
        BodyMultiSpectral, "SecondaryMultiSpectral");
    mkepler.def("Secondary",
          [SecondaryDoubleMono, SecondaryMultiMono,
           SecondaryDoubleSpectral, SecondaryMultiSpectral]
          (const int lmax=2, const int nwav=1, const bool multi=false) {
        if ((nwav == 1) && (!multi)) {
            return SecondaryDoubleMono(lmax, nwav);
        } else if ((nwav == 1) && (multi)) {
            return SecondaryMultiMono(lmax, nwav);
        } else if ((nwav > 1) && (!multi)) {
            return SecondaryDoubleSpectral(lmax, nwav);
        } else if ((nwav > 1) && (multi)) {
            return SecondaryMultiSpectral(lmax, nwav);
        } else {
            throw errors::ValueError("Invalid argument(s) to `Secondary`.");
        }
    }, "lmax"_a=2, "nwav"_a=1, "multi"_a=false);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
