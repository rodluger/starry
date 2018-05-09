/**
This defines the main Python interface to the code.
Note that this file is #include'd *twice*, once
for `starry` and once for `starry.grad` so we don't
have to duplicate any code.

*/

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include "maps.h"
#include "orbital.h"
#include "docstrings.h"
#include "vect.h"
#include "utils.h"
#include "errors.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;
using namespace vect;


/**
Define our map type (double or AutoDiffScalar)
and some other type-specific stuff.

*/
#ifndef STARRY_AUTODIFF

#undef MAPTYPE
#define MAPTYPE                         double
#undef DOCS
#define DOCS                            docstrings
#undef ADD_MODULE
#define ADD_MODULE                      add_starry

#else

#undef MAPTYPE
#define MAPTYPE                         Grad
#undef DOCS
#define DOCS                            docstrings_grad
#undef ADD_MODULE
#define ADD_MODULE                      add_starry_grad

#endif

/**
Instantiate the `starry` module with or without autodiff
*/
void ADD_MODULE(py::module &m) {

    m.doc() = DOCS::starry;

    // Surface map class
    py::class_<maps::Map<MAPTYPE>>(m, "Map", DOCS::Map::Map)

        .def(py::init<int>(), "lmax"_a=2)

        .def("__setitem__", [](maps::Map<MAPTYPE>& map, py::object index, double coeff) {
                if (py::isinstance<py::tuple>(index)) {
                    py::tuple lm = index;
                    int l = py::cast<int>(lm[0]);
                    int m = py::cast<int>(lm[1]);
                    map.set_coeff(l, m, MAPTYPE(coeff));
                } else {
                    throw errors::BadIndex();
                }
            })

        .def("__getitem__", [](maps::Map<MAPTYPE>& map, py::object index) -> py::object {
                if (py::isinstance<py::tuple>(index)) {
                    py::tuple lm = index;
                    int l = py::cast<int>(lm[0]);
                    int m = py::cast<int>(lm[1]);
                    return py::cast(get_value(map.get_coeff(l, m)));
                } else {
                    throw errors::BadIndex();
                }
            })

        .def("get_coeff", [](maps::Map<MAPTYPE> &map, int l, int m){
                return get_value(map.get_coeff(l, m));
            }, DOCS::Map::get_coeff, "l"_a, "m"_a)

        .def("set_coeff", [](maps::Map<MAPTYPE> &map, int l, int m, double coeff){
                map.set_coeff(l, m, MAPTYPE(coeff));
            }, DOCS::Map::set_coeff, "l"_a, "m"_a, "coeff"_a)

        .def("reset", &maps::Map<MAPTYPE>::reset, DOCS::Map::reset)

        .def_property_readonly("lmax", [](maps::Map<MAPTYPE> &map){return map.lmax;}, DOCS::Map::lmax)

        .def_property_readonly("y", [](maps::Map<MAPTYPE> &map){
                return get_value(map.y);
            }, DOCS::Map::y)

        .def_property_readonly("p", [](maps::Map<MAPTYPE> &map){
                return get_value(map.p);
            }, DOCS::Map::p)

        .def_property_readonly("g", [](maps::Map<MAPTYPE> &map){
                return get_value(map.g);
            }, DOCS::Map::g)

        .def_property_readonly("s", [](maps::Map<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.G.sT);
            }, DOCS::Map::s)

        .def_property_readonly("r", [](maps::Map<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.C.rT);
            }, DOCS::Map::r)

        .def_property("optimize", [](maps::Map<MAPTYPE> &map){return map.G.taylor;},
            [](maps::Map<MAPTYPE> &map, bool taylor){map.G.taylor = taylor;}, DOCS::Map::optimize)

        .def("evaluate", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& x, py::object& y) {
                return vectorize_map_evaluate(axis, theta, x, y, map);
            }, DOCS::Map::evaluate, "axis"_a=maps::yhat, "theta"_a=0, "x"_a=0, "y"_a=0)

        .def("flux", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize_map_flux(axis, theta, xo, yo, ro, map);
            }, DOCS::Map::flux, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("rotate", [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis, double theta){
                map.rotate(UnitVector<MAPTYPE>(axis), MAPTYPE(theta));
            }, DOCS::Map::rotate, "axis"_a=maps::yhat, "theta"_a=0)

        //
        // This is where things go nuts: Let's call Python from C++
        //

        .def("minimum", [](maps::Map<MAPTYPE> &map) -> double {
                py::object minimize = py::module::import("starry_maps").attr("minimize");
                Vector<double> p = get_value(map.p);
                return minimize(p).cast<double>();
            }, DOCS::Map::minimum)

        .def("load_image", [](maps::Map<MAPTYPE> &map, string& image) {
                py::object load_map = py::module::import("starry_maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
                double y_normed;
                int n = 0;
                for (int l = 0; l < map.lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        y_normed = y(n) / y(0);
                        map.set_coeff(l, m, MAPTYPE(y_normed));
                        n++;
                    }
                }
                // We need to apply some rotations to get
                // to the desired orientation
                UnitVector<MAPTYPE> xhat(maps::xhat);
                UnitVector<MAPTYPE> yhat(maps::yhat);
                UnitVector<MAPTYPE> zhat(maps::zhat);
                MAPTYPE Pi(M_PI);
                MAPTYPE PiOver2(M_PI / 2.);
                map.rotate(xhat, PiOver2);
                map.rotate(zhat, Pi);
                map.rotate(yhat, PiOver2);
            }, DOCS::Map::load_image, "image"_a)

        .def("load_healpix", [](maps::Map<MAPTYPE> &map, Matrix<double>& image) {
                py::object load_map = py::module::import("starry_maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
                double y_normed;
                int n = 0;
                for (int l = 0; l < map.lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        y_normed = y(n) / y(0);
                        map.set_coeff(l, m, MAPTYPE(y_normed));
                        n++;
                    }
                }
                // We need to apply some rotations to get
                // to the desired orientation
                UnitVector<MAPTYPE> xhat(maps::xhat);
                UnitVector<MAPTYPE> yhat(maps::yhat);
                UnitVector<MAPTYPE> zhat(maps::zhat);
                MAPTYPE Pi(M_PI);
                MAPTYPE PiOver2(M_PI / 2.);
                map.rotate(xhat, PiOver2);
                map.rotate(zhat, Pi);
                map.rotate(yhat, PiOver2);
            }, DOCS::Map::load_healpix, "image"_a)

        .def("show", [](maps::Map<MAPTYPE> &map, string cmap, int res) {
                py::object show = py::module::import("starry_maps").attr("show");
                Matrix<double> I;
                I.resize(res, res);
                Vector<double> x;
                UnitVector<MAPTYPE> yhat(maps::yhat);
                x = Vector<double>::LinSpaced(res, -1, 1);
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I(j, i) = get_value(map.evaluate(yhat, MAPTYPE(0), MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
                show(I, "cmap"_a=cmap, "res"_a=res);
            }, DOCS::Map::show, "cmap"_a="plasma", "res"_a=300)

        .def("animate", [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis, string cmap, int res, int frames) {
            std::cout << "Rendering animation..." << std::endl;
            py::object animate = py::module::import("starry_maps").attr("animate");
            vector<Matrix<double>> I;
            Vector<double> x, theta;
            x = Vector<double>::LinSpaced(res, -1, 1);
            theta = Vector<double>::LinSpaced(frames, 0, 2 * M_PI);
            UnitVector<MAPTYPE> MapType_axis(axis);
            for (int t = 0; t < frames; t++){
                I.push_back(Matrix<double>::Zero(res, res));
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I[t](j, i) = get_value(map.evaluate(axis, MAPTYPE(theta(t)), MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
            }
            animate(I, axis, "cmap"_a=cmap, "res"_a=res);
        }, DOCS::Map::animate, "axis"_a=maps::yhat, "cmap"_a="plasma", "res"_a=150, "frames"_a=50)

#ifndef STARRY_AUTODIFF

        // Methods and attributes only in `starry.Map()``

        .def_property_readonly("s_mp", [](maps::Map<MAPTYPE> &map){
                VectorT<double> sT = map.mpG.sT.template cast<double>();
                return sT;
            }, DOCS::Map::s_mp)

        .def("flux_mp", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize_map_flux_mp(axis, theta, xo, yo, ro, map);
            }, DOCS::Map::flux, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("flux_numerical", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro, double tol) {
                return vectorize_map_flux_numerical(axis, theta, xo, yo, ro, tol, map);
            }, DOCS::Map::flux_numerical, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4)

#else

        // Methods and attributes only in `starry.grad.Map()`

        .def_property_readonly("gradient", [](maps::Map<MAPTYPE> &map){
                return py::cast(map.derivs);
            }, DOCS::Map::gradient)

#endif

        .def("__repr__", [](maps::Map<MAPTYPE> &map) -> string {return map.repr();});


    // Limb-darkened surface map class
    py::class_<maps::LimbDarkenedMap<MAPTYPE>>(m, "LimbDarkenedMap", DOCS::LimbDarkenedMap::LimbDarkenedMap)

        .def(py::init<int>(), "lmax"_a=2)

        .def("__setitem__", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object index, double coeff) {
            if (py::isinstance<py::tuple>(index)) {
                throw errors::BadIndex();
            } else {
                int l = py::cast<int>(index);
                map.set_coeff(l, MAPTYPE(coeff));
            }
        })

        .def("__getitem__", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object index) {
            if (py::isinstance<py::tuple>(index)) {
                throw errors::BadIndex();
            } else {
                int l = py::cast<int>(index);
                return get_value(map.get_coeff(l));
            }
        })

        .def("get_coeff", [](maps::LimbDarkenedMap<MAPTYPE> &map, int l){
                return get_value(map.get_coeff(l));
            }, DOCS::LimbDarkenedMap::get_coeff, "l"_a)

        .def("set_coeff", [](maps::LimbDarkenedMap<MAPTYPE> &map, int l, double coeff){
                map.set_coeff(l, MAPTYPE(coeff));
            }, DOCS::LimbDarkenedMap::set_coeff, "l"_a, "coeff"_a)

        .def("reset", &maps::LimbDarkenedMap<MAPTYPE>::reset, DOCS::LimbDarkenedMap::reset)

        .def_property_readonly("lmax", [](maps::LimbDarkenedMap<MAPTYPE> &map){return map.lmax;}, DOCS::LimbDarkenedMap::lmax)

        .def_property_readonly("y", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.y);
            }, DOCS::LimbDarkenedMap::y)

        .def_property_readonly("p", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.p);
            }, DOCS::LimbDarkenedMap::p)

        .def_property_readonly("g", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.g);
            }, DOCS::LimbDarkenedMap::g)

        .def_property_readonly("s", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.G.sT);
            }, DOCS::LimbDarkenedMap::s)

        .def_property_readonly("u", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.u);
            }, DOCS::LimbDarkenedMap::u)

        .def_property("optimize", [](maps::LimbDarkenedMap<MAPTYPE> &map){return map.G.taylor;},
            [](maps::LimbDarkenedMap<MAPTYPE> &map, bool taylor){map.G.taylor = taylor;}, DOCS::LimbDarkenedMap::optimize)

        .def("evaluate", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& x, py::object& y) {
                return vectorize_ldmap_evaluate(x, y, map);
            }, DOCS::LimbDarkenedMap::evaluate, "x"_a=0, "y"_a=0)

        .def("flux", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize_ldmap_flux(xo, yo, ro, map);
            }, DOCS::LimbDarkenedMap::flux, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("show", [](maps::LimbDarkenedMap<MAPTYPE> &map, string cmap, int res) {
                py::object show = py::module::import("starry_maps").attr("show");
                Matrix<double> I;
                I.resize(res, res);
                Vector<double> x;
                UnitVector<MAPTYPE> yhat(maps::yhat);
                x = Vector<double>::LinSpaced(res, -1, 1);
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I(j, i) = get_value(map.evaluate(MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
                show(I, "cmap"_a=cmap, "res"_a=res);
            }, DOCS::LimbDarkenedMap::show, "cmap"_a="plasma", "res"_a=300)

#ifndef STARRY_AUTODIFF

        // Methods and attributes only in `starry.LimbDarkenedMap()``

        .def_property_readonly("s_mp", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                VectorT<double> sT = map.mpG.sT.template cast<double>();
                return sT;
            }, DOCS::LimbDarkenedMap::s_mp)

        .def("flux_mp", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize_ldmap_flux_mp(xo, yo, ro, map);
            }, DOCS::LimbDarkenedMap::flux, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("flux_numerical", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& xo, py::object& yo, py::object& ro, double tol) {
                return vectorize_ldmap_flux_numerical(xo, yo, ro, tol, map);
            }, DOCS::LimbDarkenedMap::flux_numerical, "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4)

#else

        // Methods and attributes only in `starry.grad.LimbDarkenedMap()`

#endif

        .def("__repr__", [](maps::LimbDarkenedMap<MAPTYPE> &map) -> string {return map.repr();});


    // Orbital system class
    py::class_<orbital::System<MAPTYPE>>(m, "System", DOCS::System::System)

        .def(py::init<vector<orbital::Body<MAPTYPE>*>, double, int>(),
            "bodies"_a, "kepler_tol"_a=1.0e-7, "kepler_max_iter"_a=100)

        .def("compute", [](orbital::System<MAPTYPE> &system, Vector<double>& time){


// DEBUG
#ifndef STARRY_AUTODIFF
                system.compute((Vector<MAPTYPE>)time);
#else
                Vector<MAPTYPE> time_g = time;
                for (int i = 0; i < time_g.size(); i++)
                    time_g(i).derivatives() = Vector<double>::Unit(1, 0);
                system.compute(time_g);
                for (int i = 0; i < time_g.size(); i++)
                    std::cout << system.flux(i).derivatives().transpose() << std::endl;
#endif


            }, DOCS::System::compute, "time"_a)

        // TODO: Return gradient as well?
        .def_property_readonly("flux", [](orbital::System<MAPTYPE> &system){return get_value(system.flux);},
            DOCS::System::flux);


     // Body class (not user-facing, just a base class)
     py::class_<orbital::Body<MAPTYPE>> PyBody(m, "Body");

     PyBody.def(py::init<int, const double&, const double&,
                         Eigen::Matrix<double, 3, 1>&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         bool>(),
                         "lmax"_a, "r"_a, "L"_a, "axis"_a,
                         "prot"_a, "theta0"_a, "a"_a, "porb"_a,
                         "inc"_a, "ecc"_a, "w"_a, "Omega"_a,
                         "lambda0"_a, "tref"_a, "is_star"_a)

        // NOTE: & is necessary in the return statement so we pass a reference back to Python!
        .def_property_readonly("map", [](orbital::Body<MAPTYPE> &body){return &body.map;}, DOCS::Body::map)

        // TODO: Return gradient as well?
        .def_property_readonly("flux", [](orbital::Body<MAPTYPE> &body){return get_value(body.flux);}, DOCS::Body::flux)

        // TODO: Return gradient as well?
        .def_property_readonly("x", [](orbital::Body<MAPTYPE> &body){return get_value(body.x) * AU;}, DOCS::Body::x)

        // TODO: Return gradient as well?
        .def_property_readonly("y", [](orbital::Body<MAPTYPE> &body){return get_value(body.y) * AU;}, DOCS::Body::y)

        // TODO: Return gradient as well?
        .def_property_readonly("z", [](orbital::Body<MAPTYPE> &body){return get_value(body.z) * AU;}, DOCS::Body::z)

        .def_property("r", [](orbital::Body<MAPTYPE> &body){return get_value(body.r);},
            [](orbital::Body<MAPTYPE> &body, double r){body.r = r;}, DOCS::Body::r)

        .def_property("L", [](orbital::Body<MAPTYPE> &body){return get_value(body.L);},
            [](orbital::Body<MAPTYPE> &body, double L){body.L = L; body.reset();}, DOCS::Body::L)

        .def_property("axis", [](orbital::Body<MAPTYPE> &body){return get_value((Vector<MAPTYPE>)body.axis);},
            [](orbital::Body<MAPTYPE> &body, UnitVector<double> axis){body.axis = (Vector<MAPTYPE>)axis;}, DOCS::Body::axis)

        .def_property("prot", [](orbital::Body<MAPTYPE> &body){return get_value(body.prot) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double prot){body.prot = prot * DAY; body.reset();}, DOCS::Body::prot)

        .def_property("theta0", [](orbital::Body<MAPTYPE> &body){return get_value(body.theta0) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double theta0){body.theta0 = theta0 * DEGREE;}, DOCS::Body::theta0)

        .def_property("a", [](orbital::Body<MAPTYPE> &body){return get_value(body.a);},
            [](orbital::Body<MAPTYPE> &body, double a){body.a = a;}, DOCS::Body::a)

        .def_property("porb", [](orbital::Body<MAPTYPE> &body){return get_value(body.porb) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double porb){body.porb = porb * DAY; body.reset();}, DOCS::Body::porb)

        .def_property("inc", [](orbital::Body<MAPTYPE> &body){return get_value(body.inc) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double inc){body.inc = inc * DEGREE; body.reset();}, DOCS::Body::inc)

        .def_property("ecc", [](orbital::Body<MAPTYPE> &body){return get_value(body.ecc);},
            [](orbital::Body<MAPTYPE> &body, double ecc){body.ecc = ecc; body.reset();}, DOCS::Body::ecc)

        .def_property("w", [](orbital::Body<MAPTYPE> &body){return get_value(body.w) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double w){body.w = w * DEGREE; body.reset();}, DOCS::Body::w)

        .def_property("Omega", [](orbital::Body<MAPTYPE> &body){return get_value(body.Omega) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double Omega){body.Omega = Omega * DEGREE; body.reset();}, DOCS::Body::Omega)

        .def_property("lambda0", [](orbital::Body<MAPTYPE> &body){return get_value(body.lambda0) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double lambda0){body.lambda0 = lambda0 * DEGREE; body.reset();}, DOCS::Body::lambda0)

        .def_property("tref", [](orbital::Body<MAPTYPE> &body){return get_value(body.tref) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double tref){body.tref = tref * DAY;}, DOCS::Body::tref)

        .def("__repr__", [](orbital::Body<MAPTYPE> &body) -> string {return body.repr();});


    // Star class
    py::class_<orbital::Star<MAPTYPE>>(m, "Star", PyBody, DOCS::Star::Star)

        .def(py::init<int>(), "lmax"_a=2)

        // NOTE: & is necessary in the return statement so we pass a reference back to Python!
        .def_property_readonly("map", [](orbital::Body<MAPTYPE> &body){return &body.ldmap;}, DOCS::Star::map)

        .def_property_readonly("r", [](orbital::Star<MAPTYPE> &star){return 1.;}, DOCS::Star::r)

        .def_property_readonly("L", [](orbital::Star<MAPTYPE> &star){return 1.;}, DOCS::Star::L)

        .def_property_readonly("axis", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("prot", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("theta0", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("a", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("porb", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("inc", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("ecc", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("w", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("Omega", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("lambda0", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("tref", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def("__setitem__", [](orbital::Star<MAPTYPE> &body, py::object index, double coeff) {
            if (py::isinstance<py::tuple>(index)) {
                throw errors::BadIndex();
            } else {
                int l = py::cast<int>(index);
                body.ldmap.set_coeff(l, MAPTYPE(coeff));
            }
        })

        .def("__getitem__", [](orbital::Star<MAPTYPE> &body, py::object index) {
            if (py::isinstance<py::tuple>(index)) {
                throw errors::BadIndex();
            } else {
                int l = py::cast<int>(index);
                return get_value(body.ldmap.get_coeff(l));
            }
        })

        .def("__repr__", [](orbital::Star<MAPTYPE> &star) -> string {return star.repr();});


    // Planet class
    py::class_<orbital::Planet<MAPTYPE>>(m, "Planet", PyBody, DOCS::Planet::Planet)

        .def(py::init<int, const double&, const double&,
                      Eigen::Matrix<double, 3, 1>&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&>(),
                      "lmax"_a=2, "r"_a=0.1, "L"_a=0., "axis"_a=maps::yhat,
                      "prot"_a=0, "theta0"_a=0, "a"_a=50., "porb"_a=1,
                      "inc"_a=90., "ecc"_a=0, "w"_a=90, "Omega"_a=0,
                      "lambda0"_a=90, "tref"_a=0)
        .def("__setitem__", [](orbital::Planet<MAPTYPE> &body, py::object index, double coeff) {
            if (py::isinstance<py::tuple>(index)) {
                // This is a (l, m) tuple
                py::tuple lm = index;
                int l = py::cast<int>(lm[0]);
                int m = py::cast<int>(lm[1]);
                body.map.set_coeff(l, m, MAPTYPE(coeff));
            } else {
                throw errors::BadIndex();
            }
        })
        .def("__getitem__", [](orbital::Planet<MAPTYPE> &body, py::object index) {
            if (py::isinstance<py::tuple>(index)) {
                // This is a (l, m) tuple
                py::tuple lm = index;
                int l = py::cast<int>(lm[0]);
                int m = py::cast<int>(lm[1]);
                return get_value(body.map.get_coeff(l, m));
            } else {
                throw errors::BadIndex();
            }
        })
        .def("__repr__", [](orbital::Planet<MAPTYPE> &planet) -> string {return planet.repr();});

} // ADD_MODULE
