#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <cmath>
#include "constants.h"
#include "ellip.h"
#include "maps.h"
#include "basis.h"
#include "fact.h"
#include "sqrtint.h"
#include "rotation.h"
#include "solver.h"
#include "orbital.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;
using std::vector;


// TODO: Ensure we are passing the flux back to Python by reference!
//PYBIND11_MAKE_OPAQUE(vector<orbital::Body<double>*>);


PYBIND11_MODULE(starry, m) {
    m.doc() = R"pbdoc(
        API
        ---
        .. currentmodule:: starry

        .. autosummary::
            :toctree: _generate

            Body
            Star
            Planet
            Map

    )pbdoc";

    // Orbital system class
    py::class_<orbital::System<double>>(m, "System")
        .def(py::init<vector<orbital::Body<double>*>, double, int>(),
            R"pbdoc(
                Instantiate an orbital system.
            )pbdoc", "bodies"_a, "kepler_tol"_a=1.0e-7, "kepler_max_iter"_a=100)
        .def("compute", &orbital::System<double>::compute,
            R"pbdoc(
                Compute the light curve.
            )pbdoc", "time"_a)
        .def_property_readonly("flux", [](orbital::System<double> &system){return system.flux;},
            R"pbdoc(
                The computed light curve.
            )pbdoc");

     // Body class
     py::class_<orbital::Body<double>> PyBody(m, "Body");
     PyBody.def(py::init<int, const double&, const double&,
                         Eigen::Matrix<double, 3, 1>&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&>(),
             R"pbdoc(
                 Instantiate a body.
             )pbdoc", "lmax"_a, "r"_a, "L"_a, "u"_a,
                      "prot"_a, "theta0"_a, "m"_a, "porb"_a,
                      "inc"_a, "ecc"_a, "w"_a, "Omega"_a,
                      "lambda0"_a, "tref"_a, "UNIT_RADIUS"_a,
                      "UNIT_MASS"_a, "UNIT_LUMINOSITY"_a)
        .def_property_readonly("map", [](orbital::Body<double> &body){return &body.map;},
            R"pbdoc(
                The body's surface map.
            )pbdoc")
        .def_property_readonly("flux", [](orbital::Body<double> &body){return &body.flux;},
            R"pbdoc(
                The body's computed light curve.
            )pbdoc")
        .def_property_readonly("x", [](orbital::Body<double> &body){return body.x * AU;},
            R"pbdoc(
                The x position of the body.
            )pbdoc")
        .def_property_readonly("y", [](orbital::Body<double> &body){return body.y * AU;},
            R"pbdoc(
                The y position of the body.
            )pbdoc")
        .def_property_readonly("z", [](orbital::Body<double> &body){return body.z * AU;},
            R"pbdoc(
                The z position of the body.
            )pbdoc")
        .def_property("r", [](orbital::Body<double> &body){return body.r / body.UNIT_RADIUS;},
                           [](orbital::Body<double> &body, double r){body.r = r * body.UNIT_RADIUS; body.reset();})
        .def_property("L", [](orbital::Body<double> &body){return body.L / body.UNIT_LUMINOSITY;},
                           [](orbital::Body<double> &body, double L){body.L = L * body.UNIT_LUMINOSITY; body.reset();})
        .def_property("u", [](orbital::Body<double> &body){return body.u;},
                           [](orbital::Body<double> &body, UnitVector<double> u){body.u = u;})
        .def_property("prot", [](orbital::Body<double> &body){return body.prot / DAY;},
                              [](orbital::Body<double> &body, double prot){body.prot = prot * DAY;})
        .def_property("theta0", [](orbital::Body<double> &body){return body.theta0 / DEGREE;},
                                [](orbital::Body<double> &body, double theta0){body.theta0 = theta0 * DEGREE;})
        .def_property("m", [](orbital::Body<double> &body){return body.m / body.UNIT_MASS;},
                           [](orbital::Body<double> &body, double m){body.m = m * body.UNIT_MASS;})
        .def_property("porb", [](orbital::Body<double> &body){return body.porb / DAY;},
                              [](orbital::Body<double> &body, double porb){body.porb = porb * DAY;})
        .def_property("inc", [](orbital::Body<double> &body){return body.inc / DEGREE;},
                             [](orbital::Body<double> &body, double inc){body.inc = inc * DEGREE; body.reset();})
        .def_property("ecc", [](orbital::Body<double> &body){return body.ecc;},
                             [](orbital::Body<double> &body, double ecc){body.ecc = ecc; body.reset();})
        .def_property("w", [](orbital::Body<double> &body){return body.w / DEGREE;},
                           [](orbital::Body<double> &body, double w){body.w = w * DEGREE; body.reset();})
        .def_property("Omega", [](orbital::Body<double> &body){return body.Omega / DEGREE;},
                               [](orbital::Body<double> &body, double Omega){body.Omega = Omega * DEGREE; body.reset();})
        .def_property("lambda0", [](orbital::Body<double> &body){return body.lambda0 / DEGREE;},
                                 [](orbital::Body<double> &body, double lambda0){body.lambda0 = lambda0 * DEGREE; body.reset();})
        .def_property("tref", [](orbital::Body<double> &body){return body.tref / DAY;},
                              [](orbital::Body<double> &body, double tref){body.tref = tref * DAY;})
        .def("__repr__", [](orbital::Body<double> &body) -> string {return body.repr();});

    // Star class
    py::class_<orbital::Star<double>>(m, "Star", PyBody)
        .def(py::init<const double&, const double&, const double&>(),
            R"pbdoc(
                Instantiate a star.
            )pbdoc", "r"_a=1, "L"_a=1, "m"_a=1)
        .def("__repr__", [](orbital::Star<double> &star) -> string {return star.repr();});

    // Planet class
    py::class_<orbital::Planet<double>>(m, "Planet", PyBody)
        .def(py::init<int, const double&, const double&,
                      Eigen::Matrix<double, 3, 1>&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&>(),
            R"pbdoc(
                Instantiate a planet.
            )pbdoc", "lmax"_a=2, "r"_a=1, "L"_a=1.e-9, "u"_a=maps::yhat,
                     "prot"_a=1, "theta0"_a=0, "porb"_a=1,
                     "inc"_a=90., "ecc"_a=0, "w"_a=0, "Omega"_a=0,
                     "lambda0"_a=0, "tref"_a=0)
        .def("__repr__", [](orbital::Planet<double> &planet) -> string {return planet.repr();});

    // Surface map class
    py::class_<maps::Map<double>>(m, "Map")
        .def(py::init<int>(),
            R"pbdoc(
                Instantiate a starry map.
            )pbdoc", "lmax"_a=2)
        .def_property("use_mp", [](maps::Map<double> &map){return map.use_mp;},
                                [](maps::Map<double> &map, bool use_mp){map.use_mp = use_mp;})
        .def("evaluate", py::vectorize(&maps::Map<double>::evaluate),
            R"pbdoc(
                Return the specific intensity at a point on the map.
            )pbdoc", "u"_a=maps::yhat, "theta"_a=0, "x"_a=0, "y"_a=0)
        .def("rotate", [](maps::Map<double> &map, UnitVector<double>& u,
                          double theta){return map.rotate(u, theta);},
            R"pbdoc(
                Rotate the base map an angle `theta` about `u`.
            )pbdoc", "u"_a=maps::yhat, "theta"_a=0)
        .def("flux", py::vectorize(&maps::Map<double>::flux),
            R"pbdoc(
                Return the total flux received by the observer.
            )pbdoc", "u"_a=maps::yhat, "theta"_a=0, "xo"_a=0,
                     "yo"_a=0, "ro"_a=0, "numerical"_a=false,
                     "tol"_a=1e-4)
        .def("get_coeff", &maps::Map<double>::get_coeff,
            R"pbdoc(
                Get the (l, m) coefficient of the map.
            )pbdoc", "l"_a, "m"_a)
        .def("set_coeff", &maps::Map<double>::set_coeff,
            R"pbdoc(
                Set the (l, m) coefficient of the map.
            )pbdoc", "l"_a, "m"_a, "coeff"_a)
        .def("reset", &maps::Map<double>::reset,
            R"pbdoc(
                Reset the map coefficients.
            )pbdoc")
        .def("limbdark", &maps::Map<double>::limbdark,
            R"pbdoc(
                Set the linear and quadratic limb darkening coefficients.
                Note that this will overwrite all existing coefficients.
            )pbdoc", "u1"_a, "u2"_a)
        .def_property_readonly("lmax", [](maps::Map<double> &map){return map.lmax;},
            R"pbdoc(
                The highest spherical harmonic order of the map.
            )pbdoc")
        .def_property_readonly("y", [](maps::Map<double> &map){map.update(true); return map.y;},
            R"pbdoc(
                The spherical harmonic map vector.
            )pbdoc")
        .def_property_readonly("p", [](maps::Map<double> &map){map.update(true); return map.p;},
            R"pbdoc(
                The polynomial map vector.
            )pbdoc")
        .def_property_readonly("g", [](maps::Map<double> &map){map.update(true); return map.g;},
            R"pbdoc(
                The Green's polynomial map vector.
            )pbdoc")
        .def("__setitem__", [](maps::Map<double>& map, vector<int> lm, double coeff){
            if (lm.size() == 1) {
                int l = (int)floor(sqrt(lm[0]));
                int m = lm[0] - l * l - l;
                map.set_coeff(l, m, coeff);
            } else if (lm.size() == 2) {
                map.set_coeff(lm[0], lm[1], coeff);
            } else {
                std::cout << "ERROR: Invalid spherical harmonic index." << std::endl;
            }
        })
        .def("__getitem__", [](maps::Map<double>& map, vector<int> lm){
            if (lm.size() == 1) {
                int l = (int)floor(sqrt(lm[0]));
                int m = lm[0] - l * l - l;
                return map.get_coeff(l, m);
            } else if (lm.size() == 2) {
                return map.get_coeff(lm[0], lm[1]);
            } else {
                std::cout << "ERROR: Invalid spherical harmonic index." << std::endl;
                return double(0.);
            }
        })
        .def("__repr__", [](maps::Map<double> &map) -> string {return map.repr();})
        //
        // This is where things go nuts: Let's call Python from C++
        //
        .def("load_image", [](maps::Map<double> &map, string& image) {
            py::object load_map = py::module::import("starry_maps").attr("load_map");
            Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
            int n = 0;
            for (int l = 0; l < map.lmax + 1; l++) {
                for (int m = -l; m < l + 1; m++) {
                    map.set_coeff(l, m, y(n));
                    n++;
                }
            }
            // We need to apply some rotations to get
            // to the desired orientation
            map.rotate(maps::xhat, M_PI / 2.);
            map.rotate(maps::zhat, M_PI);
            map.rotate(maps::yhat, M_PI / 2);
        })
        .def("load_healpix", [](maps::Map<double> &map, Matrix<double>& image) {
            py::object load_map = py::module::import("starry_maps").attr("load_map");
            Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
            int n = 0;
            for (int l = 0; l < map.lmax + 1; l++) {
                for (int m = -l; m < l + 1; m++) {
                    map.set_coeff(l, m, y(n));
                    n++;
                }
            }
            // We need to apply some rotations to get
            // to the desired orientation
            map.rotate(maps::xhat, M_PI / 2.);
            map.rotate(maps::zhat, M_PI);
            map.rotate(maps::yhat, M_PI / 2);
        })
        .def("show", [](maps::Map<double> &map, string cmap, int res) {
            py::object show = py::module::import("starry_maps").attr("show");
            Matrix<double> I;
            I.resize(res, res);
            Vector<double> x;
            x = Vector<double>::LinSpaced(res, -1, 1);
            for (int i = 0; i < res; i++){
                for (int j = 0; j < res; j++){
                    I(j, i) = map.evaluate(maps::yhat, 0, x(i), x(j));
                }
            }
            show(I, "cmap"_a=cmap, "res"_a=res);
        }, "cmap"_a="plasma", "res"_a=300)
        .def("animate", [](maps::Map<double> &map, UnitVector<double>& u, string cmap, int res, int frames) {
            std::cout << "Rendering animation..." << std::endl;
            py::object animate = py::module::import("starry_maps").attr("animate");
            vector<Matrix<double>> I;
            Vector<double> x, theta;
            x = Vector<double>::LinSpaced(res, -1, 1);
            theta = Vector<double>::LinSpaced(frames, 0, 2 * M_PI);
            for (int t = 0; t < frames; t++){
                I.push_back(Matrix<double>::Zero(res, res));
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I[t](j, i) = map.evaluate(u, theta(t), x(i), x(j));
                    }
                }
            }
            animate(I, u, "cmap"_a=cmap, "res"_a=res);
        }, "u"_a=maps::yhat, "cmap"_a="plasma", "res"_a=150, "frames"_a=50);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
