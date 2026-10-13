#include "state.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
PYBIND11_MODULE(_core, m) {
  m.doc() = "Original O(N+M) finite-window state map and Euclidean adjoint";
  py::class_<forcewindow::StateMap>(m, "StateMap")
      .def(py::init<std::array<double, 4>, std::array<double, 2>>())
      .def("forward", &forcewindow::StateMap::forward, py::arg("force"),
           py::arg("observations"))
      .def("adjoint", &forcewindow::StateMap::adjoint, py::arg("residual"), py::arg("forces"));
}
