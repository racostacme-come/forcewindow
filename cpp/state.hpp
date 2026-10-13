#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace forcewindow {
// A in row-major order; displacement is the first state coordinate.
class StateMap {
  std::array<double, 4> a_;
  std::array<double, 2> b_;

  static void finite(const std::vector<double> &v) {
    for (double x : v)
      if (!std::isfinite(x))
        throw std::invalid_argument("Inputs must be finite");
  }

public:
  StateMap(std::array<double, 4> a, std::array<double, 2> b) : a_(a), b_(b) {
    for (double x : a_)
      if (!std::isfinite(x))
        throw std::invalid_argument("A must be finite");
    for (double x : b_)
      if (!std::isfinite(x))
        throw std::invalid_argument("B must be finite");
  }

  std::vector<double> forward(const std::vector<double> &f, std::size_t observations) const {
    finite(f);
    if (f.empty() || observations == 0)
      throw std::invalid_argument("Nonempty force and observation window required");
    std::vector<double> y(observations);
    double q = 0.0, v = 0.0;
    for (std::size_t i = 0; i < observations; ++i) {
      y[i] = q;
      if (i + 1 == observations)
        break;
      const double u = i < f.size() ? f[i] : 0.0;
      const double next_q = a_[0] * q + a_[1] * v + b_[0] * u;
      v = a_[2] * q + a_[3] * v + b_[1] * u;
      q = next_q;
    }
    finite(y);
    return y;
  }

  std::vector<double> adjoint(const std::vector<double> &r, std::size_t forces) const {
    finite(r);
    if (r.empty() || forces == 0)
      throw std::invalid_argument("Nonempty residual and force window required");
    std::vector<double> g(forces, 0.0);
    double p = 0.0, w = 0.0;
    for (std::size_t i = r.size(); i-- > 0;) {
      if (i < forces)
        g[i] = b_[0] * p + b_[1] * w;
      const double next_p = a_[0] * p + a_[2] * w + r[i];
      w = a_[1] * p + a_[3] * w;
      p = next_p;
    }
    finite(g);
    return g;
  }
};
} // namespace forcewindow
