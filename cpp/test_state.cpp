#include "state.hpp"
#include <iostream>
#include <limits>
#include <numeric>

int main() {
  using forcewindow::StateMap;
  StateMap s({1.0, 1.0, 0.0, 1.0}, {0.5, 1.0});
  const std::vector<double> f{2.0, 0.0, -1.0};
  auto y = s.forward(f, 5);
  const std::vector<double> expected{0.0, 1.0, 3.0, 4.5, 5.5};
  if (y != expected)
    return 1;
  for (std::size_t m : {1U, 2U, 3U, 4U, 8U}) {
    std::vector<double> r(m);
    for (std::size_t i = 0; i < m; ++i)
      r[i] = static_cast<double>(i) - 2.0;
    auto z = s.forward(f, m);
    auto g = s.adjoint(r, f.size());
    double lhs = std::inner_product(z.begin(), z.end(), r.begin(), 0.0);
    double rhs = std::inner_product(f.begin(), f.end(), g.begin(), 0.0);
    if (std::abs(lhs - rhs) > 1e-12)
      return 2;
  }
  if (s.forward({0.0, 0.0, 1.0}, 3) != std::vector<double>(3, 0.0))
    return 3;
  try {
    s.forward({}, 3);
    return 4;
  } catch (const std::invalid_argument &) {
  }
  try {
    s.adjoint({0.0}, 0);
    return 5;
  } catch (const std::invalid_argument &) {
  }
  try {
    s.forward({std::numeric_limits<double>::quiet_NaN()}, 1);
    return 6;
  } catch (const std::invalid_argument &) {
  }
  std::cout << "Forward, adjoint, causal nullspace and input checks passed\n";
}
