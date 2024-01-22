#include "single_variable.hpp"

#include <gtest/gtest.h>

using Autodiff::SingleVariable;

static auto x = SingleVariable<5, double>(0.0);
static auto y = 2.0                          //
                + 3.0 * x                    //
                + 5.0 * x * x / 2.           //
                + 7.0 * x * x * x / 6.       //
                + 13.0 * x * x * x * x / 24. //
                + 17.0 * x * x * x * x * x / 120.;

TEST(autodiff, SingleVariable) {
  EXPECT_NEAR((x).derivative(0), 0., 1e-8);
  EXPECT_NEAR((x).derivative(1), 1., 1e-8);
  EXPECT_NEAR((x).derivative(2), 0., 1e-8);
  EXPECT_NEAR((x).derivative(3), 0., 1e-8);
  EXPECT_NEAR((x).derivative(4), 0., 1e-8);
  EXPECT_NEAR((x).derivative(5), 0., 1e-8);

  EXPECT_NEAR((y).derivative(0), 2., 1e-8);
  EXPECT_NEAR((y).derivative(1), 3., 1e-8);
  EXPECT_NEAR((y).derivative(2), 5., 1e-8);
  EXPECT_NEAR((y).derivative(3), 7., 1e-8);
  EXPECT_NEAR((y).derivative(4), 13., 1e-8);
  EXPECT_NEAR((y).derivative(5), 17., 1e-8);
}

TEST(autodiff, SingleVariableMul) {
  EXPECT_NEAR((y * y).derivative(0), 4.0000000000000, 1e-8);
  EXPECT_NEAR((y * y).derivative(1), 12.000000000000, 1e-8);
  EXPECT_NEAR((y * y).derivative(2), 38.000000000000, 1e-8);
  EXPECT_NEAR((y * y).derivative(3), 118.00000000000, 1e-8);
  EXPECT_NEAR((y * y).derivative(4), 370.00000000000, 1e-8);
  EXPECT_NEAR((y * y).derivative(5), 1158.0000000000, 1e-8);
}

TEST(autodiff, SingleVariableDiv) {
  EXPECT_NEAR((y / y).derivative(0), 1., 1e-8);
  EXPECT_NEAR((y / y).derivative(1), 0., 1e-8);
  EXPECT_NEAR((y / y).derivative(2), 0., 1e-8);
  EXPECT_NEAR((y / y).derivative(3), 0., 1e-8);
  EXPECT_NEAR((y / y).derivative(4), 0., 1e-8);
  EXPECT_NEAR((y / y).derivative(5), 0., 1e-8);

  EXPECT_NEAR((1. / y).derivative(0), 0.500000000000, 1e-8);
  EXPECT_NEAR((1. / y).derivative(1), -0.750000000000, 1e-8);
  EXPECT_NEAR((1. / y).derivative(2), 1.000000000000, 1e-8);
  EXPECT_NEAR((1. / y).derivative(3), -0.625000000000, 1e-8);
  EXPECT_NEAR((1. / y).derivative(4), -4.000000000000, 1e-8);
  EXPECT_NEAR((1. / y).derivative(5), 30.750000000000, 1e-8);
}

TEST(autodiff, SingleVariablePow) {
  EXPECT_NEAR((y.pow(1.4)).derivative(0), 2.639015821546, 1e-8);
  EXPECT_NEAR((y.pow(1.4)).derivative(1), 5.541933225246, 1e-8);
  EXPECT_NEAR((y.pow(1.4)).derivative(2), 12.561715310558, 1e-8);
  EXPECT_NEAR((y.pow(1.4)).derivative(3), 26.564333259680, 1e-8);
  EXPECT_NEAR((y.pow(1.4)).derivative(4), 60.015442207266, 1e-8);
  EXPECT_NEAR((y.pow(1.4)).derivative(5), 129.933949625831, 1e-8);
}

TEST(autodiff, SingleVariableSqrt) {
  EXPECT_NEAR((y.sqrt()).derivative(0), 1.414213562373, 1e-8);
  EXPECT_NEAR((y.sqrt()).derivative(1), 1.060660171780, 1e-8);
  EXPECT_NEAR((y.sqrt()).derivative(2), 0.972271824132, 1e-8);
  EXPECT_NEAR((y.sqrt()).derivative(3), 0.287262129857, 1e-8);
  EXPECT_NEAR((y.sqrt()).derivative(4), 1.729097050870, 1e-8);
  EXPECT_NEAR((y.sqrt()).derivative(5), -2.448633443445, 1e-8);
}

TEST(autodiff, SingleVariableExp) {
  EXPECT_NEAR((y.exp()).derivative(0), 7.389056098931, 1e-8);
  EXPECT_NEAR((y.exp()).derivative(1), 22.167168296792, 1e-8);
  EXPECT_NEAR((y.exp()).derivative(2), 103.446785385029, 1e-8);
  EXPECT_NEAR((y.exp()).derivative(3), 583.735431815521, 1e-8);
  EXPECT_NEAR((y.exp()).derivative(4), 3864.476339740730, 1e-8);
  EXPECT_NEAR((y.exp()).derivative(5), 28891.209346818800, 1e-8);
}

TEST(autodiff, SingleVariableLog) {
  EXPECT_NEAR((y.log()).derivative(0), 0.693147180560, 1e-8);
  EXPECT_NEAR((y.log()).derivative(1), 1.500000000000, 1e-8);
  EXPECT_NEAR((y.log()).derivative(2), 0.250000000000, 1e-8);
  EXPECT_NEAR((y.log()).derivative(3), -1.000000000000, 1e-8);
  EXPECT_NEAR((y.log()).derivative(4), 3.875000000000, 1e-8);
  EXPECT_NEAR((y.log()).derivative(5), -13.000000000000, 1e-8);
}

TEST(autodiff, SingleVariableSin) {
  EXPECT_NEAR((y.sin()).derivative(0), 0.909297426826, 1e-8);
  EXPECT_NEAR((y.sin()).derivative(1), -1.248440509641, 1e-8);
  EXPECT_NEAR((y.sin()).derivative(2), -10.264411024167, 1e-8);
  EXPECT_NEAR((y.sin()).derivative(3), -32.595447476213, 1e-8);
  EXPECT_NEAR((y.sin()).derivative(4), 36.024537700212, 1e-8);
  EXPECT_NEAR((y.sin()).derivative(5), 1354.123949232650, 1e-8);
}

TEST(autodiff, SingleVariableCos) {
  EXPECT_NEAR((y.cos()).derivative(0), -0.416146836547, 1e-8);
  EXPECT_NEAR((y.cos()).derivative(1), -2.727892280477, 1e-8);
  EXPECT_NEAR((y.cos()).derivative(2), -0.801165605204, 1e-8);
  EXPECT_NEAR((y.cos()).derivative(3), 36.912556181135, 1e-8);
  EXPECT_NEAR((y.cos()).derivative(4), 266.148891944877, 1e-8);
  EXPECT_NEAR((y.cos()).derivative(5), 1024.401449683940, 1e-8);
}

TEST(autodiff, SingleVariableTan) {
  EXPECT_NEAR((y.tan()).derivative(0), -2.185039863262, 1e-8);
  EXPECT_NEAR((y.tan()).derivative(1), 17.323197612126, 1e-8);
  EXPECT_NEAR((y.tan()).derivative(2), -198.239268029700, 1e-8);
  EXPECT_NEAR((y.tan()).derivative(3), 3682.906519299890, 1e-8);
  EXPECT_NEAR((y.tan()).derivative(4), -89615.364906299300, 1e-8);
  EXPECT_NEAR((y.tan()).derivative(5), 2737217.050492670000, 1e-8);
}
