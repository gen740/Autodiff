#include "variable.hpp"

#include <gtest/gtest.h>

using Autodiff::Variable;

class AutoDiffFixture : public ::testing::Test {
protected:
  void SetUp() override {
    x.set({0, 0, 0}, 1.0);
    x.set({1, 0, 0}, 2.0);
    x.set({2, 0, 0}, 3.0);
    x.set({3, 0, 0}, 4.0);
    x.set({1, 1, 0}, 5.0);
    x.set({1, 2, 0}, 6.0);
    x.set({1, 3, 0}, 7.0);
    x.set({2, 2, 0}, 8.0);
    x.set({2, 3, 0}, 9.0);
    x.set({3, 3, 0}, 10.0);
    x.set({1, 1, 1}, 11.0);
    x.set({1, 1, 2}, 12.0);
    x.set({1, 1, 3}, 13.0);
    x.set({1, 2, 2}, 14.0);
    x.set({1, 2, 3}, 15.0);
    x.set({1, 3, 3}, 16.0);
    x.set({2, 2, 2}, 17.0);
    x.set({2, 2, 3}, 18.0);
    x.set({2, 3, 3}, 19.0);
    x.set({3, 3, 3}, 20.0);

    y.set({0, 0, 0}, 21.0);
    y.set({1, 0, 0}, 22.0);
    y.set({2, 0, 0}, 23.0);
    y.set({3, 0, 0}, 24.0);
    y.set({1, 1, 0}, 25.0);
    y.set({1, 2, 0}, 26.0);
    y.set({1, 3, 0}, 27.0);
    y.set({2, 2, 0}, 28.0);
    y.set({2, 3, 0}, 29.0);
    y.set({3, 3, 0}, 30.0);
    y.set({1, 1, 1}, 31.0);
    y.set({1, 1, 2}, 32.0);
    y.set({1, 1, 3}, 33.0);
    y.set({1, 2, 2}, 34.0);
    y.set({1, 2, 3}, 35.0);
    y.set({1, 3, 3}, 36.0);
    y.set({2, 2, 2}, 37.0);
    y.set({2, 2, 3}, 38.0);
    y.set({2, 3, 3}, 39.0);
    y.set({3, 3, 3}, 40.0);
  }

  Variable<3, 3> x;
  Variable<3, 3> y;
};

TEST_F(AutoDiffFixture, VariableMul) {
  EXPECT_NEAR((x * y).derivative(0, 0, 0), 21.0000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 0, 0), 64.0000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(2, 0, 0), 86.0000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(3, 0, 0), 108.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 1, 0), 218.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 2, 0), 264.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 3, 0), 310.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(2, 2, 0), 334.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(2, 3, 0), 382.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(3, 3, 0), 432.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 1, 1), 742.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 1, 2), 842.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 1, 3), 942.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 2, 2), 992.000000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 2, 3), 1096.00000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(1, 3, 3), 1204.00000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(2, 2, 2), 1198.00000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(2, 2, 3), 1308.00000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(2, 3, 3), 1422.00000000000, 1e-8);
  EXPECT_NEAR((x * y).derivative(3, 3, 3), 1540.00000000000, 1e-8);
};

TEST_F(AutoDiffFixture, VariableInv) {
  EXPECT_NEAR(x.inv().derivative(0, 0, 0), 1.00000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 0, 0), -2.00000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(2, 0, 0), -3.00000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(3, 0, 0), -4.00000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 1, 0), 3.00000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 2, 0), 6.00000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 3, 0), 9.00000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(2, 2, 0), 10.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(2, 3, 0), 15.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(3, 3, 0), 22.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 1, 1), 0.999999999999993, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 1, 2), -6.00000000000001, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 1, 3), -13.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 2, 2), -18.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 2, 3), -33.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(1, 3, 3), -56.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(2, 2, 2), -35.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(2, 2, 3), -62.0000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(2, 3, 3), -103.000000000000, 1e-8);
  EXPECT_NEAR(x.inv().derivative(3, 3, 3), -164.000000000000, 1e-8);
};

TEST_F(AutoDiffFixture, VariableSin) {
  EXPECT_NEAR(x.sin().derivative(0, 0, 0), 0.841470984807897, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 0, 0), 1.08060461173628, 1e-8);
  EXPECT_NEAR(x.sin().derivative(2, 0, 0), 1.62090691760442, 1e-8);
  EXPECT_NEAR(x.sin().derivative(3, 0, 0), 2.16120922347256, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 1, 0), -0.664372409890887, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 2, 0), -1.80701207363854, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 3, 0), -2.94965173738619, 1e-8);
  EXPECT_NEAR(x.sin().derivative(2, 2, 0), -3.25082041632595, 1e-8);
  EXPECT_NEAR(x.sin().derivative(2, 3, 0), -5.23493106488150, 1e-8);
  EXPECT_NEAR(x.sin().derivative(3, 3, 0), -8.06051269824495, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 1, 1), -23.6232226266325, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 1, 2), -32.8173684075080, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 1, 3), -42.0115141883835, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 2, 2), -45.9177004334832, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 2, 3), -57.8753927957107, 1e-8);
  EXPECT_NEAR(x.sin().derivative(1, 3, 3), -72.5966317392904, 1e-8);
  EXPECT_NEAR(x.sin().derivative(2, 2, 2), -65.9889339648499, 1e-8);
  EXPECT_NEAR(x.sin().derivative(2, 2, 3), -82.0919461991056, 1e-8);
  EXPECT_NEAR(x.sin().derivative(2, 3, 3), -101.498807320582, 1e-8);
  EXPECT_NEAR(x.sin().derivative(3, 3, 3), -124.749819635146, 1e-8);
};

TEST_F(AutoDiffFixture, VariableCos) {
  EXPECT_NEAR(x.cos().derivative(0, 0, 0), 0.540302305868140, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 0, 0), -1.68294196961579, 1e-8);
  EXPECT_NEAR(x.cos().derivative(2, 0, 0), -2.52441295442369, 1e-8);
  EXPECT_NEAR(x.cos().derivative(3, 0, 0), -3.36588393923159, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 1, 0), -6.36856414751204, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 2, 0), -8.29063974405622, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 3, 0), -10.2127153406004, 1e-8);
  EXPECT_NEAR(x.cos().derivative(2, 2, 0), -11.5944886312764, 1e-8);
  EXPECT_NEAR(x.cos().derivative(2, 3, 0), -14.0568665336887, 1e-8);
  EXPECT_NEAR(x.cos().derivative(3, 3, 0), -17.0595467419692, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 1, 1), -18.7334821304679, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 1, 2), -21.0717899288574, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 1, 3), -23.4100977272470, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 2, 2), -24.7298359659117, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 2, 3), -26.4658064064217, 1e-8);
  EXPECT_NEAR(x.cos().derivative(1, 3, 3), -27.5994394890523, 1e-8);
  EXPECT_NEAR(x.cos().derivative(2, 2, 2), -30.4870561744271, 1e-8);
  EXPECT_NEAR(x.cos().derivative(2, 2, 3), -31.3195205781179, 1e-8);
  EXPECT_NEAR(x.cos().derivative(2, 3, 3), -30.7081766391213, 1e-8);
  EXPECT_NEAR(x.cos().derivative(3, 3, 3), -27.8115533726293, 1e-8);
};

TEST_F(AutoDiffFixture, VariableTan) {
  EXPECT_NEAR(x.tan().derivative(0, 0, 0), 1.55740772465490, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 0, 0), 6.85103764162952, 1e-8);
  EXPECT_NEAR(x.tan().derivative(2, 0, 0), 10.2765564624443, 1e-8);
  EXPECT_NEAR(x.tan().derivative(3, 0, 0), 13.7020752832590, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 1, 0), 59.8070298839751, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 2, 0), 84.5722665947405, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 3, 0), 109.337503305506, 1e-8);
  EXPECT_NEAR(x.tan().derivative(2, 2, 0), 123.432881071296, 1e-8);
  EXPECT_NEAR(x.tan().derivative(2, 3, 0), 158.867976727037, 1e-8);
  EXPECT_NEAR(x.tan().derivative(3, 3, 0), 204.972931327753, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 1, 1), 811.400474316445, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 1, 2), 1137.66672311115, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 1, 3), 1463.93297190585, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 2, 2), 1623.44392624112, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 2, 3), 2084.45589266033, 1e-8);
  EXPECT_NEAR(x.tan().derivative(1, 3, 3), 2680.21357670405, 1e-8);
  EXPECT_NEAR(x.tan().derivative(2, 2, 2), 2357.44466040857, 1e-8);
  EXPECT_NEAR(x.tan().derivative(2, 2, 3), 3020.57520326454, 1e-8);
  EXPECT_NEAR(x.tan().derivative(2, 3, 3), 3875.15446361230, 1e-8);
  EXPECT_NEAR(x.tan().derivative(3, 3, 3), 4977.88544131911, 1e-8);
};

TEST_F(AutoDiffFixture, VariableExp) {
  EXPECT_NEAR(x.exp().derivative(0, 0, 0), 2.71828182845905, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 0, 0), 5.43656365691809, 1e-8);
  EXPECT_NEAR(x.exp().derivative(2, 0, 0), 8.15484548537714, 1e-8);
  EXPECT_NEAR(x.exp().derivative(3, 0, 0), 10.8731273138362, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 1, 0), 24.4645364561314, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 2, 0), 32.6193819415085, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 3, 0), 40.7742274268857, 1e-8);
  EXPECT_NEAR(x.exp().derivative(2, 2, 0), 46.2107910838038, 1e-8);
  EXPECT_NEAR(x.exp().derivative(2, 3, 0), 57.0839183976399, 1e-8);
  EXPECT_NEAR(x.exp().derivative(3, 3, 0), 70.6753275399352, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 1, 1), 133.195809594493, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 1, 2), 171.251755192920, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 1, 3), 209.307700791346, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 2, 2), 228.335673590560, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 2, 3), 277.264746502823, 1e-8);
  EXPECT_NEAR(x.exp().derivative(1, 3, 3), 337.066946728922, 1e-8);
  EXPECT_NEAR(x.exp().derivative(2, 2, 2), 315.320692101249, 1e-8);
  EXPECT_NEAR(x.exp().derivative(2, 2, 3), 380.559455984266, 1e-8);
  EXPECT_NEAR(x.exp().derivative(2, 3, 3), 459.389629009579, 1e-8);
  EXPECT_NEAR(x.exp().derivative(3, 3, 3), 554.529493005645, 1e-8);
};

TEST_F(AutoDiffFixture, VariableLog) {
  EXPECT_NEAR(x.log().derivative(0, 0, 0), 0., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 0, 0), 2., 1e-8);
  EXPECT_NEAR(x.log().derivative(2, 0, 0), 3., 1e-8);
  EXPECT_NEAR(x.log().derivative(3, 0, 0), 4., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 1, 0), 1., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 2, 0), 0., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 3, 0), -1., 1e-8);
  EXPECT_NEAR(x.log().derivative(2, 2, 0), -1., 1e-8);
  EXPECT_NEAR(x.log().derivative(2, 3, 0), -3., 1e-8);
  EXPECT_NEAR(x.log().derivative(3, 3, 0), -6., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 1, 1), -3., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 1, 2), -3., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 1, 3), -3., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 2, 2), -2., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 2, 3), 0., 1e-8);
  EXPECT_NEAR(x.log().derivative(1, 3, 3), 4., 1e-8);
  EXPECT_NEAR(x.log().derivative(2, 2, 2), -1., 1e-8);
  EXPECT_NEAR(x.log().derivative(2, 2, 3), 4., 1e-8);
  EXPECT_NEAR(x.log().derivative(2, 3, 3), 13., 1e-8);
  EXPECT_NEAR(x.log().derivative(3, 3, 3), 28., 1e-8);
};

TEST_F(AutoDiffFixture, VariablePow) {
  EXPECT_NEAR(x.pow(1.4).derivative(0, 0, 0), 1.00000000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 0, 0), 2.80000000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(2, 0, 0), 4.20000000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(3, 0, 0), 5.60000000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 1, 0), 9.24000000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 2, 0), 11.7600000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 3, 0), 14.2800000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(2, 2, 0), 16.2400000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(2, 3, 0), 19.3200000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(3, 3, 0), 22.9600000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 1, 1), 29.5120000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 1, 2), 34.6080000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 1, 3), 39.7040000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 2, 2), 42.6720000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 2, 3), 48.2160000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(1, 3, 3), 54.2080000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(2, 2, 2), 55.0480000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(2, 2, 3), 61.2640000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(2, 3, 3), 67.5920000000000, 1e-8);
  EXPECT_NEAR(x.pow(1.4).derivative(3, 3, 3), 73.6960000000000, 1e-8);
};

TEST_F(AutoDiffFixture, VariableSqrt) {
  EXPECT_NEAR(x.sqrt().derivative(0, 0, 0), 1.0000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 0, 0), 1.0000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(2, 0, 0), 1.5000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(3, 0, 0), 2.0000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 1, 0), 1.5000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 2, 0), 1.5000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 3, 0), 1.5000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(2, 2, 0), 1.7500000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(2, 3, 0), 1.5000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(3, 3, 0), 1.0000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 1, 1), 1.0000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 1, 2), 0.7500000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 1, 3), 0.5000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 2, 2), 0.7500000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 2, 3), 0.7500000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(1, 3, 3), 1.0000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(2, 2, 2), 0.6250000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(2, 2, 3), 1.0000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(2, 3, 3), 2.0000000000000, 1e-8);
  EXPECT_NEAR(x.sqrt().derivative(3, 3, 3), 4.0000000000000, 1e-8);
};

TEST_F(AutoDiffFixture, VariableCbrt) {
  EXPECT_NEAR(x.cbrt().derivative(0, 0, 0), 1.00000000000000, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 0, 0), 0.666666666666667, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(2, 0, 0), 1.00000000000000, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(3, 0, 0), 1.33333333333333, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 1, 0), 0.777777777777778, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 2, 0), 0.666666666666667, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 3, 0), 0.555555555555556, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(2, 2, 0), 0.666666666666666, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(2, 3, 0), 0.333333333333333, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(3, 3, 0), -0.222222222222222, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 1, 1), -0.0370370370370381, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 1, 2), -0.222222222222224, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 1, 3), -0.407407407407410, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 2, 2), -0.222222222222226, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 2, 3), -0.111111111111114, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(1, 3, 3), 0.296296296296291, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(2, 2, 2), -0.333333333333336, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(2, 2, 3), 0.222222222222220, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(2, 3, 3), 1.44444444444444, 1e-8);
  EXPECT_NEAR(x.cbrt().derivative(3, 3, 3), 3.70370370370369, 1e-8);
}
