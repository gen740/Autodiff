#pragma once

#include <array>
#include <vector>

namespace Autodiff {

// autodiff::variable  で用いる定数を記述する

// ```python
// def foo(n: int, depth: int | None = None) -> Generator[list[int], None,
// None]:
//    depth = depth if depth is not None else n
//    if depth == 0:
//        yield []
//        return
//
//    for i in foo(n, depth - 1):
//        k = copy.deepcopy(i)
//        k.append(depth)
//        yield sorted(k, reverse=True)
//        k = copy.deepcopy(i)
//        k.append(0)
//        yield sorted(k, reverse=True)
// ```
// によって生成される数列を定数として記録する
inline const auto MULT_COEFF = std::array<std::vector<std::vector<uint8_t>>, 5>{
    std::vector<std::vector<uint8_t>>{{}},                             //
    std::vector<std::vector<uint8_t>>{{1}, {0}},                       //
    std::vector<std::vector<uint8_t>>{{2, 1}, {1, 0}, {2, 0}, {0, 0}}, //
    std::vector<std::vector<uint8_t>>{
        {3, 2, 1},
        {2, 1, 0},
        {3, 1, 0},
        {1, 0, 0},
        {3, 2, 0},
        {2, 0, 0},
        {3, 0, 0},
        {0, 0, 0},
    },
    std::vector<std::vector<uint8_t>>{
        {4, 3, 2, 1},
        {3, 2, 1, 0},
        {4, 2, 1, 0},
        {2, 1, 0, 0},
        {4, 3, 1, 0},
        {3, 1, 0, 0},
        {4, 1, 0, 0},
        {1, 0, 0, 0},
        {4, 3, 2, 0},
        {3, 2, 0, 0},
        {4, 2, 0, 0},
        {2, 0, 0, 0},
        {4, 3, 0, 0},
        {3, 0, 0, 0},
        {4, 0, 0, 0},
        {0, 0, 0, 0},
    }};

// ```python
// def single_coeff(
//     n: int, depth: int | None = None
// ) -> Generator[list[list[int]], None, None]:
//     depth = depth if depth is not None else n
//     if depth == 0:
//         yield []
//         return
//
//     for i in single_coeff(n, depth - 1):
//         for j in range(len(i)):
//             ret = copy.deepcopy(i)
//             ret[j].append(depth)
//             yield sorted(ret)
//
//         ret = copy.deepcopy(i)
//         ret.append([depth])
//
//         yield sorted(ret)
// ```
//
inline const auto SINGLE_COEFF =
    std::array<std::vector<std::vector<std::vector<uint8_t>>>, 5>{
        std::vector<std::vector<std::vector<uint8_t>>>{{}},
        std::vector<std::vector<std::vector<uint8_t>>>{{{1}}},
        std::vector<std::vector<std::vector<uint8_t>>>{{{1, 2}}, {{1}, {2}}},
        std::vector<std::vector<std::vector<uint8_t>>>{
            {{1, 2, 3}},
            {{1, 2}, {3}},
            {{1, 3}, {2}},
            {{1}, {2, 3}},
            {{1}, {2}, {3}}},
        std::vector<std::vector<std::vector<uint8_t>>>{{{1, 2, 3, 4}},
                                                       {{1, 2, 3}, {4}},
                                                       {{1, 2, 4}, {3}},
                                                       {{1, 2}, {3, 4}},
                                                       {{1, 2}, {3}, {4}},
                                                       {{1, 3, 4}, {2}},
                                                       {{1, 3}, {2, 4}},
                                                       {{1, 3}, {2}, {4}},
                                                       {{1, 4}, {2, 3}},
                                                       {{1}, {2, 3, 4}},
                                                       {{1}, {2, 3}, {4}},
                                                       {{1, 4}, {2}, {3}},
                                                       {{1}, {2, 4}, {3}},
                                                       {{1}, {2}, {3, 4}},
                                                       {{1}, {2}, {3}, {4}}},
    };

} // namespace Autodiff
