module;

#include <type_traits>

export module tensor;

import :cpu;

template <typename T> requires std::is_arithmetic_v<T>
class CpuTensor {
};
