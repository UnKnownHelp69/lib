module;

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <complex>
#include <concepts>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

module tensor;

import :cpu;

namespace {

constexpr std::string_view kInvalidDimsSizeError =
    "all nested array must have same size (ragged tensors not supported)";
constexpr std::string_view kInvalidIndexesCountError = "index count is invalid";
constexpr std::string_view kOutOfRangeError = "tensor index out of range";
constexpr std::string_view kInvalidDimsError = "tensor has invalid dimensions for operation";

template <typename Indexer>
void CheckIndexing(Indexer indexer) {
  CpuTensor tensor({{1, 2}, {3, 4}});

  static_assert(std::same_as<decltype(std::invoke(indexer, tensor, 0, 0)), int&>);
  static_assert(std::same_as<decltype(std::invoke(indexer, std::as_const(tensor), 0, 0)), const int&>);

  EXPECT_THAT(
      [&]() {
        std::invoke(indexer, tensor, 0);
      },
      testing::ThrowsMessage<std::invalid_argument>(testing::StrEq(kInvalidIndexesCountError))
  );

  EXPECT_THAT(
      [&]() {
        std::invoke(indexer, tensor, 2, 2);
      },
      testing::ThrowsMessage<std::out_of_range>(testing::StrEq(kOutOfRangeError))
  );
}

} // namespace

TEST(CpuTensorTest, InitializerListCtor) {
  {
    SCOPED_TRACE("1 dimension tensor");
    const std::vector<float> real = {1, 2, 3};
    CpuTensor<float> tensor({1, 2, 3});

    for (std::size_t i = 0; i < 3; ++i) {
      SCOPED_TRACE(std::format("vector index: {}", i));
      EXPECT_EQ(real.at(i), tensor.at(i));
      EXPECT_EQ(real[i], tensor[i]);
    }
  }

  {
    SCOPED_TRACE("2 dimension tensor");
    const std::vector<std::vector<int>> real = {{1, 2}, {3, 4}, {5, 6}};
    CpuTensor tensor({{1, 2}, {3, 4}, {5, 6}});

    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 2; ++j) {
        SCOPED_TRACE(std::format("index: {}, {}", i, j));
        EXPECT_EQ(real.at(i).at(j), tensor.at(i, j));
        EXPECT_EQ(real[i][j], (tensor[i, j]));
      }
    }

    EXPECT_THAT(
        []() {
          CpuTensor({{1, 2}, {2}});
        },
        testing::ThrowsMessage<std::invalid_argument>(testing::StrEq(kInvalidDimsSizeError))
    );
  }
}

TEST(CpuTensorTest, IndexationTest) {
  CpuTensor tensor({{1, 2}, {3, 4}});

  CheckIndexing([](auto&& tensor, auto... indexes) -> decltype(auto) {
    return tensor[indexes...];
  });

  CheckIndexing([](auto&& tensor, auto... indexes) -> decltype(auto) {
    return tensor.at(indexes...);
  });
}

TEST(CpuTensorTest, AdditionTest) {
  CpuTensor<std::size_t> lhs(3, 3, 3);
  CpuTensor<std::size_t> rhs(3, 3, 3);

  CpuTensor<std::size_t> res(3, 3, 3);

  // begging for iterator
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 3; ++k) {
        auto a = 3 * i + 5 * j + k;
        auto b = i + 8 * j + 4 * k;

        lhs[i, j, k] = a;
        rhs[i, j, k] = b;

        res[i, j, k] = a + b;
      }
    }
  }

  lhs += rhs;

  EXPECT_TRUE(lhs == res);

  CpuTensor<std::size_t> different(3, 4, 3);

  EXPECT_THAT([&]() {
    different += lhs;
  }, testing::ThrowsMessage<std::invalid_argument>(testing::StrEq(kInvalidDimsError)));
}
