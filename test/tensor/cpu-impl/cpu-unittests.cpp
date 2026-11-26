module;

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <complex>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

module tensor;

import :cpu;

constexpr std::string_view kInvalidDimsSizeError =
    "all nested array must have same size (ragged tensors not supported)";

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
        testing::ThrowsMessage<std::invalid_argument>(testing::StrEq(kInvalidDimsSizeError.data()))
    );
  }
}
