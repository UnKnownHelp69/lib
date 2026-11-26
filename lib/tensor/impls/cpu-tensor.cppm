module;

#include <algorithm>
#include <array>
#include <concepts>
#include <execution>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>

module tensor:cpu;

// Want dims > 0!!
template <typename T>
  requires std::is_arithmetic_v<T>
class CpuTensor {
  template <typename... DimsSizes>
  static constexpr bool kCorrectDimSizes = sizeof...(DimsSizes) > 0 && (std::integral<DimsSizes> && ...);

  static constexpr std::string_view kInvalidDimsSizeError =
      "all nested array must have same size (ragged tensors not supported)";

  static constexpr std::string_view kInvalidIndexesCountError = "index count is invalid";

  static constexpr std::string_view kOutOfRangeError = "tensor index out of range";

  static constexpr std::string_view kInvalidDimsError = "tensor has invalid dimensions for operation";

  using FlatIterator = T*;
  using ConstFlatIterator = const T*;

public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;

  template <typename... Dims>
    requires kCorrectDimSizes<Dims...>
  explicit CpuTensor(Dims... dims_sizes)
      : dims_{static_cast<std::size_t>(dims_sizes)...}
      , data_((dims_sizes * ...)) {}

  CpuTensor(std::initializer_list<T> initializer_list)
      : dims_{{1}}
      , data_{initializer_list} {}

  CpuTensor(std::initializer_list<std::initializer_list<T>> initializer_matrix)
      : dims_({initializer_matrix.size(), initializer_matrix.begin()->size()}) {
    data_.reserve(dims_.at(0) * dims_.at(1));
    for (auto list : initializer_matrix) {
      if (list.size() != initializer_matrix.begin()->size()) {
        throw std::invalid_argument{kInvalidDimsSizeError.data()};
      }
      data_.insert(data_.end(), list);
    }
  }

  constexpr auto data(this auto&& self) noexcept {
    return self.data_.data();
  }

  constexpr std::size_t rank() const noexcept {
    return dims_.size();
  }

  constexpr std::size_t extend(std::size_t idx) const {
    return dims_.at(idx);
  }

  template <typename... Indexes>
    requires kCorrectDimSizes<Indexes...>
  constexpr decltype(auto) operator[](this auto&& self, Indexes... indexes) {
    return self.AtFlat(self.ComposeFlatIndex(indexes...));
  }

  template <typename... Indexes>
    requires kCorrectDimSizes<Indexes...>
  constexpr decltype(auto) at(this auto&& self, Indexes... indexes) {
    return self.AtFlat(self.ComposeFlatIndex(indexes...));
  }

  CpuTensor& operator+=(const CpuTensor& other) & {
    if (!IsEqualDims(other)) {
      throw std::invalid_argument(kInvalidDimsError.data());
    }

    std::transform(
        std::execution::par_unseq,
        FlatBegin(),
        FlatEnd(),
        other.FlatBegin(),
        FlatBegin(),
        [](T lhs, T rhs) {
          return lhs + rhs;
        }
    );

    return *this;
  }

  template <typename U> requires std::is_arithmetic_v<U>
  friend bool operator==(const CpuTensor<U>& lhs, const CpuTensor<U>& rhs) noexcept;

private:
  auto FlatBegin(this auto&& self) noexcept {
    return self.data_.data();
  }

  auto FlatEnd(this auto&& self) noexcept {
    return self.FlatBegin() + self.FlatSize();
  }

  constexpr bool IsEqualDims(const CpuTensor& other) const noexcept {
    return dims_ == other.dims_;
  }

  constexpr decltype(auto) AtFlat(this auto&& self, std::size_t flat_index) {
    if (flat_index >= self.FlatSize()) {
      throw std::out_of_range(kOutOfRangeError.data());
    }
    return self.data_[flat_index];
  }

  constexpr std::size_t FlatSize() const noexcept {
    return data_.size();
  }

  template <typename... Indexes>
    requires kCorrectDimSizes<Indexes...>
  constexpr std::size_t ComposeFlatIndex(Indexes... indexes) const {
    if (sizeof...(Indexes) != rank()) {
      throw std::invalid_argument(kInvalidIndexesCountError.data());
    }

    std::array<std::size_t, sizeof...(Indexes)> index_arr{static_cast<std::size_t>(indexes)...};
    std::size_t flat_index = 0;
    for (std::size_t i = 0; i < rank(); ++i) {
      flat_index = (flat_index * extend(i)) + index_arr[i];
    }
    return flat_index;
  }

  std::vector<std::size_t> dims_;
  std::vector<T> data_;
};

template <typename T> requires std::is_arithmetic_v<T>
bool operator==(const CpuTensor<T>& lhs, const CpuTensor<T>& rhs) noexcept {
  if (!lhs.IsEqualDims(rhs)) {
    return false;
  }

  return std::equal(lhs.FlatBegin(), lhs.FlatEnd(), rhs.FlatBegin());
}
