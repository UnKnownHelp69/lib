module;

#include <array>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

module tensor:cpu;

// Want dims > 0!!
template <typename T>
  requires std::is_arithmetic_v<T>
class CpuTensor {
  template <typename... DimsSizes>
  static constexpr bool kCorrectDimSizes = sizeof...(DimsSizes) > 0 && (std::is_same_v<DimsSizes, std::size_t> && ...);

  static constexpr std::string_view kInvalidDimsSizeError =
      "all nested array must have same size (ragged tensors not supported)";

public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;

  template <typename... Dims>
    requires kCorrectDimSizes<Dims...>
  explicit CpuTensor(Dims... dims_sizes)
      : dims_{dims_sizes...}
      , data_{(dims_sizes * ...)} {}

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

  constexpr std::size_t dims_cnt() const noexcept {
    return dims_.size();
  }

  constexpr std::size_t extend(std::size_t idx) const {
    return dims_.at(idx);
  }

  template <typename... Indexes>
    requires kCorrectDimSizes<Indexes...>
  T& operator[](Indexes... indexes) noexcept {
    return AtFlat(ComposeFlatIndex(indexes...));
  }

  template <typename... Indexes>
    requires kCorrectDimSizes<Indexes...>
  T& at(Indexes... indexes) {
    std::size_t flat_index = ComposeFlatIndex(indexes...);
    if (flat_index >= FlatSize()) {
      throw std::out_of_range("tensor index out of range");
    }

    return AtFlat(flat_index);
  }

private:
  constexpr T& AtFlat(std::size_t flat_index) {
    return data_[flat_index];
  }

  constexpr std::size_t FlatSize() const noexcept {
    return data_.size();
  }

  template <typename... Indexes>
    requires kCorrectDimSizes<Indexes...>
  constexpr std::size_t ComposeFlatIndex(Indexes... indexes) {
    std::array<std::size_t, sizeof...(Indexes)> index_arr{indexes...};
    std::size_t flat_index = 0;
    for (std::size_t i = 0; i < dims_cnt(); ++i) {
      flat_index = (flat_index * extend(i)) + index_arr[i];
    }
    return flat_index;
  }

  std::vector<std::size_t> dims_;
  std::vector<T> data_;
};
