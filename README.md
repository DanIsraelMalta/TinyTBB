# TinyTBB
TinyTBB is a header only library which provide basic constructs used for parallel operations (loops, reductions, tasks) with API similar to Intel TBB.

Notice - tested with GCC/CLANG and requires c++20.

## Usage example:

define global parameters
```cpp
#include "TinyTBB.h"
#include <assert.h>
#include <functional>

// variadic summation
template <class... Args> using variadic_sum = std::plus<Args...>;

// number of elements in vector
constexpr std::size_t N{ 5'000'000 };

// number of threads in thread pool
constexpr std::size_t thread_count{ 4 };

// step size for forced vectorization
constexpr std::size_t step{ 4 };

// define amount of used threads
TinyTBB::set_thread_count(thread_count);
```

fill a vector:
```cpp
std::vector<std::int32_t> A;
A.reserve(N);
for (std::size_t i{}; i < N; ++i) {
    A.emplace_back(i);
}
```

accumulate vector elements in parallel manner, with internal operations explicitly vectorized, using reduction operation:
```cpp
const std::int32_t sum = TinyTBB::reduce<step>(A, variadic_sum<>{});
```

accumulate vector elements in parallel manner using parallel for-loop syntax:
```cpp
std::int32_t sum{};
TinyTBB::for_loop(A, [&](auto low, auto high) {
    for (auto it = low; it != high; ++it) {
        sum += *it;
    }
});
```

accumulate first half of vector elements in parallel manner using parallel for-loop syntax with defined indices:
```cpp
std::int32_t sum{};
TinyTBB::for_loop(TinyTBB::IndexRange<std::size_t>{0, A.size() / 2},
                  [&](auto low, auto high) {
                      for (auto i = low; i != high; ++i) {
                          sum += A[i];
                      }
                  });
```

accumulate first half of vector elements in parallel manner, without explicitly vectoriztion, using reduction operation with defined range:
```cpp
const std::int32_t sum = TinyTBB::reduce(TinyTBB::Range{ A.begin(), A.begin() + A.size() / 2u }, variadic_sum<>());
```

launch asynchronously background task and get its future:
```cpp
auto heavy_task = [](double x) -> double {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return x + 32;
};
auto heavy_task_future = TinyTBB::task_with_future(heavy_task, 10);

...
// few seconds later...
...


const double result = heavy_task_future.get();
assert(result == 42);
```
