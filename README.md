# TinyTBB
TinyTBB is a header only library which provide basic constructs used for parallel operations (loops, reductions, tasks) with API similar to Intel TBB.

Notice - tested with GCC/CLANG and requires c++20.

## Usage example:

calculate sum of vector elements in parallel in various ways:
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

// fill vector:
std::vector<std::int32_t> A;
A.reserve(N);
for (std::size_t i{}; i < N; ++i) {
    A.emplace_back(i);
}

// accumulate vector elements in parallel manner, with internal operations explicitly vectorized, using reduction operation:
const std::int32_t sum1{ TinyTBB::reduce<step>(A, variadic_sum<>{}) };

// accumulate vector elements in parallel manner using parallel for-loop syntax:
std::int32_t sum2{};
TinyTBB::for_loop(A, [&](auto low, auto high) {
    for (auto it = low; it != high; ++it) {
        sum2 += *it;
    }
});

// accumulate first half of vector elements in parallel manner using parallel for-loop syntax with defined indices:
std::int32_t sum3{};
TinyTBB::for_loop(TinyTBB::IndexRange<std::size_t>{0, A.size() / 2},
                  [&](auto low, auto high) {
                      for (auto i = low; i != high; ++i) {
                          sum3 += A[i];
                      }
                  });

// accumulate first half of vector elements in parallel manner, without explicitly vectoriztion, using reduction operation with defined range:
const std::int32_t sum4{ TinyTBB::reduce(TinyTBB::Range{ A.begin(), A.begin() + A.size() / 2u }, variadic_sum<>()) };
```

launch asynchronously task and get its future:
```cpp
// define amount of used threads
TinyTBB::set_thread_count(4);

// define a heavy task and send it to thread pool
auto heavy_task = [](double x) -> double {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return x + 32;
};
auto heavy_task_future = TinyTBB::task_with_future(heavy_task, 10);

...
// few seconds later...
...

// get heavy task result
const double result = heavy_task_future.get();
assert(result == 42);
```
