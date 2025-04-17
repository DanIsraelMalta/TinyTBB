
#include "TinyTBB.h"
#include <assert.h>
#include <functional>

// globals
template <class... Args> using variadic_sum = std::plus<Args...>;
constexpr std::size_t N{ 5'000'000 };
constexpr std::size_t thread_count{ 4 };

std::int32_t test_sequential(std::vector<std::int32_t>& A) {
    std::int32_t sum{};
    for (auto a : A) {
        sum += a;
    }

    return sum;
}

std::int32_t test_async(std::vector<std::int32_t>& A) {
    // async sum (4 threads)
    std::int32_t sum{};
    std::vector<std::future<std::int32_t>> futures;
    const auto compute_partial_sum = [&](std::size_t low, std::size_t high) -> std::int32_t {
        std::int32_t s{};
        for (std::size_t i = low; i < high; ++i) {
            s += A[i];
        }
        return s;
    };

    const auto compute_partial_sum_for_worker = [&](std::size_t worker_num) -> std::int32_t {
        const std::size_t low{ worker_num * (N / thread_count) };
        const std::size_t high{ (worker_num + 1) * (N / thread_count) };
        return compute_partial_sum(low, high);
    };

    for (std::size_t k{}; k < thread_count; ++k) {
        futures.emplace_back(std::async(compute_partial_sum_for_worker, k));
    }
    for (auto& f : futures) {
        sum += f.get();
    }

    return sum;
}

std::int32_t parallel_reduce_sum(std::vector<std::int32_t>& A) {
    return TinyTBB::reduce<4>(A, variadic_sum<>{}); // explicit request of 4 lane vectorization
}

std::int32_t parallel_for_loop_sum(std::vector<std::int32_t>& A) {
    std::int32_t sum{};
    TinyTBB::for_loop(A, [&](auto low, auto high) {
        for (auto it = low; it != high; ++it) {
            sum += *it;
        }
    });

    return sum;
}

std::int32_t partial_parallel_for_loop_sum(std::vector<std::int32_t>& A) {
    std::int32_t sum{};
    TinyTBB::for_loop(TinyTBB::IndexRange<std::size_t>{0, A.size() / 2},
                      [&](auto low, auto high) {
                          for (auto i = low; i != high; ++i) {
                              sum += A[i];
                          }
                      });

    return sum;
}

std::int32_t partial_parallel_reduce_sum(std::vector<std::int32_t>& A) {
    return TinyTBB::reduce(TinyTBB::Range{ A.begin(), A.begin() + A.size() / 2u }, variadic_sum<>());
}

int main() {
    // define amount of used threads
    TinyTBB::set_thread_count(thread_count);

    // Launch asynchronously background task and get its future
    auto some_heavy_computation = [](double x) -> double {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return x + 32;
    };
    auto future = TinyTBB::task_with_future(some_heavy_computation, 10);

    // initialize vector
    std::vector<std::int32_t> A;
    A.reserve(N);
    for (std::size_t i{}; i < N; ++i) {
        A.emplace_back(i);
    }

    // test various ways to calculate sum of vector
    const std::int32_t sum{ test_sequential(A) };
    assert(sum == test_async(A));
    assert(sum == parallel_reduce_sum(A));
    assert(sum == parallel_for_loop_sum(A));
    assert(partial_parallel_for_loop_sum(A) == partial_parallel_reduce_sum(A));

    // check async task
    const double result = future.get();
    assert(result == 42);
}
