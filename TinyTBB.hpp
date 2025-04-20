#pragma once

#include <iterator>
#include <functional>
#include <cstddef>
#include <type_traits>
#include <concepts>
#include <queue>
#include <vector>
#include <array>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>

// compiler friendly std::move replacement
#ifndef MOV
#define MOV(...) static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)
#else
#undef MOV
#define MOV(...) static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)
#endif

// compiler friendly std::forward replacement
#ifndef FWD
#define FWD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)
#else
#undef FWD
#define FWD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)
#endif

/**
* \brief constructs for parallel operations (loops, reductions, tasks) with API similar to Intel TBB
**/
namespace TinyTBB {

    /**
    \brief internal utilities
    **/
    namespace details {

        // std::min / std::max replacements for integrals
        template<typename T, typename U>
            requires(std::is_integral_v<T> && std::is_integral_v<U>)
        [[nodiscard]] constexpr std::size_t min(const T a, const U b) noexcept {
            return (static_cast<std::size_t>(a) < static_cast<std::size_t>(b)) ? static_cast<std::size_t>(a) : static_cast<std::size_t>(b);
        }

        template<typename T, typename U>
            requires(std::is_integral_v<T>&& std::is_integral_v<U>)
        [[nodiscard]] constexpr std::size_t max(const T a, const U b) noexcept {
            return (static_cast<std::size_t>(a) < static_cast<std::size_t>(b)) ? static_cast<std::size_t>(b) : static_cast<std::size_t>(a);
        }

        /**
        * \brief return number of threads supported by hardware or 0
        * @param {size_t, out} number of threads
        **/
        [[nodiscard]] std::size_t max_thread_count() noexcept {
            const std::size_t detected_threads{ std::thread::hardware_concurrency() };
            return detected_threads ? detected_threads : 1;
        };

        // concept of a collection iterable via 'begin' and 'end' iterators
        template<class T> concept Iterable = requires(T collection) {
            { collection.begin() } -> std::forward_iterator;
            { collection.end() } -> std::forward_iterator;
        };

        /**
        * \brief compile time for loop (unrolls loop)
        *        taken from https://github.com/DanIsraelMalta/Numerics/blob/main/Utilities.h
        **/
        template<std::size_t Start, std::size_t Inc, std::size_t End, class F>
            requires(std::is_invocable_v<F, decltype(Start)>)
        constexpr void static_for(F&& f) noexcept {
            if constexpr (Start < End) {
                f(std::integral_constant<decltype(Start), Start>());
                static_for<Start + Inc, Inc, End>(FWD(f));
            }
        }

        /**
        * \brief mutex protected object
        **/
        template<class T, class M = std::mutex>
        struct Mutex {
            Mutex() = default;
            Mutex(const T& value) : value(value), mutex() {}
            Mutex(T&& value) : value(std::move(value)), mutex() {}

            Mutex(const Mutex& other) = delete;
            Mutex(Mutex&& other) noexcept = delete;
            Mutex& operator=(const Mutex& other) = delete;
            Mutex& operator=(Mutex&& other) noexcept = delete;

            /**
            * \brief apply operations on value
            * @param {callable, in} function to apply
            **/
            template<class F>
                requires(std::is_invocable_v<F, T>)
            [[nodiscard]] auto apply(F&& func) const {
                const std::lock_guard lock(this->mutex);
                return FWD(func)(this->value);
            }
            template<class F>
                requires(std::is_invocable_v<F, T>)
            [[nodiscard]] auto apply(F&& func) {
                const std::lock_guard lock(this->mutex);
                return FWD(func)(this->value);
            }

            /**
            * \brief release object and return ownership over it
            * @param {T&&, out} get object and its ownership
            **/
            [[nodiscard]] T&& release() noexcept {
                return MOV(this->value);
            }

            // internals
            private:
                T value;
                mutable M mutex;
        };

        /**
        * \brief single-queue thread pool
        **/
        class ThreadPool {
            // API
            public:
                ThreadPool() = default;
                explicit ThreadPool(std::size_t thread_count) {
                    this->start_threads(thread_count);
                }

                ThreadPool(const ThreadPool& other) = delete;
                ThreadPool(ThreadPool&& other) noexcept = delete;
                ThreadPool& operator=(const ThreadPool& other) = delete;
                ThreadPool& operator=(ThreadPool&& other) noexcept = delete;

                ~ThreadPool() {
                    this->resume();
                    this->wait_for_tasks();
                    this->stop_all_threads();
                }

                /**
                * \brief return number of threads
                * @param{size_t, out} number of threads in thread pool
                **/
                [[nodiscard]] std::size_t get_thread_count() const {
                    const std::lock_guard<std::recursive_mutex> thread_lock(this->thread_mutex);
                    return this->threads.size();
                }

                /**
                * \brief set number of threads in thread pool
                * @param {size_t, in} number of threads in thread pool
                **/
                void set_thread_count(const std::size_t thread_count) {
                    // all threads need to be free
                    this->wait_for_tasks();

                    const std::size_t current_thread_count{ this->get_thread_count() };
                    if (thread_count == current_thread_count) {
                        return;
                    }

                    if (thread_count > current_thread_count) {
                        this->start_threads(thread_count - current_thread_count);
                    }
                    else {
                        this->stop_all_threads();
                        {
                            const std::lock_guard<std::mutex> task_lock(this->task_mutex);
                            this->stopping = false;
                        }
                        this->start_threads(thread_count);
                    }
                }

                /**
                * \brief add task to thread
                * @param {callable,    in} task
                * @param {variadic..., in} task arguments
                **/
                template<class F, class... Args>
                    requires(std::is_invocable_v<F, Args...>)
                void add_task(F&& func, Args&&... args) {
                    const std::lock_guard<std::mutex> task_lock(this->task_mutex);
                    this->tasks.emplace(std::bind(FWD(func), FWD(args)...));
                    this->task_cv.notify_one();
                }

                /**
                * \brief add task to thread and return future handle to it
                * @param {callable,    in}  task
                * @param {variadic..., in}  task arguments
                * @param {future,      out} future to task
                **/
                template<class F, class... Args,
                    class FuncReturnType = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
                    requires(std::is_invocable_v<F, Args...>)
                [[nodiscard]] std::future<FuncReturnType> add_task_with_future(F&& func, Args&&... args) {
                    std::packaged_task<FuncReturnType()> new_task(std::bind(FWD(func), FWD(args)...));
                    auto                                 future = new_task.get_future();
                    this->add_task(MOV(new_task));
                    return future;
                }

                /**
                * \brief wait for all tasks in thread pool to complete
                **/
                void wait_for_tasks() {
                    std::unique_lock<std::mutex> task_lock(this->task_mutex);
                    this->waiting = true;
                    this->task_finished_cv.wait(task_lock, [&] { return this->tasks.empty() && this->tasks_running == 0; });
                    this->waiting = false;
                }

                /**
                * \brief clear all tasks from thread pool
                **/
                void clear_task_queue() {
                    const std::lock_guard<std::mutex> task_lock(this->task_mutex);
                    this->tasks = {};
                }

                /**
                * \brief pause tasks
                **/
                void pause() {
                    const std::lock_guard<std::mutex> task_lock(this->task_mutex);
                    this->paused = true;
                }

                /**
                * \brief resume tasks
                **/
                void resume() {
                    const std::lock_guard<std::mutex> task_lock(this->task_mutex);
                    this->paused = false;
                    this->task_cv.notify_all();
                }

                /**
                * \brief tests if thread pool is paused
                * @param {bool, out} true if thread pool is pausing, false otherwise
                **/
                [[nodiscard]] bool is_paused() const noexcept {
                    const std::lock_guard<std::mutex> task_lock(this->task_mutex);
                    return this->paused;
                }

            // internals
            private:
                // properties
                mutable std::recursive_mutex thread_mutex;
                mutable std::mutex task_mutex;
                std::condition_variable task_cv;
                std::condition_variable task_finished_cv;
                std::int32_t tasks_running{};
                bool stopping{ false };
                bool paused{ false };
                bool waiting{ false };
                std::vector<std::thread> threads;
                std::queue<std::packaged_task<void()>> tasks{};
                
                /**
                * \brief thread managers
                **/
                void manager() {
                    bool task_was_finished{ false };

                    while (true) {
                        std::unique_lock<std::mutex> task_lock(this->task_mutex);

                        if (task_was_finished) {
                            --this->tasks_running;
                            if (this->waiting) {
                                this->task_finished_cv.notify_all();
                            }
                        }

                        this->task_cv.wait(task_lock, [&] { return this->stopping || (!this->paused && !this->tasks.empty()); });
                        if (this->stopping) {
                            break;
                        }

                        std::packaged_task<void()> task_to_execute = MOV(this->tasks.front());
                        this->tasks.pop();
                        ++this->tasks_running;
                        task_lock.unlock();

                        task_to_execute();
                        task_was_finished = true;
                    }
                }

                /**
                * \brief start thread pool with given amount of threads
                * @param {size_t, in} amount of threads
                **/
                void start_threads(const std::size_t worker_count_increase) {
                    const std::lock_guard<std::recursive_mutex> thread_lock(this->thread_mutex);
                    for (std::size_t i{}; i < worker_count_increase; ++i) {
                        this->threads.emplace_back(&ThreadPool::manager, this);
                    }
                }

                /**
                * \brief stop all threads in thread pool
                **/
                void stop_all_threads() {
                    const std::lock_guard<std::recursive_mutex> thread_lock(this->thread_mutex);

                    {
                        const std::lock_guard<std::mutex> task_lock(this->task_mutex);
                        this->stopping = true;
                        this->task_cv.notify_all();
                    }

                    for (auto& worker : this->threads) {
                        if (worker.joinable()) {
                            worker.join();
                        }
                    }

                    this->threads.clear();
                }
            };

            /**
            * \brief return reference for a static thread pool
            * @param {ThreadPool&, out} reference for static thread pool
            **/
            inline ThreadPool& static_thread_pool() {
                static ThreadPool pool(max_thread_count());
                return pool;
            }
    };

    /**
    * \brief get amount of threads in thread pool
    * @param {size_t, out} amount of threads in thread pool
    **/
    [[nodiscard]] std::size_t get_thread_count() {
        return details::static_thread_pool().get_thread_count();
    }

    /**
    * \brief set amount of threads in thread pool
    * @param {size_t, in} amount of threads in thread pool
    **/
    void set_thread_count(std::size_t thread_count) {
        details::static_thread_pool().set_thread_count(thread_count);
    }

    /**
    * \brief add task to static thread pool
    * @param {callable,    in}  task
    * @param {variadic..., in}  task arguments
    **/
    template <class F, class... Args>
        requires(std::is_invocable_v<F, Args...>)
    void task(F&& func, Args&&... args) {
        details::static_thread_pool().add_task(FWD(func), FWD(args)...);
    }

    /**
    * \brief add task to static thread pool and return future for it
    * @param {callable,    in}  task
    * @param {variadic..., in}  task arguments
    * @param {future,      out} future to task
    **/
    template <class F, class... Args,
        class T = typename std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
        requires(std::is_invocable_v<F, Args...>)
    std::future<T> task_with_future(F&& func, Args&&... args) {
        return details::static_thread_pool().add_task_with_future(FWD(func), FWD(args)...);
    }

    /**
    * \brief wait for all static thread pool tasks to complete
    **/
    inline void wait_for_tasks() {
        details::static_thread_pool().wait_for_tasks();
    }

    /**
    * \brief a range given as two indices [first, last]
    **/
    template<class I = std::size_t>
        requires(std::is_integral_v<I>)
    struct IndexRange {
        static constexpr std::size_t default_grains_per_thread{ 4 }; // how many tasks per thread

        IndexRange() = delete;
        constexpr IndexRange(I first, I last, std::size_t grain_size) : first(first), last(last), grain_size(grain_size) {}
        IndexRange(I first, I last) : IndexRange(first, last, details::max(1, (last - first) / (get_thread_count() * default_grains_per_thread))) {};

        // properties
        I first;
        I last;
        std::size_t grain_size;
    };

    /**
    * \brief a range given as two iterators [begin, end]
    **/
    template<class It>
        requires(std::random_access_iterator<It>)
    struct Range {
        static constexpr std::size_t default_grains_per_thread{ 4 }; // how many tasks per thread

        Range() = delete;
        constexpr Range(It begin, It end, std::size_t grain_size) : begin(begin), end(end), grain_size(grain_size) {}
        Range(It begin, It end) : Range(begin, end, details::max(1, (end - begin) / (get_thread_count() * default_grains_per_thread))) {}

        template<details::Iterable Collection>
        Range(const Collection& collection) : Range(collection.begin(), collection.end()) {}

        template<details::Iterable Collection>
        Range(Collection& collection) : Range(collection.begin(), collection.end()) {}

        // properties
        It        begin;
        It        end;
        std::size_t grain_size;
    };

    // Range deduction guides
    template<details::Iterable Collection> Range(const Collection& collection) -> Range<typename Collection::const_iterator>;
    template<details::Iterable Collection> Range(Collection& collection) -> Range<typename Collection::iterator>;

    /**
    * \brief given indexed range and callable, apply callable on range in parallel manner
    * @param {IndexRange in} range
    * @param {callable,  in} task
    **/
    template<class Idx, class F>
    void for_loop(IndexRange<Idx> range, F&& func) {
        for (Idx i = range.first; i < range.last; i += range.grain_size) {
            task(FWD(func), i, details::min(i + range.grain_size, range.last));
        }

        wait_for_tasks();
    }

    /**
    * \brief given iterable range and callable, apply callable on range in parallel manner
    * @param {Range,    in} range
    * @param {callable, in} task
    **/
    template<class It, class F>
        requires(std::random_access_iterator<It> && std::is_invocable_v<F, It, It>)
    void for_loop(Range<It> range, F&& func) {
        for (It i = range.begin; i < range.end; i += range.grain_size) {
            task(FWD(func), i, i + details::min(range.grain_size, static_cast<std::size_t>(range.end - i)));
        }

        wait_for_tasks();
    }

    /**
    * \brief given iterable collection and callable, apply callable on all collection nodes
    * @param {Iterable, in} collection
    * @param {callable, in} task
    **/
    template<details::Iterable Collection, class F>
    void for_loop(Collection&& collection, F&& func) {
        for_loop(Range{ FWD(collection) }, FWD(func));
    }

    /**
    * \brief calculate in parallel the generalized sum of a group of elements over an binary operation
    * @param {STEP}            size of required vectorization (1 means no explicit vectorization)
    * @param {Range,      in}  range
    * @param {callable,   in}  binary operation
    * @param {value_type, out} reduction output
    **/
    template<std::size_t STEP = 1, class BinaryOp, class It, class T = typename It::value_type>
        requires(std::random_access_iterator<It> &&
                 std::is_invocable_v<BinaryOp, T, T> && 
                 std::is_same_v<T, typename std::invoke_result_t<BinaryOp, T, T>>)
    T reduce(Range<It> range, BinaryOp&& op) {
        // housekeeping
        details::Mutex<T> result{ *range.begin };

        // reduction
        for_loop(Range<It>{range.begin + 1, range.end, range.grain_size}, [&](It low, It high) {
            if constexpr (STEP > 1) {
                if (const std::size_t range_size{ static_cast<std::size_t>(high - low) };
                    range_size > STEP) {
                    // parallel reduction, hopefully vectorized
                    std::array<T, STEP> partial_results;
                    details::static_for<0, 1, STEP>([&](std::size_t j) {
                        partial_results[j] = static_cast<T>(*(low + j));
                    });
                    It it{ low + STEP };

                    for (; it < high - STEP; it += STEP) {
                        details::static_for<0, 1, STEP>(
                            [&, it](std::size_t j) {
                                partial_results[j] = op(partial_results[j], static_cast<T>(*(it + j)));
                        });
                    }

                    // reduction of remaining elements
                    for (; it < high; ++it) {
                        partial_results[0] = op(partial_results[0], *it);
                    }

                    // collect results
                    for (std::size_t i = 1; i < partial_results.size(); ++i) {
                        partial_results[0] = op(partial_results[0], partial_results[i]);
                    }

                    // accumulate results
                    result.apply([&](auto&& res) { res = op(FWD(res), partial_results[0]); });
                    return;
                }
            }
                 
            // parallel reduction (unrolled)
            T partial_result{ *low };
            for (auto it = low + 1; it != high; ++it) {
                partial_result = op(partial_result, *it);
            }

            // accumulate results
            result.apply([&](auto&& res) { res = op(FWD(res), partial_result); });
        });

        // output
        return result.release();
    }

    /**
    * \brief calculate in parallel the generalized sum of a collection over an binary operation
    * @param {STEP}          size of required vectorization (1 means no explicit vectorization)
    * @param {Iterable,   in}  collection
    * @param {callable,   in}  binary operation
    * @param {value_type, out} reduction output
    **/
    template<std::size_t STEP = 1, class BinaryOp, details::Iterable Collection,
             class T = typename std::decay_t<Collection>::value_type>
        requires(std::is_invocable_v<BinaryOp, T, T> && 
                 std::is_same_v<T, typename std::invoke_result_t<BinaryOp, T, T>>)
    T reduce(Collection&& collection, BinaryOp&& op) {
        return reduce<STEP>(Range{ FWD(collection) }, FWD(op));
    }
}
