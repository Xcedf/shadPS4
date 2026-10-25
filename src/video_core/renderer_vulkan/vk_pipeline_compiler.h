// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "common/polyfill_thread.h"
#include "common/types.h"
#include "common/unique_function.h"

namespace Vulkan {

/// One piece of pipeline creation work. Whoever needs it first runs it: a compiler worker, or the
/// thread that waits on it if no worker has picked it up yet, so waiting never queues behind
/// unrelated work.
class PipelineCompileJob {
public:
    explicit PipelineCompileJob(Common::UniqueFunction<void> func_) : func{std::move(func_)} {}

    PipelineCompileJob(const PipelineCompileJob&) = delete;
    PipelineCompileJob& operator=(const PipelineCompileJob&) = delete;

    /// Runs the job on the calling thread if nobody has started it yet.
    bool TryRun();

    /// Returns true once the job has run or was cancelled.
    [[nodiscard]] bool IsDone() const noexcept {
        return state.load(std::memory_order_acquire) == State::Done;
    }

    /// Runs the job here if nobody has started it, otherwise waits for the thread running it.
    void Wait();

    /// Drops the job if nobody has started it yet, otherwise waits for it to finish.
    void Cancel();

private:
    enum class State : u32 {
        Pending,
        Running,
        Done,
    };

    void Finish();

    std::atomic<State> state{State::Pending};
    Common::UniqueFunction<void> func;
    std::mutex mutex;
    std::condition_variable cv;
};

/// Runs pipeline creation on worker threads, so the driver can compile several pipelines at once
/// and away from the GPU command processor.
class PipelineCompiler {
public:
    explicit PipelineCompiler(u32 num_workers);
    ~PipelineCompiler();

    PipelineCompiler(const PipelineCompiler&) = delete;
    PipelineCompiler& operator=(const PipelineCompiler&) = delete;

    std::shared_ptr<PipelineCompileJob> Submit(Common::UniqueFunction<void> func);

    [[nodiscard]] u32 NumWorkers() const noexcept {
        return static_cast<u32>(workers.size());
    }

    /// Number of workers that suits this machine: the rest of the cores stay with the game.
    static u32 DefaultNumWorkers();

    /// Returns true when called from one of the compiler worker threads.
    static bool IsCompilerThread() noexcept;

private:
    void WorkerLoop(std::stop_token stop_token);

    std::mutex queue_mutex;
    std::condition_variable_any queue_cv;
    std::deque<std::shared_ptr<PipelineCompileJob>> queue;
    std::vector<std::jthread> workers;
};

} // namespace Vulkan
