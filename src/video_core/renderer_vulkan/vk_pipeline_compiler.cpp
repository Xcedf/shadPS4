// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <thread>

#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_pipeline_compiler.h"

namespace Vulkan {

namespace {
thread_local bool is_compiler_thread = false;
}

bool PipelineCompileJob::TryRun() {
    State expected = State::Pending;
    if (!state.compare_exchange_strong(expected, State::Running, std::memory_order_acq_rel)) {
        return false;
    }
    func();
    func = {};
    Finish();
    return true;
}

void PipelineCompileJob::Wait() {
    if (IsDone() || TryRun()) {
        return;
    }
    std::unique_lock lock{mutex};
    cv.wait(lock, [this] { return IsDone(); });
}

void PipelineCompileJob::Cancel() {
    State expected = State::Pending;
    if (state.compare_exchange_strong(expected, State::Running, std::memory_order_acq_rel)) {
        func = {};
        Finish();
        return;
    }
    Wait();
}

void PipelineCompileJob::Finish() {
    {
        // Publish under the lock so a waiter can't miss the notification between its check and
        // going to sleep.
        std::scoped_lock lock{mutex};
        state.store(State::Done, std::memory_order_release);
    }
    cv.notify_all();
}

PipelineCompiler::PipelineCompiler(u32 num_workers) {
    workers.reserve(num_workers);
    for (u32 i = 0; i < num_workers; ++i) {
        workers.emplace_back([this](std::stop_token stop_token) { WorkerLoop(stop_token); });
    }
}

PipelineCompiler::~PipelineCompiler() {
    {
        std::scoped_lock lock{queue_mutex};
        for (const auto& job : queue) {
            job->Cancel();
        }
        queue.clear();
    }
    for (auto& worker : workers) {
        worker.request_stop();
    }
    workers.clear();
}

std::shared_ptr<PipelineCompileJob> PipelineCompiler::Submit(Common::UniqueFunction<void> func) {
    auto job = std::make_shared<PipelineCompileJob>(std::move(func));
    if (workers.empty()) {
        job->TryRun();
        return job;
    }
    {
        std::scoped_lock lock{queue_mutex};
        queue.push_back(job);
    }
    queue_cv.notify_one();
    return job;
}

u32 PipelineCompiler::DefaultNumWorkers() {
    const u32 num_threads = std::max(std::thread::hardware_concurrency(), 2U);
    return std::clamp(num_threads / 2, 1U, 8U);
}

bool PipelineCompiler::IsCompilerThread() noexcept {
    return is_compiler_thread;
}

void PipelineCompiler::WorkerLoop(std::stop_token stop_token) {
    Common::SetCurrentThreadName("shadPS4:PipelineCompiler");
    is_compiler_thread = true;
    // Background work: the game's threads come first. A thread that needs a pipeline that no
    // worker has started yet compiles it itself, so a busy worker never holds it back.
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    while (!stop_token.stop_requested()) {
        std::shared_ptr<PipelineCompileJob> job;
        {
            std::unique_lock lock{queue_mutex};
            Common::CondvarWait(queue_cv, lock, stop_token, [this] { return !queue.empty(); });
            if (queue.empty()) {
                continue;
            }
            job = std::move(queue.front());
            queue.pop_front();
        }
        // A job that the waiting thread already took over is skipped here.
        job->TryRun();
    }
}

} // namespace Vulkan
