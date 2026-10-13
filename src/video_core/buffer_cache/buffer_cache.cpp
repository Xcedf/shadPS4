// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "common/perf_profiler.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
// Memory is made resident in aligned granules of this size. Sparse binds get slower the more is
// already bound on some drivers, so it pays to bind a little ahead in few larger operations
// rather than a block at a time as the game streams data in.
static constexpr u64 RESIDENCY_GRANULE_SIZE = 16_MB;
static constexpr u64 RESIDENCY_CHUNK_SIZE = 256_MB;
// Memory game threads read back is copied back ahead of them for this long after they last
// faulted on it, for at most this many windows of it.
static constexpr auto HotWindowLife = std::chrono::seconds{5};
static constexpr size_t MaxHotWindows = 64;
// A window whose copies ahead go stale is skipped for up to this many chances to copy it.
static constexpr u8 MaxPrefetchBackoff = 7;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_}, page_manager{tracker},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
    readback_thread = std::jthread{[this](std::stop_token token) { ReadbackThread(token); }};

    sync_large_reads_once = EmulatorSettings.IsSyncLargeReadsOnce();
    defer_uploads = EmulatorSettings.IsUploadThread();
    if (defer_uploads) {
        upload_thread = std::jthread{[this](std::stop_token token) { UploadThread(token); }};
        SetHostWorkOn(scheduler);
    }
}

void BufferCache::SetHostWorkOn(Vulkan::Scheduler& target) {
    if (!defer_uploads) {
        return;
    }
    target.SetHostWork(
        [uploads = host_uploads] { return uploads->queued.load(std::memory_order_acquire); },
        [uploads = host_uploads](u64 mark) {
            u64 done = uploads->done.load(std::memory_order_acquire);
            if (done >= mark) {
                return;
            }
            Common::Perf::ScopedStall stall{Common::Perf::Stall::UploadWait};
            for (; done < mark; done = uploads->done.load(std::memory_order_acquire)) {
                uploads->done.wait(done, std::memory_order_acquire);
            }
        });
}

BufferCache::~BufferCache() {
    if (upload_thread.joinable()) {
        upload_thread.request_stop();
        upload_thread.join();
    }
    // Nothing is left to wait for once the thread is gone.
    host_uploads->done.store(~u64{0}, std::memory_order_release);
    host_uploads->done.notify_all();
}

void BufferCache::UploadThread(std::stop_token token) {
    Common::SetCurrentThreadName("shadPS4:UploadWorker");
    PageManager::AppliesDeferredProtection();
    auto& uploads = *host_uploads;
    while (true) {
        HostUpload upload;
        {
            std::unique_lock lk{uploads.mutex};
            if (!uploads.cv.wait(lk, token, [&] { return !uploads.queue.empty(); })) {
                return;
            }
            upload = std::move(uploads.queue.front());
            uploads.queue.pop_front();
        }
        {
            Common::Perf::ScopedStall stall{Common::Perf::Stall::UploadThread};
            // Protected first, so what the CPU writes before is copied and what it writes after
            // faults, and is uploaded again.
            for (const auto& [first_page, num_pages] : upload.protects) {
                page_manager.ApplyProtection(first_page, num_pages);
            }
            for (const auto& [address, dest, size] : upload.copies) {
                memory->CopySparseMemory(address, dest, size);
            }
            if (!upload.copies.empty()) {
                upload.staging.Flush();
            }
        }
        u64 bytes{};
        for (const auto& copy : upload.copies) {
            bytes += std::get<2>(copy);
        }
        uploads.pending_bytes.fetch_sub(bytes, std::memory_order_relaxed);
        uploads.done.fetch_add(1, std::memory_order_release);
        uploads.done.notify_all();
    }
}

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
    ApplyFinishedReadbacks();
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

std::tuple<const Buffer*, VAddr, VAddr> BufferCache::GetReadbackWindow(VAddr device_addr,
                                                                       u64 size) {
    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);

    // GPU-modified ranges come as many small scattered islands,
    // so the download is widened to a window around the request
    constexpr u64 WindowSize = 512_KB;
    const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
    const VAddr window_end =
        std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
    return {arena, window_start, window_end};
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    const auto download = [this, device_addr, size] {
        const auto [arena, window_start, window_end] = GetReadbackWindow(device_addr, size);
        SettleReadbacks(window_start, window_end);
        DownloadMemory(arena, window_start, window_end - window_start);
    };
    if (assume_locks) {
        // The GPU thread touched the memory itself, so it has to wait for the GPU.
        download();
    } else {
        // The game thread that touched the memory waits for the GPU, not the GPU thread: that
        // only records the copy back and submits it, then carries on with the game's commands.
        // Waiting for the GPU there on every read of GPU written memory took over a third of
        // its time with precise readbacks, which some games need for their effects.
        std::shared_ptr<Readback> readback;
        const auto asked = std::chrono::steady_clock::now();
        liverpool->SendCommand<true>([&] {
            readback_stats.pickup_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - asked)
                                            .count();
            ++readback_stats.pickups;
            readback = StartReadback(device_addr, size);
        });
        if (readback && !FinishReadback(*readback)) {
            // The GPU wrote the memory again after the copy was recorded.
            liverpool->SendCommand<true>(download);
        }
    }
    if (is_write) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
}

void BufferCache::OnBackingWritten(VAddr device_addr, u64 size) {
    memory_tracker->MarkRegionAsCpuModifiedUnlessGpuModified(device_addr, size);
}

std::shared_ptr<BufferCache::Readback> BufferCache::StartReadback(VAddr device_addr, u64 size) {
    PruneReadbacks();
    const auto [arena, window_start, window_end] = GetReadbackWindow(device_addr, size);

    // How far behind the GPU is when game threads read back, and whether the copy could go
    // ahead of the command buffer being recorded, for the summary.
    auto* const work_semaphore = scheduler.GetWorkSemaphore();
    work_semaphore->Refresh();
    const u64 last_submitted = scheduler.CurrentTick() - 1;
    readback_stats.in_flight +=
        last_submitted - std::min(work_semaphore->KnownGpuTick(), last_submitted);
    if (!runtime.IsTouchedInSession(arena, window_start - arena->cpu_addr,
                                    window_end - window_start)) {
        ++readback_stats.untouched;
    }
    if (std::ranges::any_of(compute_ring_writes, [&](const auto& range) {
            return range.first < window_end && window_start < range.second;
        })) {
        ++readback_stats.compute_ring;
    }

    // Game threads mostly read back the same memory again and again, a few times a frame.
    const auto now = std::chrono::steady_clock::now();
    const auto hot = std::ranges::find_if(hot_windows, [&](const HotWindow& window) {
        return window.start == window_start && window.end == window_end;
    });
    if (hot != hot_windows.end()) {
        hot->last_fault = now;
    } else if (hot_windows.size() < MaxHotWindows) {
        hot_windows.push_back({window_start, window_end, now});
    }

    for (const auto& readback : readbacks) {
        if (readback->Done() || readback->end <= window_start || window_end <= readback->start) {
            continue;
        }
        // Another game thread is already waiting for this memory, or it was copied back ahead
        // of this one touching it: wait for the same copy.
        if (!readback->stale && readback->start <= window_start && window_end <= readback->end) {
            ++readback_stats.joined;
            RatePrefetch(*readback, true);
            return readback;
        }
    }
    SettleReadbacks(window_start, window_end);
    // A copy ahead would have spared this fault, so the next chance to make one isn't skipped,
    // even if the last one went stale.
    for (auto& window : hot_windows) {
        if (window.start == window_start && window.end == window_end) {
            window.skip = 0;
        }
    }

    auto readback = RecordReadback(arena, window_start, window_end);
    if (!readback) {
        return nullptr;
    }
    ++readback_stats.on_fault;
    const bool on_graphics = readback->semaphore == scheduler.GetWorkSemaphore();
    SubmitReadbacks(on_graphics, !on_graphics);
    return readback;
}

void BufferCache::SubmitReadbacks(bool on_graphics, bool on_async) {
    // The graphics queue submits what was recorded for the second queue first.
    if (on_graphics) {
        scheduler.Flush();
    } else if (on_async) {
        runtime.FlushAsync();
    }
}

void BufferCache::NoteResidencyChange() {
    runtime.NoteResidencyChange();
}

std::shared_ptr<BufferCache::Readback> BufferCache::RecordReadback(const Buffer* arena, VAddr start,
                                                                   VAddr end) {
    auto readback = std::make_shared<Readback>();
    readback->arena_base = arena->cpu_addr;
    readback->start = start;
    readback->end = end;
    const u64 total_size_bytes = CollectDownloads(arena, start, end - start, readback->copies);
    if (total_size_bytes == 0) {
        return nullptr;
    }
    Common::Perf::ScopedStall stall{Common::Perf::Stall::BufferDownload, total_size_bytes};
    readback->staging =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 0, true);
    for (auto& copy : readback->copies) {
        copy.dstOffset += readback->staging.offset;
    }
    // Copied on the second queue, game threads waiting for it don't wait for the graphics work
    // submitted before it, which the memory doesn't depend on.
    if (runtime.CopyBufferOnAsync(arena, readback->staging.buffer, readback->copies)) {
        Common::Perf::Count(Common::Perf::Counter::AsyncReadbacks);
        auto* const async_scheduler = runtime.AsyncScheduler();
        readback->semaphore = async_scheduler->GetWorkSemaphore();
        readback->tick = async_scheduler->CurrentTick();
    } else {
        runtime.CopyBuffer(arena, readback->staging.buffer, readback->copies);
        readback->semaphore = scheduler.GetWorkSemaphore();
        readback->tick = scheduler.CurrentTick();
    }
    readbacks.push_back(readback);
    return readback;
}

void BufferCache::PrefetchReadbacks() {
    ApplyFinishedReadbacks();
    if (hot_windows.empty()) {
        return;
    }
    // The game is about to read results of the work it was just told is done. Copying the
    // memory it read back recently now, behind that work, means it is mostly written back by
    // the time the game touches it, so its threads neither fault on it nor wait for the GPU.
    const auto now = std::chrono::steady_clock::now();
    std::erase_if(hot_windows,
                  [&](const HotWindow& window) { return now - window.last_fault > HotWindowLife; });
    // Gathered once, as this runs on every fence and checked each window against every copy.
    boost::container::small_vector<std::pair<VAddr, VAddr>, 32> in_flight_ranges;
    for (const auto& readback : readbacks) {
        if (!readback->Done()) {
            in_flight_ranges.emplace_back(readback->start, readback->end);
        }
    }
    boost::container::small_vector<std::shared_ptr<Readback>, 8> prefetched;
    bool on_graphics = false;
    bool on_async = false;
    for (auto& window : hot_windows) {
        if (!memory_tracker->IsRegionGpuModified(window.start, window.end - window.start)) {
            continue;
        }
        const bool in_flight = std::ranges::any_of(in_flight_ranges, [&](const auto& range) {
            return range.first < window.end && window.start < range.second;
        });
        if (in_flight) {
            continue;
        }
        if (window.skip > 0) {
            --window.skip;
            ++readback_stats.skipped;
            continue;
        }
        const auto [arena, start, end] = GetReadbackWindow(window.start, window.end - window.start);
        if (const auto readback = RecordReadback(arena, start, end)) {
            readback->prefetched = true;
            ++readback_stats.prefetched;
            if (readback->semaphore == scheduler.GetWorkSemaphore()) {
                on_graphics = true;
            } else {
                on_async = true;
            }
            in_flight_ranges.emplace_back(readback->start, readback->end);
            prefetched.push_back(readback);
        }
    }
    if (!prefetched.empty()) {
        SubmitReadbacks(on_graphics, on_async);
        // Written back by a thread of their own once the GPU is done with them, instead of by
        // the GPU thread on a later fence: that was hundreds of megabytes a second to copy.
        {
            std::scoped_lock lock{finished_readbacks_mutex};
            finished_readbacks.insert(finished_readbacks.end(), prefetched.begin(),
                                      prefetched.end());
        }
        finished_readbacks_cv.notify_one();
    }

    if (now - last_readback_report >= std::chrono::seconds{10}) {
        last_readback_report = now;
        const double pickups = static_cast<double>(std::max<u64>(readback_stats.pickups, 1));
        LOG_INFO(Render,
                 "Readbacks: {} on game thread faults, {} of those waited for a copy already "
                 "made, {} made ahead, {} written back before the game touched them, {} skipped "
                 "as they kept going stale, {} windows tracked; game threads waited {:.2f} ms on "
                 "average for this thread to take theirs up, {:.1f} command buffers were in "
                 "flight then, {:.0f}% were of memory the one being recorded hadn't touched and "
                 "{:.0f}% of memory the game's compute rings wrote lately",
                 readback_stats.on_fault + readback_stats.joined, readback_stats.joined,
                 readback_stats.prefetched, readback_stats.written_ahead, readback_stats.skipped,
                 hot_windows.size(), static_cast<double>(readback_stats.pickup_ns) / pickups / 1e6,
                 static_cast<double>(readback_stats.in_flight) / pickups,
                 static_cast<double>(readback_stats.untouched) * 100.0 / pickups,
                 static_cast<double>(readback_stats.compute_ring) * 100.0 / pickups);
        readback_stats = {};
    }
}

void BufferCache::ApplyFinishedReadbacks() {
    // Copies made ahead are written back by a thread of their own as soon as the GPU is done
    // with them, and copies made for a game thread by the game thread waiting for them. Only
    // the copies the GPU wrote over again are left to recover here.
    for (const auto& readback : readbacks) {
        if (!readback->Done() && readback->stale) {
            RecoverReadback(*readback);
            RatePrefetch(*readback, false);
        }
    }
    PruneReadbacks();
}

void BufferCache::ReadbackThread(std::stop_token token) {
    Common::SetCurrentThreadName("shadPS4:ReadbackWriter");
    const vk::Device device = instance.GetDevice();
    while (!token.stop_requested()) {
        std::shared_ptr<Readback> readback;
        {
            std::unique_lock lock{finished_readbacks_mutex};
            Common::CondvarWait(finished_readbacks_cv, lock, token,
                                [this] { return !finished_readbacks.empty(); });
            if (finished_readbacks.empty()) {
                continue;
            }
            readback = std::move(finished_readbacks.front());
            finished_readbacks.pop_front();
        }
        // Waits in steps, so that stopping isn't held up by a copy that will never finish.
        const vk::Semaphore semaphore = readback->semaphore->Handle();
        const vk::SemaphoreWaitInfo wait_info = {
            .semaphoreCount = 1,
            .pSemaphores = &semaphore,
            .pValues = &readback->tick,
        };
        vk::Result result = vk::Result::eTimeout;
        while (result == vk::Result::eTimeout && !token.stop_requested() && !readback->Done()) {
            result = device.waitSemaphores(wait_info, 10'000'000);
        }
        // A game thread touching the memory meanwhile may have written it back already, and a
        // copy the GPU wrote over again is recovered by the GPU thread once it sees it is stale.
        if (result == vk::Result::eSuccess) {
            FinishReadback(*readback, true);
        }
    }
}

bool BufferCache::FinishReadback(Readback& readback, bool ahead) {
    if (!ahead) {
        Common::Perf::ScopedStall stall{Common::Perf::Stall::ReadbackWait};
        readback.semaphore->Wait(readback.tick);
    }
    std::scoped_lock lock{readback.mutex};
    if (readback.applied) {
        return !ahead;
    }
    if (readback.recovered || readback.stale) {
        return false;
    }
    readback.staging.Invalidate();
    for (const auto& copy : readback.copies) {
        auto* dst_addr = std::bit_cast<u8*>(readback.arena_base + copy.srcOffset);
        memory->TryWriteBacking(
            dst_addr, readback.staging.mapped + (copy.dstOffset - readback.staging.offset),
            copy.size);
    }
    // The GPU thread marks a readback stale before it marks the memory GPU modified again,
    // which takes the region locks held while checking here, so a newer write keeps its mark.
    const bool unmarked = memory_tracker->UnmarkRegionAsGpuModifiedIf(
        readback.start, readback.end - readback.start, [&] { return !readback.stale.load(); });
    if (!unmarked) {
        return false;
    }
    readback.applied_ahead = ahead;
    readback.applied.store(true, std::memory_order_release);
    return true;
}

void BufferCache::RecoverReadback(Readback& readback) {
    std::scoped_lock lock{readback.mutex};
    if (readback.applied || readback.recovered) {
        return;
    }
    // Its ranges were taken out of the GPU modified ones when the copy was recorded. The copy
    // won't be used, so they are downloaded again with whatever the GPU wrote since.
    for (const auto& copy : readback.copies) {
        gpu_modified_ranges.Add(readback.arena_base + copy.srcOffset, copy.size);
    }
    readback.recovered.store(true, std::memory_order_release);
}

void BufferCache::SettleReadbacks(VAddr start, VAddr end) {
    // Copies back of overlapping memory are finished first, so one doesn't unmark memory the
    // other hasn't written back yet.
    for (const auto& readback : readbacks) {
        if (readback->Done() || readback->end <= start || end <= readback->start) {
            continue;
        }
        // An outdated copy won't be written back, so there is no need to wait for it. Copies
        // made ahead mostly go stale like this, and waiting for them took a tenth of the GPU
        // thread's time.
        if (readback->stale || !FinishReadback(*readback)) {
            RecoverReadback(*readback);
            RatePrefetch(*readback, false);
        } else {
            RatePrefetch(*readback, true);
        }
    }
}

void BufferCache::RatePrefetch(Readback& readback, bool useful) {
    if (!readback.prefetched || std::exchange(readback.rated, true)) {
        return;
    }
    const auto hot = std::ranges::find_if(hot_windows, [&](const HotWindow& window) {
        return window.start == readback.start && window.end == readback.end;
    });
    if (hot == hot_windows.end()) {
        return;
    }
    // Most copies made ahead went stale, the GPU writing the memory again before the game read
    // it, and copied 7-11 GB every 10 s for nothing. Windows that keep going stale are copied at
    // fewer of the chances, twice as few each time, until a copy is used again.
    if (useful) {
        hot->backoff = 0;
    } else {
        hot->backoff = std::min<u8>(static_cast<u8>(hot->backoff * 2 + 1), MaxPrefetchBackoff);
        hot->skip = hot->backoff;
    }
}

void BufferCache::PruneReadbacks() {
    std::erase_if(readbacks, [this](const std::shared_ptr<Readback>& readback) {
        if (!readback->Done()) {
            return false;
        }
        // Copies that went stale were counted against their window when recovered.
        if (readback->applied.load(std::memory_order_acquire)) {
            readback_stats.written_ahead += readback->applied_ahead ? 1 : 0;
            RatePrefetch(*readback, true);
        }
        staging_pool.FreeDeferred(readback->staging);
        return true;
    });
}

u64 BufferCache::CollectDownloads(const Buffer* arena, VAddr device_addr, u64 size,
                                  DownloadCopies& copies) {
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
        for (auto& hint : gpu_modified_hints) {
            if (hint.first < address + size && address < hint.second) {
                hint = {};
            }
        }
    });
    return total_size_bytes;
}

void BufferCache::MarkGpuModifiedRange(VAddr device_addr, u64 size) {
    // Buffers written by every draw or dispatch are usually still marked from the last one, and
    // marking them again changes nothing but costs a tree update. Finding that out in the tree
    // took 0.6k cycles for each of the 1600 buffers written a frame in inFAMOUS Second Son.
    const VAddr end = device_addr + size;
    for (const auto& [hint_start, hint_end] : gpu_modified_hints) {
        if (hint_start <= device_addr && end <= hint_end && hint_start < hint_end) {
            return;
        }
    }
    if (!gpu_modified_ranges.Contains(device_addr, size)) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    gpu_modified_hints[next_gpu_modified_hint] = {device_addr, end};
    next_gpu_modified_hint = (next_gpu_modified_hint + 1) % gpu_modified_hints.size();
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    DownloadCopies copies;
    const u64 total_size_bytes = CollectDownloads(arena, device_addr, size, copies);
    if (total_size_bytes == 0) {
        return;
    }
    // Reading back stalls until the GPU catches up with everything submitted so far.
    Common::Perf::ScopedStall stall{Common::Perf::Stall::BufferDownload, total_size_bytes};
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    scheduler.Finish();

    download.buffer->Invalidate(download.offset, download.size);
    const VAddr arena_base = arena->cpu_addr;
    for (const auto& copy : copies) {
        auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
        memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                copy.size);
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer,
                                                        bool is_read_tracked, bool* unwritten) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        // Memory the GPU has an up to date copy of is bound from that copy rather than copied
        // again on every use. Only when the read is reported to the runtime, so an upload over
        // it later waits for it, and no write to it is pending, as waiting for that would end
        // the render pass. Texel buffers are left alone, as an image they alias may have been
        // copied over their GPU copy.
        if (is_read_tracked && !is_texel_buffer && size != 0) {
            Common::Perf::Count(Common::Perf::Counter::SmallBuffers);
            if (memory_tracker->IsRegionUploaded(device_addr, size)) {
                const u64 first_block = device_addr >> block_shift;
                const u64 last_block = (device_addr + size - 1) >> block_shift;
                const auto* arena = GetArena(first_block, last_block);
                const u64 offset = arena->Offset(device_addr);
                if (!runtime.IsBufferAccessed(arena, offset, size)) {
                    Common::Perf::Count(Common::Perf::Counter::SmallBuffersInPlace);
                    if (unwritten) {
                        *unwritten = true;
                    }
                    return {arena, offset};
                }
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        // Nothing writes the stream buffer.
        if (unwritten) {
            *unwritten = true;
        }
        return {&stream_buffer, offset};
    }
    // Memory the game writes all of again frame after frame, like the buffers it fills for its
    // draws, is copied for the draw as small buffers are, and left unprotected. Uploading it and
    // protecting it again made the game fault on each of its pages again the next frame, and
    // nearly half of its faults were on pages that had faulted in that or the frame before. The
    // upload would have copied all of it too.
    if (!is_written && !is_texel_buffer && is_read_tracked && size <= REWRITE_STREAM_THRESHOLD &&
        memory_tracker->IsRegionRewritten(device_addr, size) &&
        !IsRegionGpuModified(device_addr, size) && TakeRewriteCopy(device_addr, size)) {
        Common::Perf::Count(Common::Perf::Counter::RewrittenBuffersCopied);
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        if (unwritten) {
            *unwritten = true;
        }
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    bool copied_back = false;
    if (is_written) {
        // Before the memory is marked GPU modified, so a game thread about to apply an older copy
        // of it back sees this under the region lock that marking takes.
        for (const auto& readback : readbacks) {
            if (readback->start < device_addr + size && device_addr < readback->end) {
                readback->stale.store(true);
                copied_back = true;
            }
        }
    }
    // Buffers written by every dispatch are mostly still marked GPU modified, and none of them CPU
    // modified, from the last one, so marking them again changes nothing. It took the region
    // lock, which game threads hold while their faults change page protection, and waiting for
    // them there was 2% of the GPU thread. Only copies back clear the mark from other threads,
    // and this one starts them all: with none of the memory being copied back, no thread
    // changes it meanwhile. Large ranges read from may have been uploaded lately enough too.
    const bool synced = is_written
                            ? !is_texel_buffer && !copied_back &&
                                  memory_tracker->IsRegionMarkedGpuWritten(device_addr, size)
                            : size >= LARGE_READ_THRESHOLD && LargeReadSynced(device_addr, size);
    if (!synced) {
        // Also copies in an image the texel buffer aliases, so that isn't repeated here.
        SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    }
    if (is_written) {
        MarkGpuModifiedRange(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

bool BufferCache::LargeReadSynced(VAddr device_addr, u64 size) {
    // inFAMOUS Second Son binds a texel buffer over 2.5 GB of its memory a thousand times a frame,
    // and another over 0.5 GB, and uploading what the CPU wrote in them for each walked the state
    // of every 16 MB of them: 12% of the GPU thread. Games only write memory for the GPU to read
    // before submitting the command buffer reading it, or before something it waits for, see
    // Liverpool::sync_count, so uploading the range once in between uploads all they wrote for it.
    // Writes made meanwhile could be seen or not by the GPU on the console too, and are uploaded
    // once it gets to the next. The count changes 130 to 150 times a frame there, and the ranges
    // are uploaded once for every 40 times they are bound.
    if (!sync_large_reads_once) {
        return false;
    }
    auto& large_read = large_reads[((device_addr >> 12) ^ (size >> 20)) % large_reads.size()];
    if (large_read.address == device_addr && large_read.size == size &&
        large_read.sync_count == liverpool->sync_count) {
        return true;
    }
    large_read = {device_addr, size, liverpool->sync_count};
    return false;
}

bool BufferCache::TakeRewriteCopy(VAddr device_addr, u64 size) {
    const u64 frame = Common::Perf::FrameNumber();
    if (frame != rewrite_copy_frame) {
        rewrite_copy_frame = frame;
        rewrite_copy_bytes = 0;
    }
    if (rewrite_copy_bytes + size > MaxRewriteCopyBytes) {
        return false;
    }
    // Memory many draws bind, like constants of a whole pass, is uploaded once a frame instead,
    // as copying it for each draw would cost more than its faults.
    auto& copy = rewrite_copies[(device_addr >> 8) % rewrite_copies.size()];
    if (copy.address == device_addr && copy.frame == frame) {
        return false;
    }
    copy = {device_addr, frame};
    rewrite_copy_bytes += size;
    return true;
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    // Draws whose shaders read memory freely often come one after another with nothing written
    // by the CPU in between, and going over all memory in use for each was for nothing then.
    const u64 generation = cpu_modified_generation.load(std::memory_order_acquire);
    if (generation == dma_synced_generation) {
        return;
    }
    Common::Perf::ScopedStall stall{Common::Perf::Stall::DmaSync};
    Common::Perf::Count(Common::Perf::Counter::DmaSyncs);
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
    dma_synced_generation = generation;
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    // Granules never cross an arena page, so the widened range stays within the arena.
    const u64 granule_blocks = std::max<u64>(RESIDENCY_GRANULE_SIZE >> block_shift, 1);
    const u64 bind_first = Common::AlignDown(first_block, granule_blocks);
    const u64 bind_end = Common::AlignUp(last_block + 1, granule_blocks);

    // Memory is only ever made resident, so a range found resident stays so. Searching the
    // resident ranges for every buffer bound took 1% of the GPU thread.
    if (std::ranges::any_of(resident_hints, [&](const std::pair<u64, u64>& hint) {
            return hint.first <= bind_first && bind_end <= hint.second;
        })) {
        return;
    }
    if (const auto it = resident_ranges.Find(bind_first);
        it != resident_ranges.end() && bind_end <= it->end) {
        resident_hints[next_resident_hint] = {it->start, it->end};
        next_resident_hint = (next_resident_hint + 1) % resident_hints.size();
        return;
    }

    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(bind_first, bind_end, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    Common::Perf::ScopedStall stall{Common::Perf::Stall::Residency,
                                    u64{resident_blocks} << block_shift};
    // Memory that comes into use may hold what the CPU wrote, which isn't on the GPU yet.
    cpu_modified_generation.fetch_add(1, std::memory_order_release);

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        const u64 range_size = (range.end - range.start) << block_shift;
        const auto [device_memory, memory_offset] = AllocateResidency(range_size);

        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);

        LOG_DEBUG(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = range_size,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

std::pair<vk::DeviceMemory, u64> BufferCache::AllocateResidency(u64 size) {
    const auto allocate = [this](u64 allocation_size) {
        const vk::MemoryAllocateInfo alloc_info = {
            .allocationSize = allocation_size,
            .memoryTypeIndex = arena_memory_type_index,
        };
        return Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));
    };
    // Ranges keep coming as the game streams data in. Carving them out of large chunks keeps the
    // driver from allocating memory each time, which is slow and limited to a few thousand
    // allocations on some drivers. Large ranges get their own.
    if (size > RESIDENCY_CHUNK_SIZE / 4) {
        return {allocate(size), 0};
    }
    if (!residency_chunk || residency_chunk_used + size > RESIDENCY_CHUNK_SIZE) {
        residency_chunk = allocate(RESIDENCY_CHUNK_SIZE);
        residency_chunk_used = 0;
    }
    const u64 offset = residency_chunk_used;
    residency_chunk_used += size;
    return {residency_chunk, offset};
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    // Memory only read is protected again and copied on the upload thread. Memory about to be
    // written is protected from reads here: a game thread reading it after the GPU thread got
    // past what tells it the GPU is done has to fault, to get it back.
    bool defer =
        defer_uploads && !is_written &&
        host_uploads->pending_bytes.load(std::memory_order_relaxed) < MaxPendingUploadBytes;
    HostUpload upload;
    if (defer) {
        PageManager::DeferProtection(&upload.protects);
    }
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
        copies.emplace_back(total_size_bytes, addr, size);
        total_size_bytes += size;
    });
    if (defer) {
        PageManager::DeferProtection(nullptr);
        if (total_size_bytes > MaxDeferredUploadBytes) {
            // Too large to leave, it is protected and copied here after all.
            for (const auto& [first_page, num_pages] : upload.protects) {
                page_manager.ApplyProtection(first_page, num_pages);
            }
            upload.protects.clear();
            defer = false;
        }
    }
    if (!defer && total_size_bytes > MaxUploadPartBytes) {
        Common::Perf::ScopedStall stall{Common::Perf::Stall::BufferUpload, total_size_bytes};
        UploadInParts(arena, copies);
    } else if (!copies.empty()) {
        Common::Perf::ScopedStall stall{Common::Perf::Stall::BufferUpload, total_size_bytes};
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            if (defer) {
                upload.copies.emplace_back(copy.dstOffset, staging.mapped + copy.srcOffset,
                                           copy.size);
            } else {
                memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset,
                                         copy.size);
            }
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        if (defer) {
            upload.staging = staging;
        } else {
            staging.Flush();
        }
        runtime.UploadBuffer(staging.buffer, arena, copies);
    }
    if (defer && (!upload.protects.empty() || !upload.copies.empty())) {
        auto& uploads = *host_uploads;
        {
            std::scoped_lock lk{uploads.mutex};
            uploads.queue.push_back(std::move(upload));
            uploads.pending_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
            uploads.queued.fetch_add(1, std::memory_order_release);
        }
        uploads.cv.notify_one();
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

void BufferCache::UploadInParts(const Buffer* arena, std::span<const vk::BufferCopy> ranges) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    Vulkan::StagingBufferRef staging{};
    u64 part_bytes = 0;
    const auto upload_part = [&] {
        staging.Flush();
        runtime.UploadBuffer(staging.buffer, arena, copies);
        copies.clear();
        part_bytes = 0;
    };
    bool first_part = true;
    for (const auto& range : ranges) {
        for (u64 done = 0; done < range.size;) {
            if (part_bytes == 0) {
                if (!first_part) {
                    // The staging memory is free again once the GPU copied the last part.
                    scheduler.Finish();
                }
                first_part = false;
                staging = staging_pool.Request(MaxUploadPartBytes, MemoryType::HostUncached);
            }
            const u64 size = std::min(range.size - done, MaxUploadPartBytes - part_bytes);
            const VAddr addr = range.dstOffset + done;
            memory->CopySparseMemory(addr, staging.mapped + part_bytes, size);
            copies.emplace_back(staging.offset + part_bytes, addr - arena->cpu_addr, size);
            part_bytes += size;
            done += size;
            if (part_bytes == MaxUploadPartBytes) {
                upload_part();
            }
        }
    }
    if (part_bytes != 0) {
        upload_part();
    }
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    const auto start = std::chrono::steady_clock::now();
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    const auto elapsed = std::chrono::steady_clock::now() - start;
    Common::Perf::Record(Common::Perf::Stall::SparseBind,
                         std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    if (elapsed >= std::chrono::milliseconds{2}) {
        u64 bound_blocks{};
        for (const auto& range : resident_ranges) {
            bound_blocks += range.end - range.start;
        }
        LOG_INFO(Render, "Sparse bind took {:.1f} ms, {} MB resident in total",
                 std::chrono::duration<double, std::milli>(elapsed).count(),
                 (bound_blocks << block_shift) >> 20);
    }

    pending_binds.clear();
}

} // namespace VideoCore
