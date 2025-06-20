// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <mutex>
#include <span>
#include <utility>

#ifdef __linux__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/types.h"
#include "video_core/page_manager.h"

namespace VideoCore {

constexpr u64 PAGES_PER_WORD = 64;
constexpr u64 BYTES_PER_PAGE = 4_KB;
constexpr u64 BYTES_PER_WORD = PAGES_PER_WORD * BYTES_PER_PAGE;

constexpr u64 HIGHER_PAGE_BITS = 22;
constexpr u64 HIGHER_PAGE_SIZE = 1ULL << HIGHER_PAGE_BITS;
constexpr u64 HIGHER_PAGE_MASK = HIGHER_PAGE_SIZE - 1ULL;
constexpr u64 NUM_REGION_WORDS = HIGHER_PAGE_SIZE / BYTES_PER_WORD;

enum class Type {
    CPU, ///< Set if CPU page data is more up-to-date than GPU data.
    GPU, ///< Set if GPU page data is more up-to-date than CPU data.
};

using WordsArray = std::array<u64, NUM_REGION_WORDS>;

/**
 * Allows tracking CPU and GPU modification of pages in a contigious 4MB virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_} {
        cpu.fill(~u64{0});
        gpu.fill(0);
        write.fill(~u64{0});
        read.fill(~u64{0});
    }
    explicit RegionManager() = default;

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    VAddr GetCpuAddr() const {
        return cpu_addr;
    }

    static constexpr u64 ExtractBits(u64 word, size_t page_start, size_t page_end) {
        constexpr size_t number_bits = sizeof(u64) * 8;
        const size_t limit_page_end = number_bits - std::min(page_end, number_bits);
        u64 bits = (word >> page_start) << page_start;
        bits = (bits << limit_page_end) >> limit_page_end;
        return bits;
    }

    static constexpr std::pair<size_t, size_t> GetWordPage(VAddr address) {
        const size_t converted_address = static_cast<size_t>(address);
        const size_t word_number = converted_address / BYTES_PER_WORD;
        const size_t amount_pages = converted_address % BYTES_PER_WORD;
        return std::make_pair(word_number, amount_pages / BYTES_PER_PAGE);
    }

    template <typename Func>
    void IterateWords(size_t offset, size_t size, Func&& func) const {
        using FuncReturn = std::invoke_result_t<Func, std::size_t, u64>;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        const size_t start = static_cast<size_t>(std::max<s64>(static_cast<s64>(offset), 0LL));
        const size_t end = static_cast<size_t>(std::max<s64>(static_cast<s64>(offset + size), 0LL));
        if (start >= HIGHER_PAGE_SIZE || end <= start) {
            return;
        }
        auto [start_word, start_page] = GetWordPage(start);
        auto [end_word, end_page] = GetWordPage(end + BYTES_PER_PAGE - 1ULL);
        constexpr size_t num_words = NUM_REGION_WORDS;
        start_word = std::min(start_word, num_words);
        end_word = std::min(end_word, num_words);
        const size_t diff = end_word - start_word;
        end_word += (end_page + PAGES_PER_WORD - 1ULL) / PAGES_PER_WORD;
        end_word = std::min(end_word, num_words);
        end_page += diff * PAGES_PER_WORD;
        constexpr u64 base_mask{~0ULL};
        for (size_t word_index = start_word; word_index < end_word; word_index++) {
            const u64 mask = ExtractBits(base_mask, start_page, end_page);
            start_page = 0;
            end_page -= PAGES_PER_WORD;
            if constexpr (BOOL_BREAK) {
                if (func(word_index, mask)) {
                    return;
                }
            } else {
                func(word_index, mask);
            }
        }
    }

    void IteratePages(u64 mask, auto&& func) const {
        size_t offset = 0;
        while (mask != 0) {
            const size_t empty_bits = std::countr_zero(mask);
            offset += empty_bits;
            mask >>= empty_bits;

            const size_t continuous_bits = std::countr_one(mask);
            func(offset, continuous_bits);
            mask = continuous_bits < PAGES_PER_WORD ? (mask >> continuous_bits) : 0;
            offset += continuous_bits;
        }
    }

    /**
     * Change the state of a range of pages
     * - If the CPU data is modified, stop tracking writes to allow guest to write to the page.
     * - If the CPU data is not modified, track writes to get notified when a write does occur.
     * - If the GPU data is modified, track both reads and writes to wait for any pending GPU
     *   downloads before any CPU access.
     * - If the GPU data is not modified, track writes, to get notified when a write does occur.
     * @param dirty_addr    Base address to mark or unmark as modified
     * @param size          Size in bytes to mark or unmark as modified
     */
    template <Type type, bool is_dirty>
    void ChangeRegionState(u64 dirty_addr, u64 size) noexcept {
        std::scoped_lock lk{lock};
        IterateWords(dirty_addr - cpu_addr, size, [&](size_t index, u64 mask) {
            if constexpr (type == Type::CPU) {
                UpdateProtection<!is_dirty>(index, write[index], mask);
                if constexpr (is_dirty) {
                    cpu[index] |= mask;
                    write[index] |= mask;
                } else {
                    cpu[index] &= ~mask;
                    write[index] &= ~mask;
                }
            } else {
                UpdateProtection<true>(index, write[index], mask);
                UpdateProtection<is_dirty, true>(index, read[index], mask);
                write[index] &= ~mask;
                if constexpr (is_dirty) {
                    gpu[index] |= mask;
                    read[index] &= ~mask;
                } else {
                    gpu[index] &= ~mask;
                    read[index] |= mask;
                }
            }
        });
    }

    /**
     * Loop over each page in the given range, turn off those bits and notify the tracker if
     * needed. Call the given function on each turned off range.
     * - If looping over CPU pages the clear flag will re-enable write protection.
     * - If looping over GPU pages the clear flag will disable read protection.
     * @param query_cpu_range Base CPU address to loop over
     * @param size            Size in bytes of the CPU range to loop over
     * @param func            Function to call for each turned off region
     */
    template <Type type, bool clear>
    void ForEachModifiedRange(VAddr query_cpu_range, s64 size, auto&& func) {
        std::scoped_lock lk{lock};
        std::span<u64> state_words = Span<type>();
        const size_t offset = query_cpu_range - cpu_addr;
        bool pending = false;
        size_t pending_offset{};
        size_t pending_pointer{};
        const auto release = [&]() {
            func(cpu_addr + pending_offset * BYTES_PER_PAGE,
                 (pending_pointer - pending_offset) * BYTES_PER_PAGE);
        };
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            const u64 word = state_words[index] & mask;
            if constexpr (clear) {
                if constexpr (type == Type::CPU) {
                    UpdateProtection<true>(index, write[index], mask);
                    write[index] &= ~mask;
                } else {
                    UpdateProtection<false, true>(index, read[index], mask);
                    read[index] |= mask;
                }
                state_words[index] &= ~mask;
            }
            const size_t base_offset = index * PAGES_PER_WORD;
            IteratePages(word, [&](size_t pages_offset, size_t pages_size) {
                const auto reset = [&]() {
                    pending_offset = base_offset + pages_offset;
                    pending_pointer = base_offset + pages_offset + pages_size;
                };
                if (!pending) {
                    reset();
                    pending = true;
                    return;
                }
                if (pending_pointer == base_offset + pages_offset) {
                    pending_pointer += pages_size;
                    return;
                }
                release();
                reset();
            });
        });
        if (pending) {
            release();
        }
    }

    /**
     * Returns true when a region has been modified
     *
     * @param offset Offset in bytes from the start of the buffer
     * @param size   Size in bytes of the region to query for modifications
     */
    template <Type type>
    [[nodiscard]] bool IsRegionModified(u64 offset, u64 size) const noexcept {
        const std::span<const u64> state_words = Span<type>();
        bool result = false;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            const u64 word = state_words[index] & mask;
            if (word != 0) {
                result = true;
                return true;
            }
            return false;
        });
        return result;
    }

private:
    /**
     * Notify tracker about changes in the CPU tracking state of a word in the buffer
     *
     * @param word_index   Index to the word to notify to the tracker
     * @param access_bits  Bits of pages of word_index that are unprotected.
     * @param mask         Which bits from access_bits are relevant.
     *
     * @tparam add_to_tracker True when the tracker should start tracking the new pages
     */
    template <bool add_to_tracker, bool is_read = false>
    void UpdateProtection(u64 word_index, u64 access_bits, u64 mask) const {
        constexpr s32 delta = add_to_tracker ? 1 : -1;
        const u64 changed_bits = (add_to_tracker ? access_bits : ~access_bits) & mask;
        const VAddr addr = cpu_addr + word_index * BYTES_PER_WORD;
        IteratePages(changed_bits, [&](size_t offset, size_t size) {
            tracker->UpdatePageWatchers<delta, is_read>(addr + offset * BYTES_PER_PAGE,
                                                        size * BYTES_PER_PAGE);
        });
    }

    template <Type type>
    std::span<u64> Span() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    template <Type type>
    std::span<const u64> Span() const noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
    Common::AdaptiveMutex lock;
#else
    Common::SpinLock lock;
#endif
    PageManager* tracker;
    VAddr cpu_addr = 0;
    WordsArray cpu;
    WordsArray gpu;
    WordsArray write;
    WordsArray read;
};

} // namespace VideoCore
