// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    /// A flattened user data dword a specialization reads a sharp from, and what of it counts.
    /// Specializations only look at formats, strides, types and such, and at whether a sharp
    /// is set, not at addresses or sizes, which change from draw to draw with the same
    /// permutation. Comparing those made most lookups miss.
    struct SharpDword {
        u16 index;
        /// Whether the dword being zero or not counts, apart from the bits in mask.
        bool nonzero;
        /// The bits that count.
        u32 mask;

        bool Same(u32 a, u32 b) const noexcept {
            return ((a ^ b) & mask) == 0 && (!nonzero || (a == 0) == (b == 0));
        }
    };

    /// What a recent lookup of a permutation was given. Draws mostly use a program the way one
    /// of the few draws before did, and then the permutation is known without building a
    /// specialization to compare, which took a tenth of the GPU thread.
    struct LastLookup {
        bool valid{};
        size_t perm_idx{};
        Shader::Backend::Bindings start{};
        /// The flattened user data dwords listed in sharp_dwords, or all of them without a list.
        /// Kept in place with what is compared before it: lookups that don't match mostly differ
        /// here, and each of them going to memory of its own for these was a good part of
        /// looking up a permutation.
        boost::container::small_vector<u32, 32> user_data;
        Shader::RuntimeInfo runtime_info{};
        /// The fetch shader and the vertex buffer sharps it loads, which are read from memory
        /// rather than from the flattened user data.
        boost::container::small_vector<u32, 32> fetch_code;
        boost::container::small_vector<u32, 32> vertex_sharps;

        bool Matches(const Shader::Info& info, const Shader::RuntimeInfo& runtime_info_,
                     const Shader::Backend::Bindings& start_,
                     const Shader::Gcn::FetchShaderData& fetch,
                     const std::vector<SharpDword>* sharp_dwords) const;
        void Remember(const Shader::Info& info, const Shader::RuntimeInfo& runtime_info_,
                      const Shader::Backend::Bindings& start_, size_t perm_idx_,
                      const Shader::Gcn::FetchShaderData& fetch,
                      const std::vector<SharpDword>* sharp_dwords);
    };

    /// Programs drawn with many materials in turn alternate between as many lookups. Going
    /// through the last eight, newest first, took 2.7 comparisons a lookup in inFAMOUS Second
    /// Son's city, and 13% of lookups still missed all eight and built a specialization: a
    /// third of the comparisons. Lookups are kept in slots chosen by a hash of what is compared
    /// first, two a hash, and the slot that matched last is tried before hashing.
    static constexpr size_t NumLookupSlots = 16;

    Shader::Info info;
    ModuleList modules{};
    std::array<LastLookup, NumLookupSlots> last_lookups{};
    /// When each slot last matched or was filled in, by lookup_clock.
    std::array<u64, NumLookupSlots> slot_used{};
    u64 lookup_clock{};
    size_t last_hit_slot{};
    /// The flattened user data dwords that specializations read sharps from. The others, like
    /// pointers and constants that change from draw to draw, can't change the permutation.
    std::vector<SharpDword> sharp_dwords;
    bool sharp_dwords_found{};
    /// Set when a sharp is read from past the flattened user data, so all of it is compared.
    bool compare_all_dwords{};

    /// Lists the dwords sharps are read from, once the resources of the program are known.
    void FindSharpDwords();

    /// The dwords lookups compare, or nullptr for all of them.
    const std::vector<SharpDword>* LookupDwords() const {
        return compare_all_dwords ? nullptr : &sharp_dwords;
    }

    /// The first of the two slots a lookup with these inputs is kept in, from the parts of them
    /// LastLookup::Matches compares first. The other is the one after it.
    size_t LookupSlot(const Shader::Info& info, const Shader::Backend::Bindings& start) const;

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    bool is_indirect;
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {});

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);

    /// The parameters of a stage's program, as AmdGpu::GetParams finds them. Every stage of every
    /// draw read the start of its program's code to find its binary info at the end, and then
    /// the binary info, both in guest memory rarely in the CPU's caches: 1.5-2% of the GPU
    /// thread. What the binary info says is remembered for the code's address, and used again
    /// until the game could have changed the code, see Liverpool::sync_count. Then only the
    /// binary info is read, where it was, after making sure it is binary info, so a program
    /// loaded there since is found as it is.
    template <typename ProgramRegs>
    [[nodiscard]] Shader::ShaderParams ProgramParams(const ProgramRegs& pgm);

    /// Returns the pipeline once it can be used, or null to skip draws while it compiles.
    const GraphicsPipeline* ReadyGraphicsPipeline(GraphicsPipeline* pipeline);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    /// The program looked up last for each hardware stage, with its hash.
    std::array<std::pair<u64, Program*>, static_cast<size_t>(Shader::HwStage::Compute) + 1>
        last_programs{};
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    /// The pipeline found last and its key.
    GraphicsPipelineKey last_graphics_key{};
    GraphicsPipeline* last_graphics_pipeline{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start
    bool async_shader_compile{};
    /// Where the binary info of the program at an address was found, in dwords from its code,
    /// and what it said, as of Liverpool::sync_count.
    struct BinaryInfoHint {
        const u32* code{};
        u32 offset{};
        u32 length{};
        u64 hash{};
        u64 sync_count{};
    };
    std::array<BinaryInfoHint, 4096> binary_info_hints{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    // Declared last so it goes first on destruction, dropping queued compiles before the
    // pipelines and the Vulkan pipeline cache they refer to.
    std::unique_ptr<PipelineCompiler> compiler;
};

} // namespace Vulkan
