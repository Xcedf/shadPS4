// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
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

namespace VideoCore {
class BufferCache;
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

    struct SharpDword {
        u16 index;
        bool nonzero;
        u32 mask;

        bool Same(u32 a, u32 b) const noexcept {
            return ((a ^ b) & mask) == 0 && (!nonzero || (a == 0) == (b == 0));
        }
    };
    struct LastLookup {
        bool valid{};
        size_t perm_idx{};
        VAddr pgm_base{};
        Shader::Backend::Bindings start{};
        Shader::RuntimeInfo runtime_info{};
        std::vector<u32> user_data;
        /// The fetch shader and the vertex buffer sharps it loads, which are read from memory
        /// rather than from the flattened user data.
        std::vector<u32> fetch_code;
        std::vector<u32> vertex_sharps;

        bool Matches(const Shader::Info& info, const Shader::RuntimeInfo& runtime_info_,
                     const Shader::Backend::Bindings& start_,
                     const Shader::Gcn::FetchShaderData& fetch,
                     const std::vector<SharpDword>* sharp_dwords) const;
        void Remember(const Shader::Info& info, const Shader::RuntimeInfo& runtime_info_,
                      const Shader::Backend::Bindings& start_, size_t perm_idx_,
                      const Shader::Gcn::FetchShaderData& fetch,
                      const std::vector<SharpDword>* sharp_dwords);
    };

    static constexpr size_t NumLastLookups = 4;
    Shader::Info info;
    ModuleList modules{};
    std::array<LastLookup, NumLastLookups> last_lookups{};
    size_t next_last_lookup{};
    std::vector<SharpDword> sharp_dwords;
    bool sharp_dwords_found{};
    bool compare_all_dwords{};

    /// Lists the dwords sharps are read from, once the resources of the program are known.
    void FindSharpDwords();

    /// The dwords lookups compare, or nullptr for all of them.
    const std::vector<SharpDword>* LookupDwords() const {
        return compare_all_dwords ? nullptr : &sharp_dwords;
    }

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
                           AmdGpu::Liverpool* liverpool, VideoCore::BufferCache& buffer_cache,
                           u32 sparse_page_shift);
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
    void InitializeDescriptorReader(Shader::Info& info);

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

    /// Returns the pipeline once it can be used, or null to skip draws while it compiles.
    const GraphicsPipeline* ReadyGraphicsPipeline(GraphicsPipeline* pipeline);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    VideoCore::BufferCache& buffer_cache;
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
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    GraphicsPipelineKey last_graphics_key{};
    GraphicsPipeline* last_graphics_pipeline{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start
    bool async_shader_compile{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
    std::unique_ptr<PipelineCompiler> compiler;
};

} // namespace Vulkan
