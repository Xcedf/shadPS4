// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <ranges>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/perf_profiler.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

template <typename ProgramRegs>
Shader::ShaderParams PipelineCache::ProgramParams(const ProgramRegs& pgm) {
    const u32* code = pgm.template Address<u32*>();
    // Programs are placed at large alignments, so their addresses only differ in high bits.
    auto& hint = binary_info_hints[(reinterpret_cast<uintptr_t>(code) * 0x9E3779B97F4A7C15ULL) >>
                                   (64 - std::countr_zero(binary_info_hints.size()))];
    // Read since the game last could have changed it, see Liverpool::sync_count.
    if (hint.code != code || hint.sync_count != liverpool->sync_count) {
        const AmdGpu::BinaryInfo* bininfo = nullptr;
        if (hint.code == code) {
            const auto* hinted = std::bit_cast<const AmdGpu::BinaryInfo*>(code + hint.offset);
            if (hinted->Valid()) {
                bininfo = hinted;
            }
        }
        if (!bininfo) {
            bininfo = &AmdGpu::SearchBinaryInfo(code);
        }
        hint = {
            .code = code,
            .offset = static_cast<u32>(reinterpret_cast<const u32*>(bininfo) - code),
            .length = bininfo->length,
            .hash = bininfo->shader_hash,
            .sync_count = liverpool->sync_count,
        };
    }
    return {
        .user_data = pgm.user_data,
        .code = std::span{code, hint.length / sizeof(u32)},
        .hash = hint.hash,
    };
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(HwStage stage, SwStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.dx10_clamp = program.settings.dx10_clamp;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(SwStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.hw.ls.ls_stride = tess_constants.ls_stride;
        break;
    }
    case HwStage::Hull:
        BuildCommon(regs.hs_program);
        break;
    case HwStage::Export:
        BuildCommon(regs.es_program);
        info.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        break;
    case HwStage::Geometry: {
        BuildCommon(regs.gs_program);
        info.hw.gs.num_outputs = MapOutputs(info.hw.gs.outputs, regs.vs_output_control);
        info.hw.gs.output_vertices = regs.vgt_gs_max_vert_out;
        info.hw.gs.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            info.hw.gs.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            info.hw.gs.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            info.hw.gs.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        info.hw.gs.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        info.hw.gs.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        info.hw.gs.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        info.hw.gs.vs_copy = params_vc.code;
        info.hw.gs.vs_copy_hash = params_vc.hash;
        DumpShader(info.hw.gs.vs_copy, info.hw.gs.vs_copy_hash, Shader::HwStage::Vertex, 0,
                   "copy.bin");
        break;
    }
    case HwStage::Vertex: {
        BuildCommon(regs.vs_program);
        info.hw.vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.hw.vs.num_outputs = MapOutputs(info.hw.vs.outputs, regs.vs_output_control);
        info.hw.vs.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.hw.vs.clip_disable = regs.IsClipDisabled();
        break;
    }
    case HwStage::Fragment: {
        BuildCommon(regs.ps_program);
        info.hw.fs.en_flags = regs.ps_input_ena;
        info.hw.fs.addr_flags = regs.ps_input_addr;
        info.hw.fs.num_inputs = regs.num_interp;
        info.hw.fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
        info.hw.fs.depth_before_shader = regs.depth_shader_control.depth_before_shader;
        info.hw.fs.num_samples =
            regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
                ? regs.aa_config.NumSamples()
                : 1;
        info.hw.fs.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.hw.fs.mrtz_mask = regs.depth_shader_control.z_export_enable |
                               (stencil_ref_export_enable << 1) |
                               (regs.depth_shader_control.mask_export_enable << 2) |
                               (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.hw.fs.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.hw.fs.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.hw.fs.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.hw.fs.inputs[i] = {
                .param_index = u16(ps_inputs[i].input_offset),
                .is_default = u16(ps_inputs[i].use_default),
                .is_flat = u16(ps_inputs[i].flat_shade),
                .default_value = u16(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.hw.fs.color_buffers[i] = graphics_key.color_buffers[i];
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry));
        info.hw.fs.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case HwStage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.props.num_user_data = cs_pgm.settings.num_user_regs;
        info.props.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs_pgm.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs_pgm.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs_pgm.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs_pgm.settings.fp_round_mode64;
        info.props.dx10_clamp = cs_pgm.settings.dx10_clamp;
        info.hw.cs.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                     cs_pgm.num_thread_z.full};
        info.hw.cs.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                  cs_pgm.IsTgidEnabled(2)};
        info.hw.cs.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    switch (l_stage) {
    case SwStage::Vertex:
        info.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.sw.vs.vertex_sgpr_offset = draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.is_indirect = draw_indirect_params.is_indirect;
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        break;
    case SwStage::TessellationControl: {
        info.sw.tcs.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.sw.tcs.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.sw.tcs.tess_type = regs.tess_config.type;
        info.sw.tcs.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        break;
    }
    case SwStage::TessellationEval: {
        info.sw.tes.tess_type = regs.tess_config.type;
        info.sw.tes.tess_topology = regs.tess_config.topology;
        info.sw.tes.tess_partitioning = regs.tess_config.partitioning;
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance_.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);

    async_shader_compile = EmulatorSettings.IsAsyncShaderCompile();
    compiler = std::make_unique<PipelineCompiler>(PipelineCompiler::DefaultNumWorkers());
    LOG_INFO(Render_Vulkan, "Compiling pipelines on {} threads, async shader compile: {}",
             compiler->NumWorkers(), async_shader_compile);

    WarmUp();
}

PipelineCache::~PipelineCache() = default;

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params) {
    draw_indirect_params = params;
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    // Draws mostly use the pipeline the draw before did, found without hashing the key again.
    if (last_graphics_pipeline && graphics_key == last_graphics_key) {
        return ReadyGraphicsPipeline(last_graphics_pipeline);
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false,
            async_shader_compile ? compiler.get() : nullptr);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
    }
    GraphicsPipeline* pipeline = it->second.get();
    last_graphics_key = graphics_key;
    last_graphics_pipeline = pipeline;
    return ReadyGraphicsPipeline(pipeline);
}

const GraphicsPipeline* PipelineCache::ReadyGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (!pipeline->IsReady()) {
        // A pipeline met in game is still compiling: skip its draws for these few frames instead
        // of stalling the GPU thread. Pipelines from the pipeline cache are waited for, since
        // they are usually done or about to be.
        if (async_shader_compile && !pipeline->IsPreloaded()) {
            return nullptr;
        }
        pipeline->WaitReady();
    }
    return pipeline;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false, compiler.get());
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();

        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == AmdGpu::BlendControl::BlendFunc::Min ||
             bc.color_func == AmdGpu::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == AmdGpu::BlendControl::BlendFactor::DstColor;
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = nullptr;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = ProgramParams(*pgm);
        std::tie(infos[stage_out_idx], modules[stage_out_idx], key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    bind_stage(HwStage::Fragment, SwStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(HwStage::Vertex, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    default:
        LOG_WARNING(Render_Vulkan, "unimplemented shader stage {}", (u32)regs.stage_enable.raw);
        return false;
    }

    const auto* vs_info = infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = ProgramParams(cs_pgm);
    std::tie(infos[0], modules[0], compute_key.value) =
        GetProgram(HwStage::Compute, SwStage::Compute, cs_params, binding);
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");

    const auto start = std::chrono::steady_clock::now();
    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    Common::Perf::Record(Common::Perf::Stall::ShaderTranslate,
                         std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    if (elapsed >= std::chrono::milliseconds{3}) {
        LOG_INFO(Render_Vulkan, "Translated {} shader {:#x} in {:.1f} ms", info.hw_stage,
                 info.pgm_hash, std::chrono::duration<double, std::milli>(elapsed).count());
    }
    if ((info.uses_buffer_atomic_float_min_max && !profile.supports_buffer_fp32_atomic_min_max) ||
        (info.uses_image_atomic_float_min_max && !profile.supports_image_fp32_atomic_min_max)) {
        LOG_INFO(Render_Vulkan, "{} shader {:#x} does float atomic min/max with integer atomics",
                 info.hw_stage, info.pgm_hash);
    }
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.hw_stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.sw_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    // Looked at for every stage of every draw, and only copied to compile with, as compiling can
    // change it.
    const Shader::RuntimeInfo& built_runtime_info = BuildRuntimeInfo(hw_stage, sw_stage);
    // Draws mostly run the program the draw before did in the stage, found here without a lookup
    // in the map of them all, which missed the cache on most draws. Programs are never removed.
    auto& last_program = last_programs[static_cast<u32>(hw_stage)];
    if (last_program.second == nullptr || last_program.first != params.hash) {
        auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
        if (new_program) {
            auto runtime_info = built_runtime_info;
            it_pgm.value() = std::make_unique<Program>(hw_stage, sw_stage, params);
            auto& program = it_pgm.value();
            auto start = binding;
            const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
            auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
            const auto perm_hash = HashCombine(params.hash, 0);

            RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
            program->AddPermut(module, std::move(spec));
            if (auto& fetch = program->modules[0].spec.fetch_shader_data; !fetch.Empty()) {
                fetch_shader = &fetch;
            }
            last_program = {params.hash, program.get()};
            return std::make_tuple(&program->info, module, perm_hash);
        }
        last_program = {params.hash, it_pgm.value().get()};
    }

    Program* const program = last_program.second;
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();

    Common::Perf::Count(Common::Perf::Counter::ShaderLookups);
    if (!program->sharp_dwords_found) {
        program->FindSharpDwords();
    }
    const auto* lookup_dwords = program->LookupDwords();
    const u64 now = ++program->lookup_clock;
    const auto try_slot = [&](size_t slot) {
        const auto& last = program->last_lookups[slot];
        if (!last.valid || last.perm_idx >= program->modules.size()) {
            return false;
        }
        if (!last.Matches(info, built_runtime_info, binding,
                          program->modules[last.perm_idx].spec.fetch_shader_data, lookup_dwords)) {
            return false;
        }
        program->last_hit_slot = slot;
        program->slot_used[slot] = now;
        return true;
    };
    // Draws mostly look up what the draw before did, without hashing anything.
    size_t hit_slot = program->last_hit_slot;
    bool hit = try_slot(hit_slot);
    const size_t first_slot = hit ? 0 : program->LookupSlot(info, binding);
    const size_t second_slot = (first_slot + 1) % Program::NumLookupSlots;
    if (!hit) {
        for (const size_t slot : {first_slot, second_slot}) {
            if (slot != hit_slot && try_slot(slot)) {
                hit = true;
                hit_slot = slot;
                break;
            }
        }
    }
    if (hit) {
        const auto& last = program->last_lookups[hit_slot];
        auto& last_module = program->modules[last.perm_idx];
        Common::Perf::Count(Common::Perf::Counter::ShaderLookupsRemembered);
        info.AddBindings(binding);
        if (auto& fetch = last_module.spec.fetch_shader_data; !fetch.Empty()) {
            fetch_shader = &fetch;
        }
        return std::make_tuple(&program->info, last_module.module,
                               HashCombine(params.hash, last.perm_idx));
    }

    auto runtime_info = built_runtime_info;
    const auto start = binding;
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end()) {
        auto new_info = Shader::Info(hw_stage, sw_stage, params);
        module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }
    // The one of the two slots used less lately makes way.
    const size_t slot = program->slot_used[first_slot] <= program->slot_used[second_slot]
                            ? first_slot
                            : second_slot;
    program->last_lookups[slot].Remember(info, runtime_info, start, perm_idx,
                                         program->modules[perm_idx].spec.fetch_shader_data,
                                         lookup_dwords);
    program->slot_used[slot] = now;
    program->last_hit_slot = slot;
    return std::make_tuple(&program->info, module, perm_hash);
}

namespace {

/// The bits of each sharp dword a specialization is built from, see StageSpecialization.
struct SharpDwordUse {
    u32 mask;
    bool nonzero;
};

constexpr SharpDwordUse Exact{~0u, false};
constexpr SharpDwordUse Unused{0u, false};

/// V#: only the stride and swizzle bits of dword 1 (the rest is the top of the address),
/// whether num_records in dword 2 is set, and the formats, swizzles and type in dword 3.
constexpr std::array<SharpDwordUse, 4> BufferSharpUse = {
    Unused,
    SharpDwordUse{0xFFFF0000u, false},
    SharpDwordUse{0u, true},
    Exact,
};

/// T#: whether the address in dword 0 is set (with its top bits in dword 1), the formats in
/// dword 1, and the swizzles and type in dword 3, with the mip levels when a binding is made for
/// each of them. Min lod, tiling, sizes, pitch and layers aren't read.
constexpr SharpDwordUse ImageSharpUse(u32 dword, bool binds_levels) {
    switch (dword) {
    case 0:
        return SharpDwordUse{0u, true};
    case 1:
        return SharpDwordUse{0x3FF0003Fu, false};
    case 3:
        return SharpDwordUse{binds_levels ? 0xF00FFFFFu : 0xF0000FFFu, false};
    default:
        return Unused;
    }
}

/// S#: force_unnormalized and force_degamma, and whether it is set at all.
constexpr std::array<SharpDwordUse, 4> SamplerSharpUse = {
    SharpDwordUse{(1u << 15) | (1u << 20), true},
    SharpDwordUse{0u, true},
    SharpDwordUse{0u, true},
    SharpDwordUse{0u, true},
};

/// Vertex buffer sharps of a fetch shader: specializations only take the number class and
/// component swizzle of each attribute from them, out of the swizzle and formats in dword 3, and
/// whether it is set at all. Strides and the other bits vary between meshes drawn with the same
/// permutation, and comparing them made lookups of vertex shaders miss.
constexpr std::array<Program::SharpDword, 4> VertexSharpDwords = {
    Program::SharpDword{0, false, 0u},
    Program::SharpDword{1, false, 0u},
    Program::SharpDword{2, true, 0u},
    Program::SharpDword{3, false, 0x7FFFFu},
};

} // Anonymous namespace

void Program::FindSharpDwords() {
    // Specializations are built from the sharps of the resources found when the program was
    // first compiled, read where these say, and from nothing else in the flattened user data.
    sharp_dwords_found = true;
    const size_t num_dwords = info.flattened_ud_buf.size();
    std::vector<SharpDwordUse> uses(num_dwords, Unused);
    std::vector<bool> is_read(num_dwords);
    const auto add = [&](u32 dword, SharpDwordUse use) {
        if (dword < num_dwords) {
            is_read[dword] = true;
            uses[dword].mask |= use.mask;
            uses[dword].nonzero |= use.nonzero;
        } else {
            compare_all_dwords = true;
        }
    };
    const auto add_fetch = [&]<typename T>(const Shader::SharpFetch<T>& fetch, u32 count,
                                           auto&& use_of) {
        using Summary = typename Shader::SharpFetch<T>::Summary;
        if (fetch.summary == Summary::SingleLoad) {
            for (u32 i = 0; i < count; ++i) {
                add(fetch.offsets[0] + i, use_of(i));
            }
        } else if (fetch.summary == Summary::MultiLoad) {
            for (u32 i = 0; i < count; ++i) {
                if ((fetch.load_mask >> i) & 1) {
                    add(fetch.offsets[i], use_of(i));
                }
            }
        }
    };
    for (const auto& desc : info.buffers) {
        add_fetch(desc.sharp_fetch, Shader::SharpFetch<AmdGpu::Buffer>::N,
                  [](u32 i) { return BufferSharpUse[i]; });
    }
    for (const auto& desc : info.images) {
        // With a binding made for each mip level, their count is part of the specialization.
        const bool binds_levels =
            desc.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex;
        add_fetch(desc.sharp_fetch, desc.is_r128 ? 4 : Shader::SharpFetch<AmdGpu::Image>::N,
                  [binds_levels](u32 i) { return ImageSharpUse(i, binds_levels); });
    }
    for (const auto& desc : info.samplers) {
        // Post operations change bits of the sharp before it is checked to be set, so with
        // one all of it counts.
        const bool exact = desc.post_op != Shader::SharpFetchPostOp::None;
        add_fetch(desc.sharp_fetch, Shader::SharpFetch<AmdGpu::Sampler>::N,
                  [exact](u32 i) { return exact ? Exact : SamplerSharpUse[i]; });
        if (desc.post_op == Shader::SharpFetchPostOp::DisableAnisoIfSingleLod) {
            add(desc.post_op_tsharp_dw3_off, Exact);
        }
    }
    for (const auto& desc : info.fmasks) {
        for (u32 i = 0; i < sizeof(AmdGpu::Image) / sizeof(u32); ++i) {
            add(desc.sharp_idx + i, Exact);
        }
    }
    sharp_dwords.clear();
    for (u32 dword = 0; dword < num_dwords; ++dword) {
        const auto& use = uses[dword];
        if (is_read[dword] && (use.mask != 0 || use.nonzero)) {
            sharp_dwords.push_back(SharpDword{
                .index = static_cast<u16>(dword),
                .nonzero = use.nonzero,
                .mask = use.mask,
            });
        }
    }
}

size_t Program::LookupSlot(const Shader::Info& info, const Shader::Backend::Bindings& start) const {
    // From what LastLookup::Matches compares first, and the same for the inputs it finds the
    // same: only the bits of sharp dwords that count, and where bindings start only for programs
    // binding resources.
    u64 hash = 0;
    const auto mix = [&hash](u64 value) {
        hash = (hash ^ value) * 0x9E3779B97F4A7C15ULL;
        hash ^= hash >> 29;
    };
    if (!info.buffers.empty() || !info.images.empty() || !info.samplers.empty()) {
        mix(u64{start.unified} << 32 | start.buffer);
    }
    if (const auto* dwords = LookupDwords()) {
        for (const SharpDword& dword : *dwords) {
            const u32 value = info.flattened_ud_buf[dword.index];
            mix(u64{value & dword.mask} << 1 | u64{dword.nonzero && value != 0});
        }
    } else {
        for (const u32 value : info.flattened_ud_buf) {
            mix(value);
        }
    }
    return static_cast<size_t>(hash >> 32) % NumLookupSlots;
}

bool Program::LastLookup::Matches(const Shader::Info& info,
                                  const Shader::RuntimeInfo& runtime_info_,
                                  const Shader::Backend::Bindings& start_,
                                  const Shader::Gcn::FetchShaderData& fetch,
                                  const std::vector<SharpDword>* sharp_dwords) const {
    // These are everything a specialization is built from, so with all of them the same it
    // would come out the same and match the same permutation. Like specializations, programs
    // binding no resources don't compare where their bindings start, which depends on the
    // stages they are drawn with.
    //
    // Where the program is isn't compared. It only offsets the address of constant buffers
    // inlined in its code, which specializations don't look at. inFAMOUS Second Son has copies
    // of some programs at hundreds of addresses, and comparing it made three quarters of the
    // lookups that missed.
    const bool binds_resources =
        !info.buffers.empty() || !info.images.empty() || !info.samplers.empty();
    if (binds_resources && start != start_) {
        return false;
    }
    if (sharp_dwords) {
        for (size_t i = 0; i < sharp_dwords->size(); ++i) {
            const SharpDword& dword = (*sharp_dwords)[i];
            if (!dword.Same(info.flattened_ud_buf[dword.index], user_data[i])) {
                return false;
            }
        }
    } else if (!std::ranges::equal(user_data, info.flattened_ud_buf)) {
        return false;
    }
    if (!(runtime_info == runtime_info_)) {
        return false;
    }
    if (fetch.Empty()) {
        return true;
    }
    const u32* code = Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    if (std::memcmp(code, fetch_code.data(), fetch_code.size() * sizeof(u32)) != 0) {
        return false;
    }
    const u32* sharp = vertex_sharps.data();
    for (const auto& attrib : fetch.attributes) {
        const auto current =
            info.ReadUdReg<std::array<u32, 4>>(attrib.sgpr_base, attrib.dword_offset);
        for (const SharpDword& dword : VertexSharpDwords) {
            if (!dword.Same(current[dword.index], sharp[dword.index])) {
                return false;
            }
        }
        sharp += current.size();
    }
    return true;
}

void Program::LastLookup::Remember(const Shader::Info& info,
                                   const Shader::RuntimeInfo& runtime_info_,
                                   const Shader::Backend::Bindings& start_, size_t perm_idx_,
                                   const Shader::Gcn::FetchShaderData& fetch,
                                   const std::vector<SharpDword>* sharp_dwords) {
    // Tessellation stages are also specialized on constants read from a buffer in memory.
    valid = info.sw_stage != Shader::SwStage::TessellationControl &&
            info.sw_stage != Shader::SwStage::TessellationEval;
    if (!valid) {
        return;
    }
    perm_idx = perm_idx_;
    start = start_;
    runtime_info = runtime_info_;
    if (sharp_dwords) {
        user_data.clear();
        for (const SharpDword& dword : *sharp_dwords) {
            user_data.push_back(info.flattened_ud_buf[dword.index]);
        }
    } else {
        user_data.assign(info.flattened_ud_buf.begin(), info.flattened_ud_buf.end());
    }
    fetch_code.clear();
    vertex_sharps.clear();
    if (fetch.Empty()) {
        return;
    }
    const u32* code = Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    fetch_code.assign(code, code + fetch.size / sizeof(u32));
    for (const auto& attrib : fetch.attributes) {
        const auto sharp =
            info.ReadUdReg<std::array<u32, 4>>(attrib.sgpr_base, attrib.dword_offset);
        vertex_sharps.insert(vertex_sharps.end(), sharp.begin(), sharp.end());
    }
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        last_graphics_pipeline = nullptr;
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::HwStage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::HwStage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
