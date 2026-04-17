use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::OrbitCamera;
use crate::mesh::{cube_mesh, octahedron, uv_sphere, Vertex};
use crate::poscar::{element_color, element_radius, Crystal};
use crate::reciprocal::{GpuField, MAX_G};
use crate::symmetry::{self, CrystalSystem};
use crate::ui::EguiRenderer;

// ── GPU-mapped types ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SceneUniform {
    view_proj:  [[f32; 4]; 4],
    eye_pos:    [f32; 3],
    time:       f32,
    accent_col: [f32; 3],
    _pad:       f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AtomInstance {
    center_radius: [f32; 4], // xyz=centre, w=radius
    color:         [f32; 4], // rgb=colour, a=ghost opacity (1=solid)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PostParams {
    width:  f32,
    height: f32,
    _pad:   [f32; 2],
}

// ── Crystal-field pipeline types ─────────────────────────────────────────

/// Uniform block for the crystal-field fragment shader.
/// Layout must match the WGSL `FU` struct exactly.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FieldUniform {
    pub time:          f32,   //  0
    pub kscale:        f32,   //  4
    pub speed:         f32,   //  8
    pub field_mix:     f32,   // 12
    pub iso_level:     f32,   // 16
    pub color_shift:   f32,   // 20
    pub zoom:          f32,   // 24
    pub w_lattice:     f32,   // 28
    pub w_motif:       f32,   // 32
    pub w_band:        f32,   // 36
    pub mode:          u32,   // 40
    pub num_g:         u32,   // 44
    pub crystal_color: [f32; 4], // 48  (xyz=colour, w=unused)
    pub mouse:         [f32; 2], // 64
    pub mouse_down:    f32,   // 72
    pub aspect:        f32,   // 76
}                             // total 80 bytes

pub struct FieldPipeline {
    pub uniform_buf: wgpu::Buffer,
    uniform_bg:      wgpu::BindGroup,
    g_texture:       wgpu::Texture,
    field_pl:        wgpu::RenderPipeline,
}

impl FieldPipeline {
    pub fn new(device: &wgpu::Device, surface_fmt: wgpu::TextureFormat) -> Self {
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field uniform"),
            size:  std::mem::size_of::<FieldUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // G-vector texture: MAX_G columns × 2 rows, Rgba32Float
        let g_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("g_tex"),
            size:  wgpu::Extent3d { width: MAX_G as u32, height: 2, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let g_tex_view = g_texture.create_view(&Default::default());

        let bgl = field_bgl(device);
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("field bg"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&g_tex_view) },
            ],
        });

        let field_pl = build_field_pl(device, &bgl, surface_fmt);
        Self { uniform_buf, uniform_bg, g_texture, field_pl }
    }

    /// Upload packed G-vector data (from `GpuField::pack()`).
    pub fn upload_gfield(&self, queue: &wgpu::Queue, data: &[f32]) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture:   &self.g_texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset:         0,
                bytes_per_row:  Some((MAX_G * 4 * 4) as u32), // MAX_G * RGBA * f32
                rows_per_image: Some(2),
            },
            wgpu::Extent3d { width: MAX_G as u32, height: 2, depth_or_array_layers: 1 },
        );
    }
}

// ── Shape assignment ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeKind {
    Sphere,
    Octahedron,
    Cube,
}

fn shape_for(sym: &str) -> ShapeKind {
    let s: String = sym.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
    match s.as_str() {
        "F" | "Cl" | "Br" | "I" | "At"        => ShapeKind::Octahedron,
        "O" | "S"  | "Se" | "Te"              => ShapeKind::Cube,
        _                                      => ShapeKind::Sphere,
    }
}

// ── Per-shape draw group ──────────────────────────────────────────────────

struct ShapeGroup {
    vbuf:  wgpu::Buffer,
    ibuf:  wgpu::Buffer,
    icnt:  u32,
    ibuf2: wgpu::Buffer, // instance buffer
    inst_cnt: u32,
}

impl ShapeGroup {
    fn new(device: &wgpu::Device, verts: &[Vertex], idx: &[u32], insts: &[AtomInstance]) -> Self {
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let ibuf2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(insts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self { vbuf, ibuf, icnt: idx.len() as u32, ibuf2, inst_cnt: insts.len() as u32 }
    }
    fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.inst_cnt == 0 { return; }
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_vertex_buffer(1, self.ibuf2.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.icnt, 0, 0..self.inst_cnt);
    }
}

// ── Post-processing (bloom) ───────────────────────────────────────────────

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct PostProcess {
    scene_view:  wgpu::TextureView,
    bright_view: wgpu::TextureView,
    blur_a_view: wgpu::TextureView,

    sampler:      wgpu::Sampler,
    params_buf:   wgpu::Buffer,   // PostParams: width + height

    bright_bg:    wgpu::BindGroup,
    blur_h_bg:    wgpu::BindGroup,
    blur_v_bg:    wgpu::BindGroup,
    composite_bg: wgpu::BindGroup,

    bright_pl:    wgpu::RenderPipeline,
    blur_h_pl:    wgpu::RenderPipeline,
    blur_v_pl:    wgpu::RenderPipeline,
    composite_pl: wgpu::RenderPipeline,
}

impl PostProcess {
    fn new(
        device: &wgpu::Device,
        w: u32, h: u32,
        surface_fmt: wgpu::TextureFormat,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("post params"),
            contents: bytemuck::bytes_of(&PostParams { width: w as f32, height: h as f32, _pad: [0.0; 2] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (scene_view, bright_view, blur_a_view) = Self::make_textures(device, w, h);

        // ── bind group layouts ─────────────────────────────────────────
        let bgl_1 = bgl_single_tex(device);    // bright-pass: 1 HDR tex + sampler
        let bgl_blur = bgl_blur(device);        // blur: 1 tex + sampler + params
        let bgl_comp = bgl_composite(device);   // composite: 2 HDR tex + sampler

        // ── bind groups ────────────────────────────────────────────────
        let bright_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bright bg"), layout: &bgl_1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scene_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_h bg"), layout: &bgl_blur,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bright_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });
        let blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_v bg"), layout: &bgl_blur,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blur_a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bg"), layout: &bgl_comp,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scene_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&blur_a_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        // ── pipelines ──────────────────────────────────────────────────
        let bright_pl    = fullscreen_pl(device, &bgl_1,    BRIGHT_SHADER,    "fs_bright",    HDR_FORMAT);
        let blur_h_pl    = fullscreen_pl(device, &bgl_blur,  BLUR_H_SHADER,   "fs_blur_h",    HDR_FORMAT);
        let blur_v_pl    = fullscreen_pl(device, &bgl_blur,  BLUR_V_SHADER,   "fs_blur_v",    HDR_FORMAT);
        let composite_pl = fullscreen_pl(device, &bgl_comp,  COMPOSITE_SHADER,"fs_composite", surface_fmt);

        Self {
            scene_view, bright_view, blur_a_view,
            sampler, params_buf,
            bright_bg, blur_h_bg, blur_v_bg, composite_bg,
            bright_pl, blur_h_pl, blur_v_pl, composite_pl,
        }
    }

    fn make_textures(device: &wgpu::Device, w: u32, h: u32)
        -> (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView)
    {
        let make = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }).create_view(&wgpu::TextureViewDescriptor::default())
        };
        (make("scene"), make("bright"), make("blur_a"))
    }

    fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, surface_fmt: wgpu::TextureFormat) {
        // Recreate textures and all bind groups that reference them
        let (scene_view, bright_view, blur_a_view) = Self::make_textures(device, w, h);

        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(
            &PostParams { width: w as f32, height: h as f32, _pad: [0.0; 2] }
        ));

        let bgl_1    = bgl_single_tex(device);
        let bgl_blur = bgl_blur(device);
        let bgl_comp = bgl_composite(device);

        self.bright_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bright bg"), layout: &bgl_1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scene_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_h bg"), layout: &bgl_blur,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bright_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.params_buf.as_entire_binding() },
            ],
        });
        self.blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_v bg"), layout: &bgl_blur,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blur_a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.params_buf.as_entire_binding() },
            ],
        });
        self.composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bg"), layout: &bgl_comp,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&scene_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&blur_a_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

        // Rebuild composite pipeline if needed (same layout, so no — but keep for future)
        let _ = surface_fmt; // may differ post-resize; pass in if fmt can change

        self.scene_view  = scene_view;
        self.bright_view = bright_view;
        self.blur_a_view = blur_a_view;
    }
}

// ── Main GPU state ────────────────────────────────────────────────────────

pub struct GpuState {
    pub window:   Arc<Window>,
    surface:      wgpu::Surface<'static>,
    device:       wgpu::Device,
    pub queue:    wgpu::Queue,
    config:       wgpu::SurfaceConfiguration,
    pub size:     winit::dpi::PhysicalSize<u32>,

    camera:       OrbitCamera,
    scene_buf:    wgpu::Buffer,
    scene_bg:     wgpu::BindGroup,

    atom_pl:      wgpu::RenderPipeline,
    cell_pl:      wgpu::RenderPipeline,
    depth_view:   wgpu::TextureView,

    shapes:       Vec<ShapeGroup>,
    cell_vbuf:    wgpu::Buffer,
    cell_vcnt:    u32,

    post:         PostProcess,
    surface_fmt:  wgpu::TextureFormat,

    pub supercell:   usize,
    crystal:         Crystal,
    crystal_sys:     CrystalSystem,

    // ── crystal-field mode ────────────────────────────────────────────
    pub field_pl:    FieldPipeline,
    pub gpu_field:   GpuField,
    pub kpath:       Option<crate::kpoints::KpathWalker>,

    // ── egui overlay ──────────────────────────────────────────────────
    pub egui:        EguiRenderer,
}

impl GpuState {
    pub async fn new(window: Arc<Window>, crystal: Crystal) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.expect("no GPU adapter");

        log::info!("GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("crystal-viz"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
        }, None).await.expect("device creation failed");

        let caps = surface.get_capabilities(&adapter);
        let surface_fmt = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_fmt,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let crystal_sys = symmetry::detect(&crystal.lattice);
        log::info!("Crystal system: {}", crystal_sys.name());

        // ── scene uniform ─────────────────────────────────────────────
        let cc = crystal.cell_center();
        let target = glam::Vec3::from(cc);
        let max_a = crystal.lattice.iter()
            .map(|r| (r[0]*r[0]+r[1]*r[1]+r[2]*r[2]).sqrt())
            .fold(0f32, f32::max);
        let camera = OrbitCamera::new(target, max_a * 2.8);

        let scene_bgl = scene_bgl(&device);
        let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene uniform"),
            contents: bytemuck::bytes_of(&make_scene_uniform(&camera, aspect(&config), 0.0, crystal_sys)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bg"), layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0, resource: scene_buf.as_entire_binding(),
            }],
        });

        // ── pipelines ─────────────────────────────────────────────────
        let atom_pl = build_atom_pl(&device, &scene_bgl, HDR_FORMAT);
        let cell_pl = build_cell_pl(&device, &scene_bgl, HDR_FORMAT);

        // ── depth ─────────────────────────────────────────────────────
        let depth_view = make_depth(&device, &config);

        // ── scene geometry ────────────────────────────────────────────
        let shapes  = build_shape_groups(&device, &crystal, 1);
        let cell_verts = cell_lines(&crystal.lattice);
        let cell_vcnt  = cell_verts.len() as u32 / 3;
        let cell_vbuf  = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cell vbuf"), contents: bytemuck::cast_slice(&cell_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // ── post-process ──────────────────────────────────────────────
        let post = PostProcess::new(&device, size.width.max(1), size.height.max(1), surface_fmt);

        // ── crystal-field pipeline ────────────────────────────────────
        let field_pl = FieldPipeline::new(&device, surface_fmt);
        let mut gpu_field = GpuField::from_crystal(&crystal, 3);
        gpu_field.seed_kpoint([0.0, 0.0, 0.0], 1.0); // start at Γ
        field_pl.upload_gfield(&queue, &gpu_field.pack());

        // ── egui ──────────────────────────────────────────────────────
        let egui = EguiRenderer::new(&device, surface_fmt, &window);

        Self {
            window, surface, device, queue, config, size,
            camera, scene_buf, scene_bg,
            atom_pl, cell_pl, depth_view,
            shapes, cell_vbuf, cell_vcnt,
            post, surface_fmt,
            supercell: 1,
            crystal, crystal_sys,
            field_pl, gpu_field,
            kpath: None, // set by caller after construction
            egui,
        }
    }

    // ── public controls ───────────────────────────────────────────────

    /// Feed a winit window event to egui; returns true if egui consumed it.
    pub fn handle_ui_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.egui.handle_event(&self.window, event)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) { self.camera.orbit(dx, dy); }
    pub fn zoom(&mut self, d: f32)             { self.camera.zoom(d); }

    pub fn set_supercell(&mut self, n: usize) {
        if n == self.supercell { return; }
        self.supercell = n;
        self.shapes = build_shape_groups(&self.device, &self.crystal, n);
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.size = new_size;
        self.config.width  = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = make_depth(&self.device, &self.config);
        self.post.resize(&self.device, &self.queue, new_size.width, new_size.height, self.surface_fmt);
    }

    pub fn render(&mut self, time: f32, ui_fn: impl FnMut(&egui::Context)) -> Result<(), wgpu::SurfaceError> {
        // Update scene uniform
        let u = make_scene_uniform(&self.camera, aspect(&self.config), time, self.crystal_sys);
        self.queue.write_buffer(&self.scene_buf, 0, bytemuck::bytes_of(&u));

        let output = self.surface.get_current_texture()?;
        let screen_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // ── Pass 1: scene → HDR texture ───────────────────────────────
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.02, b: 0.10, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            // Cell wireframe
            pass.set_pipeline(&self.cell_pl);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            pass.set_vertex_buffer(0, self.cell_vbuf.slice(..));
            pass.draw(0..self.cell_vcnt, 0..1);
            // Atoms
            pass.set_pipeline(&self.atom_pl);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            for sg in &self.shapes { sg.draw(&mut pass); }
        }

        // ── Pass 2: bright filter ─────────────────────────────────────
        fullscreen_pass(&mut enc, &self.post.bright_view, &self.post.bright_pl, &self.post.bright_bg, false);

        // ── Pass 3: blur H ────────────────────────────────────────────
        fullscreen_pass(&mut enc, &self.post.blur_a_view, &self.post.blur_h_pl, &self.post.blur_h_bg, false);

        // ── Pass 4: blur V (blur_a → bright reused as output) ─────────
        // Re-use bright_view as the final blur result (we no longer need raw bright)
        fullscreen_pass(&mut enc, &self.post.bright_view, &self.post.blur_v_pl, &self.post.blur_v_bg, false);

        // ── Pass 5: composite to screen ───────────────────────────────
        // composite_bg: scene + bright(=final blur) + sampler
        fullscreen_pass(&mut enc, &screen_view, &self.post.composite_pl, &self.post.composite_bg, true);

        // ── Pass 6: egui overlay ──────────────────────────────────────
        let ppp = self.window.scale_factor() as f32;
        let window = Arc::clone(&self.window);
        self.egui.render(&self.device, &self.queue, &mut enc, &screen_view, &window, ppp, ui_fn);

        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }

    pub fn crystal_system(&self) -> CrystalSystem { self.crystal_sys }

    /// Render the field to an offscreen Rgba8 texture and return the PNG bytes.
    /// Call this instead of render_field when a screenshot is needed.
    pub fn screenshot_field(&mut self, params: &FieldUniform) -> Vec<u8> {
        let w = self.size.width.max(1);
        let h = self.size.height.max(1);
        let aligned_bytes_per_row = ((w * 4) + 255) & !255; // align to 256

        // Offscreen render target
        let capture_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let capture_view = capture_tex.create_view(&Default::default());

        // Readback buffer
        let buf_size = (aligned_bytes_per_row * h) as u64;
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Need a sRGB-compatible field pipeline — the existing one might be surface_fmt.
        // Use the existing pipeline; it writes to whatever target we give it.
        self.queue.write_buffer(&self.field_pl.uniform_buf, 0, bytemuck::bytes_of(params));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screenshot field"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &capture_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.field_pl.field_pl);
            pass.set_bind_group(0, &self.field_pl.uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &capture_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);

        // Synchronous readback
        let slice = readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let raw = slice.get_mapped_range();
        // De-stripe: remove row padding
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h as usize {
            let start = row * aligned_bytes_per_row as usize;
            pixels.extend_from_slice(&raw[start..start + (w * 4) as usize]);
        }
        drop(raw);
        readback_buf.unmap();

        // Encode as PNG via DynamicImage
        let img = image::RgbaImage::from_raw(w, h, pixels)
            .expect("screenshot buffer size mismatch");
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .unwrap_or_default();
        png_bytes
    }

    /// Re-upload current phase state to the G-texture.
    pub fn update_field(&self) {
        self.field_pl.upload_gfield(&self.queue, &self.gpu_field.pack());
    }

    /// Render the crystal field (bypasses atom/bloom pipeline — writes straight to swapchain).
    pub fn render_field(&mut self, params: &FieldUniform, ui_fn: impl FnMut(&egui::Context)) -> Result<(), wgpu::SurfaceError> {
        self.queue.write_buffer(&self.field_pl.uniform_buf, 0, bytemuck::bytes_of(params));

        let output = self.surface.get_current_texture()?;
        let view   = output.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("field pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.field_pl.field_pl);
            pass.set_bind_group(0, &self.field_pl.uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── egui overlay ──────────────────────────────────────────────
        let ppp = self.window.scale_factor() as f32;
        let window = Arc::clone(&self.window);
        self.egui.render(&self.device, &self.queue, &mut enc, &view, &window, ppp, ui_fn);

        self.queue.submit([enc.finish()]);
        output.present();
        Ok(())
    }
}

// ── Scene geometry helpers ────────────────────────────────────────────────

fn build_shape_groups(device: &wgpu::Device, crystal: &Crystal, supercell: usize) -> Vec<ShapeGroup> {
    let n = supercell as i32;
    let lat = &crystal.lattice;

    // Collect instances per shape kind
    let mut sphere_inst = Vec::<AtomInstance>::new();
    let mut octa_inst   = Vec::<AtomInstance>::new();
    let mut cube_inst   = Vec::<AtomInstance>::new();

    for na in 0..n {
        for nb in 0..n {
            for nc in 0..n {
                let off = [
                    na as f32 * lat[0][0] + nb as f32 * lat[1][0] + nc as f32 * lat[2][0],
                    na as f32 * lat[0][1] + nb as f32 * lat[1][1] + nc as f32 * lat[2][1],
                    na as f32 * lat[0][2] + nb as f32 * lat[1][2] + nc as f32 * lat[2][2],
                ];
                let alpha = if na == 0 && nb == 0 && nc == 0 { 1.0f32 } else { 0.30 };

                for atom in &crystal.atoms {
                    let col = element_color(&atom.species);
                    let rad = element_radius(&atom.species);
                    let inst = AtomInstance {
                        center_radius: [
                            atom.pos_cart[0] + off[0],
                            atom.pos_cart[1] + off[1],
                            atom.pos_cart[2] + off[2],
                            rad,
                        ],
                        color: [col[0], col[1], col[2], alpha],
                    };
                    match shape_for(&atom.species) {
                        ShapeKind::Sphere     => sphere_inst.push(inst),
                        ShapeKind::Octahedron => octa_inst.push(inst),
                        ShapeKind::Cube       => cube_inst.push(inst),
                    }
                }
            }
        }
    }

    let (sv, si) = uv_sphere(18, 18);
    let (ov, oi) = octahedron();
    let (cv, ci) = cube_mesh();

    vec![
        ShapeGroup::new(device, &sv, &si, &sphere_inst),
        ShapeGroup::new(device, &ov, &oi, &octa_inst),
        ShapeGroup::new(device, &cv, &ci, &cube_inst),
    ]
}

fn cell_lines(lat: &[[f32; 3]; 3]) -> Vec<f32> {
    let zero = [0f32; 3];
    let a = lat[0]; let b = lat[1]; let c = lat[2];
    let ab = add3(a, b); let ac = add3(a, c); let bc = add3(b, c);
    let abc = add3(ab, c);
    let corners = [zero, a, b, c, ab, ac, bc, abc];
    let edges: [(usize, usize); 12] = [
        (0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7),
    ];
    let mut v = Vec::with_capacity(12 * 6);
    for (i, j) in edges { v.extend_from_slice(&corners[i]); v.extend_from_slice(&corners[j]); }
    v
}

fn add3(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }

fn make_scene_uniform(cam: &OrbitCamera, asp: f32, t: f32, sys: CrystalSystem) -> SceneUniform {
    let vp  = cam.view_proj(asp);
    let eye = cam.eye();
    let ac  = sys.accent();
    SceneUniform {
        view_proj:  vp.to_cols_array_2d(),
        eye_pos:    eye.into(),
        time:       t,
        accent_col: ac,
        _pad:       0.0,
    }
}

fn aspect(c: &wgpu::SurfaceConfiguration) -> f32 { c.width as f32 / c.height as f32 }

fn make_depth(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: config.width.max(1), height: config.height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}

// ── Bind group layout helpers ─────────────────────────────────────────────

fn scene_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn tex_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn sampler_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}
fn uniform_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false, min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_single_tex(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_1tex"),
        entries: &[tex_binding(0), sampler_binding(1)],
    })
}
fn bgl_blur(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_blur"),
        entries: &[tex_binding(0), sampler_binding(1), uniform_binding(2)],
    })
}
fn field_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("field bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type:   wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled:   false,
                },
                count: None,
            },
        ],
    })
}

fn bgl_composite(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_comp"),
        entries: &[tex_binding(0), tex_binding(1), sampler_binding(2)],
    })
}

// ── Pipeline builders ─────────────────────────────────────────────────────

fn ds_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn build_atom_pl(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, fmt: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("atom"), source: wgpu::ShaderSource::Wgsl(ATOM_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    let vbuf = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
        ],
    };
    let ibuf = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<AtomInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0,  shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 3 },
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("atom pl"), layout: Some(&layout),
        vertex: wgpu::VertexState { module: &sm, entry_point: "vs_atom", buffers: &[vbuf, ibuf], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &sm, entry_point: "fs_atom",
            targets: &[Some(wgpu::ColorTargetState {
                format: fmt,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: Some(wgpu::Face::Back), ..Default::default() },
        depth_stencil: Some(ds_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None, cache: None,
    })
}

fn build_cell_pl(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, fmt: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cell"), source: wgpu::ShaderSource::Wgsl(CELL_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cell pl"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm, entry_point: "vs_cell",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm, entry_point: "fs_cell",
            targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
        depth_stencil: Some(ds_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None, cache: None,
    })
}

fn build_field_pl(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, fmt: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("field"), source: wgpu::ShaderSource::Wgsl(FIELD_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("field pl"), layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &sm, entry_point: "vs_screen", buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &sm, entry_point: "fs_field",
            targets: &[Some(wgpu::ColorTargetState {
                format: fmt, blend: None, write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None, cache: None,
    })
}

fn fullscreen_pl(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, src: &str, entry: &str, fmt: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry), source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[bgl], push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry), layout: Some(&layout),
        vertex: wgpu::VertexState { module: &sm, entry_point: "vs_screen", buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &sm, entry_point: entry,
            targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None, cache: None,
    })
}

fn fullscreen_pass(
    enc: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pl: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    clear: bool,
) {
    let load = if clear {
        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
    } else {
        wgpu::LoadOp::Load
    };
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target, resolve_target: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None, timestamp_writes: None,
    });
    pass.set_pipeline(pl);
    pass.set_bind_group(0, bg, &[]);
    pass.draw(0..3, 0..1);
}

// ── WGSL shaders ──────────────────────────────────────────────────────────

const ATOM_SHADER: &str = r#"
struct Scene {
    view_proj:  mat4x4<f32>,
    eye_pos:    vec3<f32>,
    time:       f32,
    accent_col: vec3<f32>,
}
@group(0) @binding(0) var<uniform> sc: Scene;

struct VIn {
    @location(0) mesh_pos:      vec3<f32>,
    @location(1) mesh_normal:   vec3<f32>,
    @location(2) center_radius: vec4<f32>,
    @location(3) color:         vec4<f32>,  // rgb + alpha
}
struct VOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0) normal:      vec3<f32>,
    @location(1) color:       vec3<f32>,
    @location(2) world_pos:   vec3<f32>,
    @location(3) alpha:       f32,
}

@vertex fn vs_atom(v: VIn) -> VOut {
    let world = v.center_radius.xyz + v.mesh_pos * v.center_radius.w;
    var o: VOut;
    o.clip      = sc.view_proj * vec4<f32>(world, 1.0);
    o.normal    = v.mesh_normal;
    o.color     = v.color.rgb;
    o.world_pos = world;
    o.alpha     = v.color.a;
    return o;
}

@fragment fn fs_atom(f: VOut) -> @location(0) vec4<f32> {
    let light   = normalize(vec3<f32>(2.0, 3.5, 1.5));
    let n       = normalize(f.normal);
    let view    = normalize(sc.eye_pos - f.world_pos);

    let diff    = max(dot(n, light), 0.0);
    let h       = normalize(light + view);
    let spec    = pow(max(dot(n, h), 0.0), 48.0);

    // Fresnel rim glow — colour of the crystal-system accent
    let rim     = 1.0 - max(dot(n, view), 0.0);
    let fresnel = pow(rim, 2.5);
    let glow    = sc.accent_col * fresnel * 1.4;

    // Subtle per-atom breathing pulse
    let phase  = dot(f.world_pos, vec3<f32>(0.3, 0.5, 0.2));
    let pulse  = sin(sc.time * 1.8 + phase) * 0.5 + 0.5;

    let ambient = 0.12;
    let lit     = f.color * (ambient + diff * 0.70) + vec3(1.0) * spec * 0.4;
    let emissive = (f.color * 0.15 + glow) * (0.7 + pulse * 0.3);

    return vec4<f32>(lit + emissive, f.alpha);
}
"#;

const CELL_SHADER: &str = r#"
struct Scene {
    view_proj:  mat4x4<f32>,
    eye_pos:    vec3<f32>,
    time:       f32,
    accent_col: vec3<f32>,
}
@group(0) @binding(0) var<uniform> sc: Scene;

struct VOut { @builtin(position) clip: vec4<f32> }

@vertex fn vs_cell(@location(0) pos: vec3<f32>) -> VOut {
    var o: VOut;
    o.clip = sc.view_proj * vec4<f32>(pos, 1.0);
    return o;
}

@fragment fn fs_cell(f: VOut) -> @location(0) vec4<f32> {
    // Smoothly hue-shift between accent colour and its complement over time
    let t   = sin(sc.time * 0.4) * 0.5 + 0.5;
    let col = mix(sc.accent_col, vec3<f32>(1.0) - sc.accent_col * 0.6, t);
    return vec4<f32>(col * 1.5, 0.90);  // over-bright so bloom picks it up
}
"#;

// ── Full-screen pass shaders ──────────────────────────────────────────────
// Rust's concat! only accepts literals, so the shared vertex shader is
// inlined into each string. The VS is identical in all four.

const BRIGHT_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uv = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let u = uv[vi];
    var o: VOut;
    o.pos = vec4<f32>(u.x*2.0-1.0, 1.0-u.y*2.0, 0.0, 1.0);
    o.uv  = u;
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@fragment fn fs_bright(f: VOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, smp, f.uv);
    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let knee = smoothstep(0.65, 1.0, lum);
    return vec4<f32>(c.rgb * knee, 1.0);
}
"#;

const BLUR_H_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uv = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let u = uv[vi];
    var o: VOut;
    o.pos = vec4<f32>(u.x*2.0-1.0, 1.0-u.y*2.0, 0.0, 1.0);
    o.uv  = u;
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct PP { width: f32, height: f32 }
@group(0) @binding(2) var<uniform> pp: PP;
@fragment fn fs_blur_h(f: VOut) -> @location(0) vec4<f32> {
    let ts = vec2<f32>(1.0/pp.width, 0.0);
    var c = textureSample(tex,smp,f.uv).rgb * 0.227027;
    c += textureSample(tex,smp,f.uv+ts*1.0).rgb*0.1945946; c += textureSample(tex,smp,f.uv-ts*1.0).rgb*0.1945946;
    c += textureSample(tex,smp,f.uv+ts*2.0).rgb*0.1216216; c += textureSample(tex,smp,f.uv-ts*2.0).rgb*0.1216216;
    c += textureSample(tex,smp,f.uv+ts*3.0).rgb*0.054054;  c += textureSample(tex,smp,f.uv-ts*3.0).rgb*0.054054;
    c += textureSample(tex,smp,f.uv+ts*4.0).rgb*0.016216;  c += textureSample(tex,smp,f.uv-ts*4.0).rgb*0.016216;
    return vec4<f32>(c, 1.0);
}
"#;

const BLUR_V_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uv = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let u = uv[vi];
    var o: VOut;
    o.pos = vec4<f32>(u.x*2.0-1.0, 1.0-u.y*2.0, 0.0, 1.0);
    o.uv  = u;
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct PP { width: f32, height: f32 }
@group(0) @binding(2) var<uniform> pp: PP;
@fragment fn fs_blur_v(f: VOut) -> @location(0) vec4<f32> {
    let ts = vec2<f32>(0.0, 1.0/pp.height);
    var c = textureSample(tex,smp,f.uv).rgb * 0.227027;
    c += textureSample(tex,smp,f.uv+ts*1.0).rgb*0.1945946; c += textureSample(tex,smp,f.uv-ts*1.0).rgb*0.1945946;
    c += textureSample(tex,smp,f.uv+ts*2.0).rgb*0.1216216; c += textureSample(tex,smp,f.uv-ts*2.0).rgb*0.1216216;
    c += textureSample(tex,smp,f.uv+ts*3.0).rgb*0.054054;  c += textureSample(tex,smp,f.uv-ts*3.0).rgb*0.054054;
    c += textureSample(tex,smp,f.uv+ts*4.0).rgb*0.016216;  c += textureSample(tex,smp,f.uv-ts*4.0).rgb*0.016216;
    return vec4<f32>(c, 1.0);
}
"#;

const COMPOSITE_SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uv = array<vec2<f32>,3>(vec2(0.0,0.0),vec2(2.0,0.0),vec2(0.0,2.0));
    let u = uv[vi];
    var o: VOut;
    o.pos = vec4<f32>(u.x*2.0-1.0, 1.0-u.y*2.0, 0.0, 1.0);
    o.uv  = u;
    return o;
}
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var smp:   sampler;
@fragment fn fs_composite(f: VOut) -> @location(0) vec4<f32> {
    let s = textureSample(scene, smp, f.uv).rgb;
    let b = textureSample(bloom, smp, f.uv).rgb;
    let hdr = s + b * 1.8;
    let mapped = hdr / (hdr + vec3<f32>(1.0));
    let gamma  = pow(mapped, vec3<f32>(1.0/2.2));
    return vec4<f32>(gamma, 1.0);
}
"#;

// ── Crystal-field shader ──────────────────────────────────────────────────
// Five render modes: 0=ray-march isosurface  1=BZ slice  2=Fermi surface
//                    3=charge density        4=nodal lines
const FIELD_SHADER: &str = r#"
struct FU {
    time:          f32,
    kscale:        f32,
    speed:         f32,
    field_mix:     f32,
    iso_level:     f32,
    color_shift:   f32,
    zoom:          f32,
    w_lattice:     f32,
    w_motif:       f32,
    w_band:        f32,
    mode:          u32,
    num_g:         u32,
    crystal_color: vec4<f32>,
    mouse:         vec2<f32>,
    mouse_down:    f32,
    aspect:        f32,
}
@group(0) @binding(0) var<uniform> u: FU;
@group(0) @binding(1) var g_tex: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex fn vs_screen(@builtin(vertex_index) vi: u32) -> VOut {
    var uvs = array<vec2<f32>,3>(vec2(0.0,0.0), vec2(2.0,0.0), vec2(0.0,2.0));
    let uv = uvs[vi];
    var o: VOut;
    o.pos = vec4<f32>(uv.x*2.0-1.0, 1.0-uv.y*2.0, 0.0, 1.0);
    o.uv  = uv;
    return o;
}

const TAU: f32 = 6.28318530718;

fn crystal_field(x: vec3<f32>) -> f32 {
    var v = 0.0; var ns = 0.0;
    let t = u.time * u.speed;
    for (var i = 0i; i < 64i; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let ph  = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        let G   = ga.xyz * u.kscale;
        let amp = ga.w;
        let gx  = dot(G, x);
        let lat   = cos(gx + ph + t*(1.0 + f32(i)*0.01));
        let motif = cos(gx*1.13 + ph*1.7 + t*0.7);
        let band  = cos(gx + dot(G,G)*0.12 + t*0.4);
        v  += amp * (u.w_lattice*lat + u.w_motif*motif + u.w_band*band);
        ns += amp;
    }
    return v / max(ns, 0.001);
}

fn cf2(x: vec3<f32>) -> f32 {
    return crystal_field(x*1.37 + vec3<f32>(1.618, 2.718, 3.141));
}

fn sdf(p: vec3<f32>) -> f32 {
    return abs(mix(crystal_field(p), cf2(p), u.field_mix)) - u.iso_level*0.3;
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.004;
    return normalize(vec3<f32>(
        sdf(p+vec3(e,0,0))-sdf(p-vec3(e,0,0)),
        sdf(p+vec3(0,e,0))-sdf(p-vec3(0,e,0)),
        sdf(p+vec3(0,0,e))-sdf(p-vec3(0,0,e)),
    ));
}

fn cfield_col(f: f32, f2: f32, n: vec3<f32>) -> vec3<f32> {
    let hue = fract(f*2.0 + f2*0.7 + u.color_shift + u.time*u.speed*0.1);
    var col  = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
    col = mix(col, u.crystal_color.xyz, 0.3);
    col += pow(1.0 - abs(n.z), 3.0) * 0.4 * vec3<f32>(0.6, 0.4, 1.0);
    col = mix(col, col*0.4 + vec3<f32>(0.6,0.8,1.0)*(0.5+0.5*sin(f*20.0+f2*15.0))*0.6, 0.25);
    return col;
}

// ── Mode 0: ray-march isosurface ──────────────────────────────────────────
fn render_rm(uv: vec2<f32>) -> vec3<f32> {
    var az = u.time*u.speed*0.15; var el = 0.4;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.513;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.5/u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0,1,0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);
    var t   = 0.0;
    var col = vec3<f32>(0.01, 0.005, 0.02);
    for (var i = 0; i < 80; i++) {
        let p = cp + rd*t;
        let d = sdf(p);
        if d < 0.003 {
            let nm = calc_normal(p);
            col = cfield_col(crystal_field(p), cf2(p), nm) * (0.5 + 0.5*(1.0-f32(i)/80.0));
            break;
        }
        if t > 8.0 { break; }
        t += max(d, 0.005);
    }
    col += u.crystal_color.xyz * max(crystal_field(cp + rd*4.0)*0.3, 0.0) * 0.08;
    return col;
}

// ── Mode 1: BZ slice ──────────────────────────────────────────────────────
fn render_bz(uv: vec2<f32>) -> vec3<f32> {
    let p  = vec3<f32>(uv*3.0, u.time*u.speed*0.3);
    let f  = crystal_field(p);
    let f2 = cf2(p);
    var c  = mix(vec3<f32>(0.02,0.01,0.04), u.crystal_color.xyz*0.6, smoothstep(-0.8,0.8,f));
    c += smoothstep(0.02, 0.0, abs(f))  * vec3<f32>(1.0,0.9,0.5) * 1.2;
    c += smoothstep(0.03, 0.0, abs(f - u.iso_level*0.5)) * u.crystal_color.xyz * 1.5;
    c += (0.5+0.5*sin(f*TAU*3.0)*cos(f2*TAU*2.0)) * 0.08 * u.crystal_color.xyz;
    return c;
}

// ── Mode 2: Fermi surface (scanning slice) ────────────────────────────────
fn render_fermi(uv: vec2<f32>) -> vec3<f32> {
    let z  = sin(u.time*u.speed*0.5)*0.8;
    let p  = vec3<f32>(uv*2.5/u.zoom, z);
    let f  = crystal_field(p); let f2 = cf2(p);
    var c  = vec3<f32>(0.01,0.005,0.025);
    c += smoothstep(0.025,0.0, abs(f  - u.iso_level*0.6)) * u.crystal_color.xyz * 2.0;
    c += smoothstep(0.02, 0.0, abs(f2 + u.iso_level*0.4)) * vec3<f32>(1.0,0.8,0.3) * 1.5;
    c += (0.5+0.5*sin(f*f2*30.0 + u.time*u.speed)) * u.crystal_color.xyz * 0.06;
    c += smoothstep(-1.0,1.0,f) * u.crystal_color.xyz * 0.3;
    return c;
}

// ── Mode 3: charge density ρ ∝ |ψ|² ─────────────────────────────────────
fn render_density(uv: vec2<f32>) -> vec3<f32> {
    var acc = 0.0;
    for (var di = 0; di < 6; di++) {
        let z  = (f32(di)/6.0 - 0.5)*2.0;
        let f  = crystal_field(vec3<f32>(uv*2.0/u.zoom, z));
        acc += f*f;
    }
    acc /= 6.0;
    let hue = fract(acc*1.5 + u.color_shift + u.time*u.speed*0.05);
    var col  = 0.45 + 0.45*cos(TAU*(hue + vec3<f32>(0.0,0.33,0.67)));
    col  = mix(col, u.crystal_color.xyz, 0.25) * pow(acc, 0.4) * 2.0;
    return col;
}

// ── Mode 4: nodal lines ───────────────────────────────────────────────────
fn render_nodal(uv: vec2<f32>) -> vec3<f32> {
    let p  = vec3<f32>(uv*2.5/u.zoom, u.time*u.speed*0.2);
    let f  = crystal_field(p); let f2 = cf2(p);
    let qc = select(vec3<f32>(1.0,0.5,0.2), u.crystal_color.xyz, sign(f)*sign(f2) > 0.0);
    var c  = vec3<f32>(0.01,0.005,0.02);
    c += qc * smoothstep(-1.0,1.0,f*f2) * 0.15;
    c += smoothstep(0.04,0.0, abs(f))   * vec3<f32>(0.9,0.95,1.0) * 1.8;
    c += smoothstep(0.03,0.0, abs(f2))  * u.crystal_color.xyz * 1.4;
    c += smoothstep(0.05,0.0, abs(f)+abs(f2)-0.06) * vec3<f32>(1.0,1.0,0.6) * 2.5;
    return c;
}

// ── Mode 5: volumetric emission cloud ────────────────────────────────────
fn render_cloud(uv: vec2<f32>) -> vec3<f32> {
    var az = u.time*u.speed*0.1;
    if u.mouse_down >= 0.5 { az = u.mouse.x * TAU; }
    let el  = 0.45;
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.0/u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0,1.0,0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    var col      = vec3<f32>(0.0);
    var transmit = 1.0;
    var t        = 0.0;
    let dt       = 0.09;
    for (var i = 0; i < 55; i++) {
        if t > 6.0 || transmit < 0.01 { break; }
        let p   = cp + rd * t;
        let f   = crystal_field(p);
        let f2  = cf2(p);
        let rho = max(mix(f*f, f2*f2, u.field_mix) - u.iso_level*0.05, 0.0);
        let hue = fract(f*1.5 + u.color_shift + u.time*u.speed*0.05);
        let ec  = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
        let op  = rho * dt * 3.5;
        col      += ec * op * transmit * mix(vec3<f32>(1.0), u.crystal_color.xyz, 0.6);
        transmit *= exp(-op * 2.5);
        t += dt;
    }
    return col;
}

// ── Mode 6: quantum phase portrait ───────────────────────────────────────
fn render_phase(uv: vec2<f32>) -> vec3<f32> {
    let z   = u.time * u.speed * 0.12;
    let p   = vec3<f32>(uv * 2.2 / u.zoom, z);
    let f   = crystal_field(p);
    let f2  = cf2(p);
    let amp = sqrt(f*f + f2*f2);
    let ang = atan2(f2, f);
    let hue = fract(ang / TAU + u.color_shift + u.time*u.speed*0.02);
    var col = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
    // Saturate with amplitude — dark at vortex cores
    col *= smoothstep(0.0, 0.3, amp) * 1.6;
    // Phase-singularity glow (both fields near zero)
    let core = 1.0 - smoothstep(0.0, 0.06, amp);
    col += core * vec3<f32>(1.2, 1.3, 2.0);
    // Fringe lines at ±f = 0 and ±f2 = 0
    col += smoothstep(0.02, 0.0, abs(f))  * vec3<f32>(1.0, 0.9, 0.5) * 0.9;
    col += smoothstep(0.02, 0.0, abs(f2)) * vec3<f32>(0.5, 0.9, 1.0) * 0.9;
    // Background tint
    col = mix(vec3<f32>(0.01, 0.005, 0.025), col, smoothstep(0.0, 0.15, amp));
    return col;
}

// ── Mode 7: quantised Fermi stripes (DOS) ────────────────────────────────
fn render_stripes(uv: vec2<f32>) -> vec3<f32> {
    let p  = vec3<f32>(uv * 2.5 / u.zoom, u.time*u.speed*0.12);
    let f  = crystal_field(p);
    let f2 = cf2(p);
    let fm = mix(f, f2, u.field_mix);

    // Sharp equi-value contours — number controlled by iso_level
    let n_lines  = 6.0 + u.iso_level * 14.0;
    let stripe   = fract(fm * n_lines);
    let line_w   = smoothstep(0.06, 0.0, min(stripe, 1.0 - stripe));

    // Slow interference background
    let interf = 0.5 + 0.5*sin(f*TAU*2.5 + f2*TAU*1.7 + u.time*u.speed*0.4);

    let hue = fract(fm * 0.4 + u.color_shift + u.time*u.speed*0.04);
    var col = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
    col = mix(vec3<f32>(0.01,0.005,0.02),
              col * mix(vec3<f32>(1.0), u.crystal_color.xyz, 0.55),
              smoothstep(-0.7, 0.7, fm));
    col += line_w * vec3<f32>(1.0, 0.95, 0.7) * 2.2;
    col += interf * u.crystal_color.xyz * 0.07;
    return col;
}

// ── Mode 8: gradient-warp lensing ────────────────────────────────────────
fn render_warp(uv: vec2<f32>) -> vec3<f32> {
    let z  = u.time * u.speed * 0.1;
    let p0 = vec3<f32>(uv * 2.0 / u.zoom, z);
    let e  = 0.035;
    let gx = crystal_field(p0 + vec3<f32>(e, 0.0, 0.0))
           - crystal_field(p0 - vec3<f32>(e, 0.0, 0.0));
    let gy = crystal_field(p0 + vec3<f32>(0.0, e, 0.0))
           - crystal_field(p0 - vec3<f32>(0.0, e, 0.0));

    let warp   = (u.field_mix * 0.6 + 0.1);
    let pw     = vec3<f32>((uv + vec2<f32>(gx, gy) * warp) * 2.0 / u.zoom, z);
    let f      = crystal_field(pw);
    let f2     = cf2(pw);

    let hue = fract((f + f2)*0.5 + u.color_shift + u.time*u.speed*0.04);
    var col = 0.5 + 0.5*cos(TAU*(hue + vec3<f32>(0.0, 0.333, 0.667)));
    col = mix(vec3<f32>(0.01, 0.005, 0.02),
              col * mix(vec3<f32>(1.0), u.crystal_color.xyz, 0.5),
              smoothstep(-1.0, 1.0, f));
    col += smoothstep(0.018, 0.0, abs(f))  * vec3<f32>(1.0, 0.9, 0.5) * 1.4;
    col += smoothstep(0.022, 0.0, abs(f2)) * u.crystal_color.xyz * 1.2;
    // Warp-magnitude glow at high-gradient regions
    let grad_mag = sqrt(gx*gx + gy*gy);
    col += grad_mag * u.crystal_color.xyz * 0.3;
    return col;
}

// ── Mode 9: GLKITTY-style hollow sphere + crystal chain links ────────────
fn smin_links(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

fn sd_link(p: vec3<f32>, le: f32, r1: f32, r2: f32) -> f32 {
    // distance to one chain-link (opened torus capped at ±le)
    let q = vec3<f32>(p.x, max(abs(p.y) - le, 0.0), p.z);
    return length(vec2<f32>(length(q.xy) - r1, q.z)) - r2;
}

fn links_scene(p: vec3<f32>) -> f32 {
    // crystal field drives subtle surface deformation (mask fades at radius)
    let noise  = crystal_field(p * 0.85) * 0.10 * max(1.0 - length(p) * 0.9, 0.0);
    let q      = p + noise;

    // hollow sphere centred at origin
    let sphere = abs(length(q) - 0.65) - 0.04;

    // chain links: repeat along y, alternate 90° rotation each cell
    let cell_sz = 1.1;
    let cell_y  = floor(q.y / cell_sz + 0.5);
    let lp      = vec3<f32>(q.x, q.y - cell_y * cell_sz, q.z);
    var lq: vec3<f32>;
    if (i32(cell_y) & 1) != 0 {
        lq = vec3<f32>(lp.z, lp.y, lp.x);   // perpendicular link plane
    } else {
        lq = lp;
    }
    let link = sd_link(lq, 0.28, 0.22, 0.035);
    return smin_links(sphere, link, 0.18);
}

fn links_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.003;
    return normalize(vec3<f32>(
        links_scene(p + vec3<f32>(e, 0.0, 0.0)) - links_scene(p - vec3<f32>(e, 0.0, 0.0)),
        links_scene(p + vec3<f32>(0.0, e, 0.0)) - links_scene(p - vec3<f32>(0.0, e, 0.0)),
        links_scene(p + vec3<f32>(0.0, 0.0, e)) - links_scene(p - vec3<f32>(0.0, 0.0, e)),
    ));
}

fn render_links(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed;
    var az = t * 0.22;
    var el = 0.38;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.2;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (3.2 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    // background: deep navy tinted by crystal field along ray direction
    let bg_f = crystal_field(rd * 1.8);
    var col  = mix(vec3<f32>(0.02, 0.01, 0.05),
                   vec3<f32>(0.04, 0.12, 0.18),
                   smoothstep(-0.6, 0.6, bg_f)) * 0.55;

    var ray_t = 0.0;
    for (var i = 0; i < 90; i++) {
        let p = cp + rd * ray_t;
        let d = links_scene(p);
        if d < 0.003 {
            let nm     = links_normal(p);
            let cf_val = crystal_field(p);
            let cf2v   = cf2(p);

            // teal-cyan-purple palette keyed to crystal field + normal
            let hue  = fract(cf_val * 1.4 + cf2v * 0.5 + u.color_shift + t * 0.04);
            var surf = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
            surf = mix(surf, vec3<f32>(0.10, 0.72, 0.82), 0.38);   // teal bias
            surf = mix(surf, u.crystal_color.xyz, 0.22);

            let light = normalize(vec3<f32>(1.5, 2.0, 1.0));
            let diff  = max(dot(nm, light), 0.0);
            let rim   = pow(1.0 - abs(dot(nm, -rd)), 2.5);
            let spec  = pow(max(dot(reflect(-light, nm), -rd), 0.0), 32.0);

            col = surf * (0.3 + 0.7 * diff)
                + rim  * vec3<f32>(0.3, 0.8, 1.0) * 0.65
                + surf * spec * 0.50;
            // subsurface crystal field glow (teal)
            col += cf_val * cf_val * vec3<f32>(0.2, 0.8, 1.0) * 0.40;
            // AO proxy via step count
            col *= 0.35 + 0.65 * (1.0 - f32(i) / 90.0);
            break;
        }
        if ray_t > 6.5 { break; }
        ray_t += max(d, 0.003);
    }
    return col;
}

// ── Mode 10: simulated X-ray diffraction pattern ─────────────────────────
// Orthographic projection of reciprocal-lattice G vectors onto a virtual
// detector.  Rotating crystal → single-crystal oscillation photograph.
// Radial power  → Debye-Scherrer powder rings in the background.
// Controls:
//   mouse drag   = crystal orientation
//   iso_level    = spot sharpness (high = sharper)
//   field_mix    = single-crystal ↔ powder ring blend
//   zoom         = detector scale
//   color_shift  = palette shift
fn render_xrd(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed * 0.06;
    var az = t;
    var el = 0.15;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.0;
    }
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);

    let sc    = 0.75 / u.zoom;
    // iso_level 0..1 → sigma 0.014..0.003
    let sigma = max(0.014 - u.iso_level * 0.011, 0.003);
    // Ewald-sphere thickness: how many G vectors "light up" at once
    let ewald_k = 20.0 + u.iso_level * 60.0;

    var spots  = 0.0;
    var powder = 0.0;

    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp = ga.w;
        let G   = ga.xyz * u.kscale;

        // rotate G with the crystal
        let gx  = G.x*ca - G.z*sa;
        let gyr = G.x*sa + G.z*ca;
        let gy  = G.y*ce - gyr*se;
        let gz  = G.y*se + gyr*ce;

        // single-crystal spot: Ewald weighting when gz ≈ 0
        let ew = exp(-gz*gz * ewald_k);
        let sx = gx * sc;  let sy = gy * sc;
        let d2 = (uv.x - sx)*(uv.x - sx) + (uv.y - sy)*(uv.y - sy);
        spots += amp*amp * ew * exp(-d2 / (sigma*sigma));

        // powder rings: intensity at radial distance |G|
        let G_r = length(G) * sc;
        let dr  = length(uv) - G_r;
        powder += amp*amp * exp(-dr*dr * 28000.0);
    }

    // colors
    let hue_shift = u.color_shift + t * 0.03;
    let ring_col  = mix(vec3<f32>(0.55, 0.28, 0.04), u.crystal_color.xyz, 0.4);
    let spot_col  = mix(vec3<f32>(0.82, 0.93, 1.00), u.crystal_color.xyz, 0.25);

    var col = vec3<f32>(0.005, 0.003, 0.012);        // near-black film bg

    // blend powder ↔ single-crystal via field_mix
    col += powder * ring_col * mix(0.5, 0.0, u.field_mix);
    col += spots  * spot_col * mix(0.4, 1.2, u.field_mix);

    // colour the spots by crystal field at the G vector (hkl colour coding)
    let cf_bg = crystal_field(vec3<f32>(uv * 2.0 / u.zoom, t * 0.2)) * 0.015;
    col += cf_bg * u.crystal_color.xyz;

    let r = length(uv);
    // beam-stop: hard shadow for the central direct beam
    col *= smoothstep(0.020, 0.036, r);
    // faint transmitted-beam halo
    col += exp(-r*r * 500.0) * vec3<f32>(1.0, 0.92, 0.70) * 0.25;
    // circular detector boundary vignette
    col *= 1.0 - smoothstep(0.80, 0.98, r / (1.0 / u.zoom));

    return col;
}

// ── Mode 11: 3-D reciprocal-lattice point cloud ───────────────────────────
// Ray-casts a solid sphere for each G vector.  Depth is resolved per-ray
// (frontmost sphere wins).  Glow halos add soft depth cues for occluded
// points.  Origin is marked with a bright white sphere.
// Controls:
//   mouse drag  = orbit camera
//   iso_level   = sphere size scale
//   color_shift = palette rotation
//   zoom        = camera distance
fn render_recip3d(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed * 0.12;
    var az = t;
    var el = 0.30;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let cp  = vec3<f32>(sin(az)*cos(el), sin(el), cos(az)*cos(el)) * (4.5 / u.zoom);
    let fwd = normalize(-cp);
    let rgt = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up  = cross(fwd, rgt);
    let rd  = normalize(fwd + uv.x*rgt + uv.y*up);

    let G_scale  = 0.42;                          // reciprocal-space → world units
    let base_r   = mix(0.025, 0.10, u.iso_level); // sphere radius range

    var best_t   = 1e9;
    var best_col = vec3<f32>(0.0);
    var got_hit  = false;

    // ── origin marker (bright white) ──────────────────────────────────────
    {
        let r   = base_r * 1.4;
        let oc  = -cp;
        let b   = dot(oc, rd);
        let h   = b*b - dot(oc, oc) + r*r;
        if h >= 0.0 {
            let th = b - sqrt(h);
            if th > 0.0 {
                best_t   = th;
                got_hit  = true;
                let nm   = normalize(cp + rd*th);
                let diff = max(dot(nm, normalize(vec3<f32>(1.0, 2.0, 1.5))), 0.0);
                best_col = vec3<f32>(1.0, 0.97, 0.85) * (0.35 + 0.65*diff);
            }
        }
    }

    // ── G-vector spheres ──────────────────────────────────────────────────
    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga  = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp = ga.w;
        let G   = ga.xyz * u.kscale * G_scale;

        let r   = base_r * (0.5 + amp);          // bigger sphere = stronger reflection
        let oc  = G - cp;
        let b   = dot(oc, rd);
        let h   = b*b - dot(oc, oc) + r*r;
        if h < 0.0 { continue; }
        let th  = b - sqrt(h);
        if th <= 0.0 || th >= best_t { continue; }

        best_t  = th;
        got_hit = true;

        let p   = cp + rd * th;
        let nm  = normalize(p - G);

        let hue  = fract(amp*1.8 + length(G)*0.9 + u.color_shift + t*0.025);
        var surf = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
        surf = mix(surf, u.crystal_color.xyz, 0.30);

        let light = normalize(vec3<f32>(2.0, 3.0, 1.5));
        let diff  = max(dot(nm, light), 0.0);
        let spec  = pow(max(dot(reflect(-light, nm), -rd), 0.0), 28.0);
        let rim   = pow(1.0 - abs(dot(nm, -rd)), 2.2);

        best_col = surf * (0.25 + 0.75*diff)
                 + rim  * vec3<f32>(0.4, 0.75, 1.0) * 0.35
                 + spec * 0.55;
        best_col *= (0.4 + 0.6*amp);             // dim weak reflections
    }

    var col = vec3<f32>(0.005, 0.003, 0.012);     // near-black bg
    if got_hit { col = best_col; }

    // ── glow halos — add soft depth cue for all G vectors ─────────────────
    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga    = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp   = ga.w;
        let G     = ga.xyz * u.kscale * G_scale;
        let oc    = G - cp;
        let proj  = dot(oc, rd);
        if proj < 0.1 { continue; }
        let cl    = cp + rd*proj;
        let dist2 = dot(G - cl, G - cl);
        let gr    = base_r * (0.8 + amp) * 2.5;  // halo wider than sphere
        col += amp*amp * exp(-dist2 / (gr*gr)) * u.crystal_color.xyz * 0.12;
    }

    // ── faint spokes from origin to each G vector ─────────────────────────
    for (var i = 0; i < 128; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga   = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let amp  = ga.w;
        let G    = ga.xyz * u.kscale * G_scale;
        // closest distance from ray to the line segment origin→G
        let oc   = -cp;
        let ab   = G;
        let ab2  = dot(ab, ab);
        let s    = clamp(dot(rd, ab) / max(ab2, 0.0001), 0.0, 1.0);
        // closest point on the line segment
        let seg_pt = s * ab;
        let ray_t2 = dot(seg_pt - cp, rd);
        let ray_pt = cp + rd * max(ray_t2, 0.0);
        let d2    = dot(seg_pt - ray_pt, seg_pt - ray_pt);
        let spoke_w = 0.007 * amp;
        col += amp * exp(-d2 / (spoke_w*spoke_w)) * 0.06
             * mix(vec3<f32>(0.3,0.4,0.5), u.crystal_color.xyz, 0.5);
    }

    return col;
}

// ── Mode 12: non-Euclidean kaleidoscope (Kleinian circle inversions) ─────
// Alternately inverts the pixel coordinate through circles centred on the
// first 4 projected G vectors — each inversion is a "reflection" in the
// hyperbolic metric, producing a fractal Kleinian / Schottky tiling.
// The crystal field is sampled at every depth level so the pattern changes
// completely with each crystal in the tour.
// Controls:
//   mouse drag  = orbit (rotates generator circles)
//   iso_level   = inversion-circle radius scale
//   color_shift = palette rotation
fn circle_inv(z: vec2<f32>, c: vec2<f32>, r: f32) -> vec2<f32> {
    let d = z - c;
    return c + r * r * d / max(dot(d, d), 1e-15);
}

fn render_noneuclidean(uv: vec2<f32>) -> vec3<f32> {
    let t  = u.time * u.speed * 0.07;
    var az = t * 0.18;
    var el = 0.28;
    if u.mouse_down >= 0.5 {
        az = u.mouse.x * TAU;
        el = (u.mouse.y - 0.5) * 2.5;
    }
    let ca = cos(az); let sa = sin(az);
    let ce = cos(el); let se = sin(el);

    // Project first 4 G vectors into the 2D screen for circle centres
    let sc = u.kscale * 0.40;
    var ctrs: array<vec2<f32>, 4>;
    var rads: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) {
        let ga  = textureLoad(g_tex, vec2<i32>(i32(i), 0), 0);
        let G   = ga.xyz * sc;
        let gx  = G.x*ca - G.z*sa;
        let gyr = G.x*sa + G.z*ca;
        let gy  = G.y*ce - gyr*se;
        ctrs[i] = vec2<f32>(gx, gy);
        // iso_level 0..1 → radius factor 0.45..0.80
        rads[i] = max(length(ctrs[i]) * (0.45 + u.iso_level * 0.35), 0.04);
    }

    var z    = uv;
    var col  = vec3<f32>(0.0);
    var last = -1i;   // index of circle used in previous iteration (no back-flipping)

    for (var iter = 0i; iter < 52i; iter++) {
        // Find the first circle (other than last) that contains z
        var inv_c = -1i;
        for (var ci = 0i; ci < 4i; ci++) {
            if ci != last && inv_c < 0i {
                let d2 = dot(z - ctrs[u32(ci)], z - ctrs[u32(ci)]);
                if d2 < rads[u32(ci)] * rads[u32(ci)] { inv_c = ci; }
            }
        }
        if inv_c < 0i { break; }   // no circle contains z → in fundamental domain

        let ic = u32(inv_c);
        z    = circle_inv(z, ctrs[ic], rads[ic]);
        last = inv_c;

        // Sample crystal field at this folded coordinate
        let f   = crystal_field(vec3<f32>(z,        t + f32(iter) * 0.08));
        let f2v = cf2(          vec3<f32>(z * 1.17, t + f32(iter) * 0.13));
        let dcay = exp(-f32(iter) * 0.050);
        let hue  = fract(f32(ic) * 0.25 + f * 0.9 + f2v * 0.4 + u.color_shift + t * 0.04);
        var  tc  = 0.5 + 0.5 * cos(TAU * (hue + vec3<f32>(0.0, 0.333, 0.667)));
        tc   = mix(tc, u.crystal_color.xyz, 0.28);
        col += tc * dcay * (0.08 + 0.45 * abs(f));
    }

    // Crystal field at the final deepest-folded position
    let f_fin  = crystal_field(vec3<f32>(z * 1.4, t * 0.18));
    let f2_fin = cf2(          vec3<f32>(z * 1.6, t * 0.22));
    let hue_f  = fract(f_fin * 1.1 + f2_fin * 0.5 + u.color_shift + t * 0.025);
    var  base  = 0.5 + 0.5 * cos(TAU * (hue_f + vec3<f32>(0.0, 0.333, 0.667)));
    base  = mix(base, u.crystal_color.xyz, 0.28);
    col   = col * 0.72 + base * (0.10 + 0.32 * abs(f_fin));

    return col;
}

@fragment fn fs_field(f: VOut) -> @location(0) vec4<f32> {
    let uv = (f.uv*2.0 - 1.0) * vec2<f32>(u.aspect, 1.0);
    var col: vec3<f32>;
    switch u.mode {
        case 0u  { col = render_rm(uv); }
        case 1u  { col = render_bz(uv); }
        case 2u  { col = render_fermi(uv); }
        case 3u  { col = render_density(uv); }
        case 4u  { col = render_nodal(uv); }
        case 5u  { col = render_cloud(uv); }
        case 6u  { col = render_phase(uv); }
        case 7u  { col = render_stripes(uv); }
        case 8u  { col = render_warp(uv); }
        case 9u  { col = render_links(uv); }
        case 10u { col = render_xrd(uv); }
        case 11u { col = render_recip3d(uv); }
        default  { col = render_noneuclidean(uv); }
    }
    // vignette — skip for XRD and RECIP3D (own detector/space boundaries)
    if u.mode != 10u && u.mode != 11u {
        col *= 1.0 - 0.35*dot(f.uv*2.0-1.0, f.uv*2.0-1.0);
    }
    // tone-map — skip for XRD (preserves sharp spots)
    if u.mode != 10u {
        col = col / (col + vec3<f32>(0.6));
        col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(0.85));
    }
    return vec4<f32>(col, 1.0);
}
"#;
