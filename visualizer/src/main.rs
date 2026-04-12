use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

// ── GPU state ──────────────────────────────────────────────────────────────

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    pipeline: wgpu::RenderPipeline,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        log::info!("Adapter: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("crystal-viz device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let pipeline = build_pipeline(&device, format);

        Self { surface, device, queue, config, size, window, pipeline }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Deep-space dark purple background
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.02,
                            b: 0.10,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            // Draw 3 vertices → one glowing triangle
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }
}

// ── Pipeline ───────────────────────────────────────────────────────────────

fn build_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("triangle shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("triangle pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

// ── WGSL shader ────────────────────────────────────────────────────────────
// A hardcoded triangle with per-vertex neon colours.
// Position is in clip space; colour is interpolated across the face.

const SHADER: &str = r#"
struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       col: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.6),   // top
        vec2<f32>(-0.6, -0.5),   // bottom-left
        vec2<f32>( 0.6, -0.5),   // bottom-right
    );
    // Neon: magenta / cyan / yellow
    var colours = array<vec4<f32>, 3>(
        vec4<f32>(1.0, 0.2, 0.9, 1.0),
        vec4<f32>(0.2, 0.9, 1.0, 1.0),
        vec4<f32>(1.0, 0.9, 0.1, 1.0),
    );
    var out: VertexOut;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    out.col = colours[vi];
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.col;
}
"#;

// ── App shell ──────────────────────────────────────────────────────────────

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title("crystal-viz  —  hello wgpu")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.state = Some(pollster::block_on(State::new(window)));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => {
                state.window.request_redraw();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let sz = state.size;
                        state.resize(sz);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory — exiting");
                        event_loop.exit();
                    }
                    Err(e) => log::warn!("Surface error: {e:?}"),
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
