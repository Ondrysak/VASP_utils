// ── Fragment-shader dispatch ─────────────────────────────────────────────
// Switches on `u.mode` and calls the matching `render_<name>(uv)` function.
// Adding a mode = add a new file + a case here + bump MODE_NAMES in main.rs.

@fragment fn fs_field(f: VOut) -> @location(0) vec4<f32> {
    let uv = (f.uv*2.0 - 1.0) * vec2<f32>(u.aspect, 1.0);
    var col: vec3<f32>;
    switch u.mode {
        // ── core modes (0..12) ────────────────────────────────────────────
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
        case 12u { col = render_noneuclidean(uv); }
        // ── new modes (13..20) ────────────────────────────────────────────
        case 13u { col = render_phonon(uv); }
        case 14u { col = render_moire(uv); }
        case 15u { col = render_ewald(uv); }
        case 16u { col = render_wannier(uv); }
        case 17u { col = render_magnetic(uv); }
        case 18u { col = render_dispersion(uv); }
        case 19u { col = render_kikuchi(uv); }
        default  { col = render_defect(uv); }
    }
    // vignette — skip for XRD, RECIP3D, KIKUCHI (own boundaries)
    if u.mode != 10u && u.mode != 11u && u.mode != 19u {
        col *= 1.0 - 0.35*dot(f.uv*2.0-1.0, f.uv*2.0-1.0);
    }
    // tone-map — skip for XRD (preserves sharp spots)
    if u.mode != 10u {
        col = col / (col + vec3<f32>(0.6));
        col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(0.85));
    }
    return vec4<f32>(col, 1.0);
}
