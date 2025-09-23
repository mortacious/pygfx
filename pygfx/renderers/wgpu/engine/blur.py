"""
Depth-aware bilateral blur post-processing pass for pygfx.

This pass performs a small-kernel bilateral blur guided by the depth buffer to
smooth noisy post-effects (e.g. SSAO) while preserving edges.
"""

from .effectpasses import EffectPass


class BilateralBlurPass(EffectPass):
    """Depth-aware bilateral blur.

    Parameters
    ----------
    radius : int | float
        Pixel radius for the kernel footprint (use small values like 1–3). Default 2.0.
    sigma_spatial : float
        Sigma for spatial Gaussian (in pixels). Default 2.0.
    sigma_depth : float
        Sigma for depth Gaussian (in depth units 0..1). Default 0.02.
    """

    USES_DEPTH = True

    uniform_type = dict(
        EffectPass.uniform_type,
        radius="f4",
        sigma_spatial="f4",
        sigma_depth="f4",
    )

    wgsl = """
        // Bilateral blur: weight by spatial distance and depth similarity
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texIndex = vec2i(varyings.position.xy);

            let center_color = textureLoad(colorTex, texIndex, 0);
            let center_depth = textureLoad(depthTex, texIndex, 0);

            if (center_depth >= 0.9999) {
                return center_color;
            }

            let dims = textureDimensions(depthTex);
            let max_i = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);

            let r = i32(round(u_effect.radius));
            let two_sigma_s2 = 2.0 * u_effect.sigma_spatial * u_effect.sigma_spatial;
            let two_sigma_d2 = 2.0 * u_effect.sigma_depth * u_effect.sigma_depth;

            var sum_color: vec3<f32> = vec3<f32>(0.0);
            var sum_w: f32 = 0.0;

            for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
                for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
                    let off = vec2<i32>(dx, dy);
                    var ni = texIndex + off;
                    ni = clamp(ni, vec2<i32>(0, 0), max_i);

                    let c = textureLoad(colorTex, ni, 0).rgb;
                    let d = textureLoad(depthTex, ni, 0);

                    // Spatial distance (pixels)
                    let dist2 = f32(dx*dx + dy*dy);
                    let w_s = exp(-dist2 / max(two_sigma_s2, 1e-6));

                    // Depth similarity (range component)
                    let dz = center_depth - d;
                    let w_d = exp(-(dz*dz) / max(two_sigma_d2, 1e-8));

                    let w = w_s * w_d;
                    sum_color = sum_color + c * w;
                    sum_w = sum_w + w;
                }
            }

            let out_rgb = mix(center_color.rgb, sum_color / max(sum_w, 1e-6), 1.0);
            return vec4<f32>(out_rgb, center_color.a);
        }
    """

    def __init__(self, *, radius=2.0, sigma_spatial=2.0, sigma_depth=0.02):
        super().__init__()
        self.radius = radius
        self.sigma_spatial = sigma_spatial
        self.sigma_depth = sigma_depth

    @property
    def radius(self):
        return float(self._uniform_data["radius"])

    @radius.setter
    def radius(self, value):
        self._uniform_data["radius"] = float(value)

    @property
    def sigma_spatial(self):
        return float(self._uniform_data["sigma_spatial"])

    @sigma_spatial.setter
    def sigma_spatial(self, value):
        self._uniform_data["sigma_spatial"] = float(value)

    @property
    def sigma_depth(self):
        return float(self._uniform_data["sigma_depth"])

    @sigma_depth.setter
    def sigma_depth(self, value):
        self._uniform_data["sigma_depth"] = float(value)


