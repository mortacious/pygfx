"""
Screen Space Ambient Occlusion (SSAO) post-processing pass for pygfx.

This pass estimates ambient occlusion from the depth buffer in screen space.

Notes:
- This is a compact SSAO variant that uses a fixed sample kernel in a small
  hemisphere around the pixel and a fast approximate normal from depth.
- It is intended as a reasonable default; for production you may want bilateral
  blur passes and more sophisticated reprojection/temporal filtering.
"""

from .effectpasses import EffectPass


class SSAOPass(EffectPass):
    """Screen Space Ambient Occlusion.

    Parameters
    ----------
    radius : float
        Sampling radius in pixels for the AO samples. Default 4.0.
    bias : float
        Depth bias to reduce self-occlusion artifacts. Default 0.025.
    intensity : float
        Strength of the occlusion effect. Default 1.0.
    power : float
        Final power curve applied to the occlusion factor. Default 1.0.
    """

    USES_DEPTH = True

    uniform_type = dict(
        EffectPass.uniform_type,
        radius="f4",
        bias="f4",
        intensity="f4",
        power="f4",
        normalize_depth="f4",
        falloff="f4",
        num_samples="f4",
    )

    # WGSL fragment shader implementing a compact SSAO
    wgsl = """
        const AO_SCALE: f32 = 200.0;
        const GOLDEN_ANGLE: f32 = 2.39996323; // ~pi * (3 - sqrt(5))
        // Utility: approximate view-space normal using depth derivatives in screen space
        fn approximate_normal(depthTex: texture_depth_2d, texIndex: vec2<i32>) -> vec3<f32> {
            let dims = textureDimensions(depthTex);

            let left  = max(0, texIndex.x - 1);
            let right = min(i32(dims.x) - 1, texIndex.x + 1);
            let up    = max(0, texIndex.y - 1);
            let down  = min(i32(dims.y) - 1, texIndex.y + 1);

            let dC = textureLoad(depthTex, texIndex, 0);
            let dL = textureLoad(depthTex, vec2<i32>(left, texIndex.y), 0);
            let dR = textureLoad(depthTex, vec2<i32>(right, texIndex.y), 0);
            let dU = textureLoad(depthTex, vec2<i32>(texIndex.x, up), 0);
            let dD = textureLoad(depthTex, vec2<i32>(texIndex.x, down), 0);

            // Form gradients; depth increases with distance, not linear in view, but good enough
            let dx = dR - dL;
            let dy = dD - dU;

            // Build a pseudo normal in screen space
            var n = normalize(vec3<f32>(-dx, -dy, 1.0));
            return n;
        }

        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texIndex = vec2i(varyings.position.xy);

            let depth_c = textureLoad(depthTex, texIndex, 0);
            let color_c = textureLoad(colorTex, texIndex, 0);

            if (depth_c >= 1.0) {
                return color_c;
            }

            let dims = textureDimensions(depthTex);
            let max_i = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);

            let n = approximate_normal(depthTex, texIndex);

            // Depth-normalized sampling radius (approximate)
            var radius_px = u_effect.radius;
            if (u_effect.normalize_depth > 0.5) {
                radius_px = u_effect.radius * max(depth_c, 1e-3);
            }

            var occl: f32 = 0.0;
            var taps: f32 = 0.0;

            // Golden-angle spiral sampling with smooth falloff and distance weighting
            let ns_i = max(4, i32(round(u_effect.num_samples)));
            let ns = f32(ns_i);
            var angle: f32 = 0.0;
            for (var i: i32 = 0; i < ns_i; i = i + 1) {
                angle = angle + GOLDEN_ANGLE;
                let dir = vec2<f32>(cos(angle), sin(angle));
                let r = (f32(i) + 0.5) / ns; // 0..1 radial
                let off_f = dir * radius_px * r;
                let off = vec2<i32>(i32(round(off_f.x)), i32(round(off_f.y)));
                var ni = texIndex + off;
                ni = clamp(ni, vec2<i32>(0, 0), max_i);

                let depth_n = textureLoad(depthTex, ni, 0);
                if (depth_n < 0.9999) {
                    var dd = (depth_c - depth_n) * AO_SCALE;
                    if (u_effect.normalize_depth > 0.5) {
                        dd = dd / max(depth_c, 1e-3);
                    }
                    // Depth-scaled bias
                    let bias = select(u_effect.bias, u_effect.bias * max(depth_c, 1e-3), u_effect.normalize_depth > 0.5);
                    let dd2 = max(0.0, dd - bias);
                    // Smooth range falloff
                    let range_w = smoothstep(0.0, u_effect.falloff, dd2);
                    // Distance weighting (favor near samples)
                    let dist_w = 1.0 - r;
                    // Facing term
                    let dirv = normalize(vec3<f32>(off_f.x, off_f.y, 1.0));
                    let facing = max(0.0, dot(n, dirv));

                    let w = range_w * dist_w * facing;
                    occl = occl + w;
                    taps = taps + dist_w;
                }
            }

            let ao_raw = clamp(occl / max(taps, 1e-5), 0.0, 1.0);
            // Map to occlusion factor; higher occl -> darker
            let ao = pow(1.0 - ao_raw * u_effect.intensity, u_effect.power);
            let shaded_rgb = color_c.rgb * ao;
            return vec4<f32>(shaded_rgb, color_c.a);
        }
    """

    def __init__(self, *, radius=4.0, bias=0.025, intensity=1.0, power=1.0, normalize_depth=False, falloff=0.5, num_samples=16):
        super().__init__()
        self.radius = radius
        self.bias = bias
        self.intensity = intensity
        self.power = power
        self.normalize_depth = normalize_depth
        self.falloff = falloff
        self.num_samples = num_samples

    @property
    def radius(self):
        return float(self._uniform_data["radius"])

    @radius.setter
    def radius(self, value):
        self._uniform_data["radius"] = float(value)

    @property
    def bias(self):
        return float(self._uniform_data["bias"])

    @bias.setter
    def bias(self, value):
        self._uniform_data["bias"] = float(value)

    @property
    def intensity(self):
        return float(self._uniform_data["intensity"])

    @intensity.setter
    def intensity(self, value):
        self._uniform_data["intensity"] = float(value)

    @property
    def power(self):
        return float(self._uniform_data["power"])

    @power.setter
    def power(self, value):
        self._uniform_data["power"] = float(value)

    @property
    def normalize_depth(self):
        return bool(self._uniform_data["normalize_depth"])

    @normalize_depth.setter
    def normalize_depth(self, value):
        self._uniform_data["normalize_depth"] = 1.0 if bool(value) else 0.0

    @property
    def falloff(self):
        return float(self._uniform_data["falloff"])

    @falloff.setter
    def falloff(self, value):
        self._uniform_data["falloff"] = float(value)

    @property
    def num_samples(self):
        return int(self._uniform_data["num_samples"])

    @num_samples.setter
    def num_samples(self, value):
        self._uniform_data["num_samples"] = float(int(value))


