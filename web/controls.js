import { app } from "../../../scripts/app.js";

const ORDER = ["attention_backend", "sol_quality_profile", "sol_tau", "easycache_mode", "dual_gpu", "dual_gpu_secondary"];
const LEGACY = [
    "mixed_precision", "attention_backend", "sol_tau", "sol_start_percent", "sol_end_percent",
    "long_projection_reuse", "sol_topk_tail_blocks", "sol_min_tokens", "sol_minimum_gain_percent",
    "sol_route_memory_limit_mib", "easycache_mode", "easycache_video_threshold",
    "easycache_video_frame_p95_threshold", "easycache_audio_threshold", "easycache_start_percent",
    "easycache_end_percent", "easycache_video_storage", "easycache_memory_limit_mib", "easycache_save_report",
    "sol_quality_profile", "easycache_quality_profile", "sol_adaptive_budget", "dual_gpu", "dual_gpu_secondary",
];
const DEFAULTS = ["Flash", "quality", 1.0, "Off", false, "auto"];
const LABELS = {attention_backend: "Backend", sol_quality_profile: "SOL_Quality",
    sol_tau: "SOL_Tau", easycache_mode: "EasyCache", dual_gpu: "Dual_GPU",
    dual_gpu_secondary: "Dual_GPU_ID"};
const PRESETS = {quality: 1.0, speed: 2.0, ultra: 2.5};
const NODE = "H3V100Optimize";
const backend = value => ({flash_attn: "Flash", sol_attn: "SOL"})[value] ?? value;
const cacheMode = (value, profile) => {
    if (["Off", "Quality", "Speed"].includes(value)) return value;
    if (value === true || value === "active") return profile === "speed" ? "Speed" : "Quality";
    return "Off";
};

function migrate(info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values)) return info;
    if (values.length > ORDER.length && typeof values[0] === "boolean" && ["flash_attn", "sol_attn"].includes(values[1])) {
        const old = Object.fromEntries(LEGACY.map((key, i) => [key, values[i]]));
        const next = ORDER.map((key, i) => old[key] ?? DEFAULTS[i]);
        next[0] = backend(next[0]);
        next[3] = cacheMode(next[3], old.easycache_quality_profile);
        // An old Adaptive Budget slot must never become the dual-card switch.
        next[4] = old.dual_gpu === true;
        return {...info, widgets_values: next};
    }
    if (values.length === ORDER.length) {
        const next = [...values];
        next[0] = backend(next[0]);
        next[3] = cacheMode(next[3]);
        return {...info, widgets_values: next};
    }
    return info;
}

function hide(widget, hidden) {
    if (!widget) return;
    if (!widget._h3Display) widget._h3Display = {type: widget.type, computeSize: widget.computeSize};
    widget.hidden = hidden;
    widget.type = hidden ? "hidden" : widget._h3Display.type;
    widget.computeSize = hidden ? () => [0, -4] : widget._h3Display.computeSize;
}

app.registerExtension({
    name: "H3.V100.Controls",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info, ...args) {
            const result = configure.call(this, migrate(info), ...args);
            this._h3UpdateControls?.();
            return result;
        };
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE) return;
        const widget = name => node.widgets?.find(item => item.name === name);
        node._h3UpdateControls = () => {
            for (const [name, label] of Object.entries(LABELS)) {
                if (widget(name)) widget(name).label = label;
            }
            const sol = widget("attention_backend")?.value === "SOL";
            hide(widget("sol_tau"), !sol || widget("sol_quality_profile")?.value !== "manual");
            hide(widget("dual_gpu_secondary"), widget("dual_gpu")?.value !== true);
            node.setSize?.(node.computeSize());
            node.graph?.setDirtyCanvas(true, true);
        };
        for (const name of ["attention_backend", "sol_quality_profile", "dual_gpu"]) {
            const item = widget(name);
            if (!item) continue;
            const callback = item.callback;
            item.callback = function(value, ...args) {
                const result = callback?.call(this, value, ...args);
                if (name === "sol_quality_profile" && Object.hasOwn(PRESETS, value)) {
                    widget("sol_tau").value = PRESETS[value];
                }
                node._h3UpdateControls();
                return result;
            };
        }
        node._h3UpdateControls();
    },
});
