export function clamp(min, max) {
    return (value) => Math.min(max, Math.max(min, value));
}