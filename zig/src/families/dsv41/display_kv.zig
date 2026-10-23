//! The KV pool's first planes in the GB10's UEFI display reservation (TF_DS_DISPLAY_KV_MIB).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const prompt = @import("prompt.zig");
const ops_mod = @import("ops.zig");

const linux = std.os.linux;

pub const env_name = "TF_DS_DISPLAY_KV_MIB";
pub const card_env = "TF_DS_DISPLAY_KV_CARD";
pub const default_card = "/dev/dri/card0";
pub const default_sysfs = "/sys/class/drm";
pub const step_mib: usize = 16;
pub const largest_mib: usize = 2032;
pub const span_align: usize = 65536;
pub const max_planes: usize = 64;
pub const copy_chunk: usize = 128 << 20;

const drm_ioctl_mode_create_dumb: u32 = 0xC02064B2;
const drm_ioctl_mode_map_dumb: u32 = 0xC01064B3;
const drm_ioctl_gem_close: u32 = 0x40086409;
const dumb_width: u32 = 4096;
const dumb_bpp: u32 = 32;
const dumb_pitch: usize = 16384;
const host_register_devicemap: c_uint = 0x02;
const host_register_iomemory: c_uint = 0x04;

pub const Error = error{
    BadDisplayKv,
    DisplayConnected,
    DisplayCardUnavailable,
    DisplayIoctlFailed,
    ShortDisplayBuffer,
    DisplayMmapFailed,
    DisplayRegisterFailed,
    DisjointDisplaySpan,
    DisplayKvTooLarge,
};

const CreateDumb = extern struct {
    height: u32,
    width: u32,
    bpp: u32,
    flags: u32 = 0,
    handle: u32 = 0,
    pitch: u32 = 0,
    size: u64 = 0,
};

const MapDumb = extern struct {
    handle: u32,
    pad: u32 = 0,
    offset: u64 = 0,
};

const GemClose = extern struct {
    handle: u32,
    pad: u32 = 0,
};

pub const DumbBuf = struct {
    handle: u32,
    size: u64,
};

pub const Plan = struct {
    planes: usize,
    ordinary: usize,
    display: usize,
};

/// Parses TF_DS_DISPLAY_KV_MIB: empty or "0" is off (0); else a multiple of 16 up to 2032.
pub fn parseMib(raw: []const u8) Error!usize {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return 0;
    const v = std.fmt.parseInt(usize, trimmed, 10) catch return error.BadDisplayKv;
    return checkMib(v);
}

pub fn checkMib(v: usize) Error!usize {
    if (v == 0) return 0;
    if (v % step_mib != 0 or v > largest_mib) return error.BadDisplayKv;
    return v;
}

pub fn envMib() Error!usize {
    const raw = std.c.getenv(env_name) orelse return 0;
    return parseMib(std.mem.span(raw));
}

/// Bytes one pool position takes across every kv-source layer's compressed and index planes.
pub fn positionBytes(c: *const Config) !usize {
    var total: usize = 0;
    for (0..c.layers) |i| {
        if (!c.kv_sources.has(i)) continue;
        const r: usize = c.ratios[i];
        if (r == 0) return error.BadConfig;
        var row: usize = c.head_dim / 2 + c.head_dim / 16;
        if (c.index_sources.has(i)) row += c.index_head_dim / 2 + c.index_head_dim / 32;
        if (row % r != 0) return error.BadConfig;
        total += row / r;
    }
    if (total == 0) return error.BadConfig;
    return total;
}

/// Positions `display_bytes` of the display reservation add to the pool.
pub fn positionsAdded(c: *const Config, display_bytes: usize) !usize {
    if (display_bytes == 0) return 0;
    return display_bytes / try positionBytes(c);
}

/// The byte sizes of the pool planes for `cap` positions, in the order Caches.init allocates them.
pub fn poolPlaneSizes(c: *const Config, cap: usize, out: *[max_planes]usize) usize {
    var n: usize = 0;
    for (0..c.layers) |i| {
        if (!c.kv_sources.has(i)) continue;
        const r: usize = c.ratios[i];
        const rows = cap / r + 2;
        out[n] = rows * c.head_dim / 2;
        out[n + 1] = rows * c.head_dim / 16;
        n += 2;
        if (c.index_sources.has(i)) {
            out[n] = rows * c.index_head_dim / 2;
            out[n + 1] = rows * c.index_head_dim / 32;
            n += 2;
        }
    }
    return n;
}

/// How many first pool planes (256-byte aligned) cover `display` bytes, and the 64 KiB-aligned ordinary prefix.
pub fn planPlanes(sizes: []const usize, display: usize) Error!Plan {
    if (display == 0 or display % span_align != 0) return error.BadDisplayKv;
    var sum: usize = 0;
    for (sizes, 0..) |raw, idx| {
        sum += std.mem.alignForward(usize, raw, 256);
        if (sum >= display) {
            const total = std.mem.alignForward(usize, sum, span_align);
            return .{ .planes = idx + 1, .ordinary = total - display, .display = display };
        }
    }
    return error.DisplayKvTooLarge;
}

/// Fails with error.DisplayConnected if any `<card>-*/status` in `sysfs_dir` reads `connected`.
pub fn checkHeadlessDir(io: std.Io, sysfs_dir: std.Io.Dir, card: []const u8) Error!void {
    var prefix_buf: [64]u8 = undefined;
    const prefix = std.fmt.bufPrint(&prefix_buf, "{s}-", .{card}) catch return error.DisplayCardUnavailable;
    var it = sysfs_dir.iterate();
    while (it.next(io) catch return error.DisplayCardUnavailable) |entry| {
        if (!std.mem.startsWith(u8, entry.name, prefix)) continue;
        var path_buf: [256]u8 = undefined;
        const status_path = std.fmt.bufPrint(&path_buf, "{s}/status", .{entry.name}) catch return error.DisplayCardUnavailable;
        var f = sysfs_dir.openFile(io, status_path, .{}) catch return error.DisplayCardUnavailable;
        defer f.close(io);
        var status_buf: [64]u8 = undefined;
        const n = f.readPositionalAll(io, &status_buf, 0) catch return error.DisplayCardUnavailable;
        if (std.mem.eql(u8, std.mem.trim(u8, status_buf[0..n], " \t\r\n"), "connected")) {
            return error.DisplayConnected;
        }
    }
}

pub fn checkHeadless(io: std.Io, sysfs_path: []const u8, card: []const u8) Error!void {
    var dir = std.Io.Dir.cwd().openDir(io, sysfs_path, .{ .iterate = true }) catch return error.DisplayCardUnavailable;
    defer dir.close(io);
    return checkHeadlessDir(io, dir, card);
}

/// Every DRM, mmap and CUDA registration call the span makes, replaceable by tests.
pub const Sys = struct {
    ctx: ?*anyopaque = null,
    openCard: *const fn (ctx: ?*anyopaque, path: []const u8) Error!i32 = realOpenCard,
    createDumb: *const fn (ctx: ?*anyopaque, fd: i32, bytes: usize) Error!DumbBuf = realCreateDumb,
    mapOffset: *const fn (ctx: ?*anyopaque, fd: i32, handle: u32) Error!u64 = realMapOffset,
    reserve: *const fn (ctx: ?*anyopaque, bytes: usize) Error!usize = realReserve,
    mapOrdinary: *const fn (ctx: ?*anyopaque, addr: usize, bytes: usize) Error!void = realMapOrdinary,
    mapDisplay: *const fn (ctx: ?*anyopaque, addr: usize, bytes: usize, fd: i32, offset: u64) Error!void = realMapDisplay,
    register: *const fn (ctx: ?*anyopaque, addr: usize, bytes: usize, io_mem: bool) Error!void = realRegister,
    devicePointer: *const fn (ctx: ?*anyopaque, addr: usize) Error!u64 = realDevicePointer,
    unregister: *const fn (ctx: ?*anyopaque, addr: usize) void = realUnregister,
    unmap: *const fn (ctx: ?*anyopaque, addr: usize, bytes: usize) void = realUnmap,
    closeGem: *const fn (ctx: ?*anyopaque, fd: i32, handle: u32) void = realCloseGem,
    closeFd: *const fn (ctx: ?*anyopaque, fd: i32) void = realCloseFd,
};

/// One contiguous UVA span: ordinary host memory followed by the DRM dumb buffer, both registered with CUDA.
pub const Span = struct {
    sys: Sys,
    base: usize,
    ordinary: usize,
    display: usize,
    planes: usize,
    ptr: u64,
    fd: i32,
    handle: u32,

    pub fn total(s: Span) usize {
        return s.ordinary + s.display;
    }

    pub fn cut(s: Span) u64 {
        return s.ptr + s.ordinary;
    }

    pub fn contains(s: Span, dev_ptr: u64) bool {
        return dev_ptr >= s.ptr and dev_ptr < s.ptr + s.total();
    }

    pub fn map(d: *const cuda.Driver, io: std.Io, ordinary: usize, display: usize, planes: usize) Error!Span {
        const card_path = if (std.c.getenv(card_env)) |p| blk: {
            const s = std.mem.trim(u8, std.mem.span(p), " \t\r\n");
            break :blk if (s.len > 0) s else default_card;
        } else default_card;
        const card_name = std.fs.path.basename(card_path);
        try checkHeadless(io, default_sysfs, card_name);
        return mapWith(.{ .ctx = @ptrCast(@constCast(d)) }, card_path, ordinary, display, planes);
    }

    pub fn mapWith(sys: Sys, card_path: []const u8, ordinary: usize, display: usize, planes: usize) Error!Span {
        if (display == 0) return error.BadDisplayKv;
        const total_bytes = ordinary + display;
        const fd = try sys.openCard(sys.ctx, card_path);
        errdefer sys.closeFd(sys.ctx, fd);
        const dumb = try sys.createDumb(sys.ctx, fd, display);
        errdefer sys.closeGem(sys.ctx, fd, dumb.handle);
        if (dumb.size < display) return error.ShortDisplayBuffer;
        const offset = try sys.mapOffset(sys.ctx, fd, dumb.handle);
        const base = try sys.reserve(sys.ctx, total_bytes);
        errdefer sys.unmap(sys.ctx, base, total_bytes);
        if (ordinary > 0) try sys.mapOrdinary(sys.ctx, base, ordinary);
        try sys.mapDisplay(sys.ctx, base + ordinary, display, fd, offset);
        if (ordinary > 0) try sys.register(sys.ctx, base, ordinary, false);
        errdefer if (ordinary > 0) sys.unregister(sys.ctx, base);
        try sys.register(sys.ctx, base + ordinary, display, true);
        errdefer sys.unregister(sys.ctx, base + ordinary);
        const ptr = try sys.devicePointer(sys.ctx, base);
        const top = try sys.devicePointer(sys.ctx, base + ordinary);
        if (top < ptr or top - ptr != ordinary) return error.DisjointDisplaySpan;
        return .{
            .sys = sys,
            .base = base,
            .ordinary = ordinary,
            .display = display,
            .planes = planes,
            .ptr = ptr,
            .fd = fd,
            .handle = dumb.handle,
        };
    }

    pub fn deinit(s: *Span) void {
        s.sys.unregister(s.sys.ctx, s.base + s.ordinary);
        if (s.ordinary > 0) s.sys.unregister(s.sys.ctx, s.base);
        s.sys.unmap(s.sys.ctx, s.base, s.total());
        s.sys.closeGem(s.sys.ctx, s.fd, s.handle);
        s.sys.closeFd(s.sys.ctx, s.fd);
        s.* = undefined;
    }
};

/// Allocates `Caches` with its first `span.planes` pool planes carved from `span` and the rest from `a`.
pub fn initCaches(e: *const prompt.Engine, a: *prompt.Arena, cap: usize, slots: usize, span: *const Span) !prompt.Caches {
    const c = e.c;
    var cs: prompt.Caches = .{ .cap = cap, .slots = slots };
    var sp_at: u64 = span.ptr;
    var sp_left: usize = span.planes;
    const cut = span.cut();
    for (e.w.layers, 0..) |lay, i| {
        if (lay.comp_wkv == null) continue;
        const r: usize = lay.ratio;
        const rows = cap / r + 2;
        const cc_bytes = rows * c.head_dim / 2;
        const cs_bytes = rows * c.head_dim / 16;
        cs.comp_codes[i] = try takePlane(a, &sp_at, &sp_left, cc_bytes);
        cs.comp_scales[i] = try takePlane(a, &sp_at, &sp_left, cs_bytes);
        try fillSplit(e, cut, cs.comp_codes[i], 0, cc_bytes);
        try fillSplit(e, cut, cs.comp_scales[i], 0, cs_bytes);
        if (lay.idx_wk != null) {
            const ic_bytes = rows * c.index_head_dim / 2;
            const is_bytes = rows * c.index_head_dim / 32;
            cs.idx_codes[i] = try takePlane(a, &sp_at, &sp_left, ic_bytes);
            cs.idx_scales[i] = try takePlane(a, &sp_at, &sp_left, is_bytes);
            try fillSplit(e, cut, cs.idx_codes[i], 0, ic_bytes);
            try fillSplit(e, cut, cs.idx_scales[i], 127, is_bytes);
        }
        if (r > 1) {
            const rb = slots * prompt.raw_rows * c.head_dim * 4;
            cs.raw_kv[i] = try a.take(rb);
            cs.raw_score[i] = try a.take(rb);
            try prompt.fill(e, cs.raw_kv[i], 0, rb);
            try prompt.fill(e, cs.raw_score[i], 0, rb);
        }
    }
    return cs;
}

fn takePlane(a: *prompt.Arena, sp_at: *u64, sp_left: *usize, bytes: usize) !u64 {
    if (sp_left.* > 0) {
        const ptr = sp_at.*;
        sp_at.* += std.mem.alignForward(usize, bytes, 256);
        sp_left.* -= 1;
        return ptr;
    }
    return a.take(bytes);
}

fn fillSplit(e: *const prompt.Engine, cut: u64, dst: u64, value: u8, bytes: usize) !void {
    if (dst < cut and dst + bytes > cut) {
        const left: usize = @intCast(cut - dst);
        try prompt.fill(e, dst, value, left);
        return prompt.fill(e, cut, value, bytes - left);
    }
    return prompt.fill(e, dst, value, bytes);
}

/// Copies `n` rows of `row` bytes in `plane` through `ops.copyRows`, safe across the registration cut.
pub fn copySpan(ops: *const ops_mod.Ops, s: cuda.Stream, plane: u64, from: usize, to: usize, n: usize, row: usize) !void {
    if (n == 0 or from == to) return;
    const gap = if (from > to) from - to else to - from;
    if (gap >= n) return copyRange(ops, s, plane + to * row, plane + from * row, n * row);
    var done: usize = 0;
    while (done < n) : (done += gap) {
        const k = @min(gap, n - done);
        const off = if (to < from) done else n - done - k;
        try copyRange(ops, s, plane + (to + off) * row, plane + (from + off) * row, k * row);
    }
}

fn copyRange(ops: *const ops_mod.Ops, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
    var off: usize = 0;
    while (off < bytes) {
        const rem = bytes - off;
        const aligned = rem & ~@as(usize, 15);
        const step = if (aligned > 0) @min(aligned, copy_chunk) else rem;
        try ops.copyRows(s, src + off, step, dst + off, step, step, 1);
        off += step;
    }
}

const RegisterFn = *const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result;
const UnregisterFn = *const fn (?*anyopaque) callconv(.c) cuda.abi.Result;

fn driverOf(ctx: ?*anyopaque) *const cuda.Driver {
    return @ptrCast(@alignCast(ctx.?));
}

fn realOpenCard(_: ?*anyopaque, path: []const u8) Error!i32 {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= path_buf.len) return error.DisplayCardUnavailable;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const rc = linux.open(path_buf[0..path.len :0], .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return error.DisplayCardUnavailable;
    return @intCast(rc);
}

fn realCreateDumb(_: ?*anyopaque, fd: i32, bytes: usize) Error!DumbBuf {
    const height: u32 = @intCast((bytes + dumb_pitch - 1) / dumb_pitch);
    var req: CreateDumb = .{ .height = height, .width = dumb_width, .bpp = dumb_bpp };
    if (linux.errno(linux.ioctl(fd, drm_ioctl_mode_create_dumb, @intFromPtr(&req))) != .SUCCESS) {
        return error.DisplayIoctlFailed;
    }
    return .{ .handle = req.handle, .size = req.size };
}

fn realMapOffset(_: ?*anyopaque, fd: i32, handle: u32) Error!u64 {
    var req: MapDumb = .{ .handle = handle };
    if (linux.errno(linux.ioctl(fd, drm_ioctl_mode_map_dumb, @intFromPtr(&req))) != .SUCCESS) {
        return error.DisplayIoctlFailed;
    }
    return req.offset;
}

fn realReserve(_: ?*anyopaque, bytes: usize) Error!usize {
    const rc = linux.mmap(null, bytes, .{}, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .NORESERVE = true }, -1, 0);
    if (linux.errno(rc) != .SUCCESS or rc == 0) return error.DisplayMmapFailed;
    return rc;
}

fn realMapOrdinary(_: ?*anyopaque, addr: usize, bytes: usize) Error!void {
    const ptr: [*]u8 = @ptrFromInt(addr);
    const rc = linux.mmap(ptr, bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .FIXED = true }, -1, 0);
    if (linux.errno(rc) != .SUCCESS or rc != addr) return error.DisplayMmapFailed;
}

fn realMapDisplay(_: ?*anyopaque, addr: usize, bytes: usize, fd: i32, offset: u64) Error!void {
    const ptr: [*]u8 = @ptrFromInt(addr);
    const rc = linux.mmap(ptr, bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED, .FIXED = true }, fd, @bitCast(offset));
    if (linux.errno(rc) != .SUCCESS or rc != addr) return error.DisplayMmapFailed;
}

fn realRegister(ctx: ?*anyopaque, addr: usize, bytes: usize, io_mem: bool) Error!void {
    const d = driverOf(ctx);
    const reg = @constCast(&d.lib).lookup(RegisterFn, "cuMemHostRegister_v2") orelse return error.DisplayRegisterFailed;
    const flags: c_uint = host_register_devicemap | (if (io_mem) host_register_iomemory else 0);
    if (reg(@ptrFromInt(addr), bytes, flags) != cuda.abi.success) return error.DisplayRegisterFailed;
}

fn realDevicePointer(ctx: ?*anyopaque, addr: usize) Error!u64 {
    const d = driverOf(ctx);
    var dev: u64 = 0;
    if (d.api.cuMemHostGetDevicePointer_v2(&dev, @ptrFromInt(addr), 0) != cuda.abi.success or dev == 0) return error.DisplayRegisterFailed;
    return dev;
}

fn realUnregister(ctx: ?*anyopaque, addr: usize) void {
    const d = driverOf(ctx);
    if (@constCast(&d.lib).lookup(UnregisterFn, "cuMemHostUnregister")) |unreg| {
        _ = unreg(@ptrFromInt(addr));
    }
}

fn realUnmap(_: ?*anyopaque, addr: usize, bytes: usize) void {
    const ptr: [*]const u8 = @ptrFromInt(addr);
    _ = linux.munmap(ptr, bytes);
}

fn realCloseGem(_: ?*anyopaque, fd: i32, handle: u32) void {
    var req: GemClose = .{ .handle = handle };
    _ = linux.ioctl(fd, drm_ioctl_gem_close, @intFromPtr(&req));
}

fn realCloseFd(_: ?*anyopaque, fd: i32) void {
    _ = linux.close(fd);
}
