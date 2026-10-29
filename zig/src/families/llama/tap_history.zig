pub const History = struct {
    const Span = struct { begin: usize, end: usize };
    span: ?Span = null,

    pub fn covers(h: History, begin: usize, end: usize) bool {
        if (end <= begin) return true;
        const s = h.span orelse return false;
        return s.begin <= begin and end <= s.end;
    }

    pub fn truncate(h: *History, end: usize) void {
        const s = h.span orelse return;
        h.span = if (end <= s.begin) null else .{ .begin = s.begin, .end = @min(s.end, end) };
    }

    pub fn append(h: *History, position: usize, window: usize) void {
        const begin = if (h.span) |s| if (s.end == position) s.begin else position else position;
        h.span = .{ .begin = @max(begin, (position + 1) -| window), .end = position + 1 };
    }
};
