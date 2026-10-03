struct Pair { a: u8, b: u4 }
proc twostate {
    in_ch: chan<Pair> in;
    out_ch: chan<u8> out;
    config(in_ch: chan<Pair> in, out_ch: chan<u8> out) { (in_ch, out_ch) }
    init { (u8:1, u8[4]:[0, ...]) }
    next(st: (u8, u8[4])) {
        let (c, arr) = st;
        let (tok, x) = recv(join(), in_ch);
        assert!(x.a != u8:255, "never_ff");
        let arr2 = update(arr, x.b, x.a);
        let tok = send(tok, out_ch, c);
        (c + u8:1, arr2)
    }
}
proc nostate {
    in_ch: chan<u8> in;
    out_ch: chan<u8> out;
    config(in_ch: chan<u8> in, out_ch: chan<u8> out) { (in_ch, out_ch) }
    init { () }
    next(st: ()) {
        let (tok, x) = recv(join(), in_ch);
        let tok = send(tok, out_ch, x ^ u8:0x5a);
    }
}
fn g(a: u8, b: s8) -> (u8, s8, u16, u4, bool) { (a >> 2, b >> 1, a ++ (b as u8), a[2:6], and_reduce(a)) }
