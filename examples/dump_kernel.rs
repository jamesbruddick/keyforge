fn main() {
    let scope = keyforge::scan::derive::Scope::default();
    let layout = keyforge::gpu::Layout::new(&scope, 1);
    print!(
        "{}",
        keyforge::gpu::source::assemble(
            &layout,
            &keyforge::vuln::mt19937::MilkSad,
            keyforge::gpu::source::Dialect::Metal
        )
    );
}
