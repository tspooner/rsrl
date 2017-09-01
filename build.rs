extern crate gcc;


fn main() {
    gcc::Build::new()
        .file("extern/tiles.c")
        .include("extern")
        .compile("libtiles.a");
}
