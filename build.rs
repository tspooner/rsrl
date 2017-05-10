extern crate gcc;


fn main() {
    gcc::Config::new()
        .file("extern/tiles.c")
        .include("extern")
        .compile("libtiles.a");
}
