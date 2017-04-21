extern crate gcc;


fn main() {
    gcc::Config::new()
        .cpp(true)
        .file("extern/tiles.cpp")
        .include("extern")
        .compile("libtiles.a");
}
