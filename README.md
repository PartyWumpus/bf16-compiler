# bf16-compiler
A quick experiment where I rewrote the [bf16](https://github.com/p2r3/bf16) project (a visual brainfuck runtime for interactive games) in rust and as a compiler instead of an interpreter.

Currently just reads in whatever you put in `file.b`, because it isn't worth the effort to make a nice interface.

The design is a tad jank, as it was mostly hacked together by a very tired wumpus.

Requires llvm 18.1, SDL2 and ofc cargo. If you have nix then `nix develop` will get the env set up, and then just `cargo run --release` to build and run.

The project is GPL3 because the original was GPL3 and some of the code here was directly translated from the original C.
