{
	inputs = {
		flake-utils.url = "github:numtide/flake-utils";
		nixpkgs.url = "github:nixos/nixpkgs";

		rust-overlay.url = "github:oxalica/rust-overlay";
	};

	outputs = {
		self,
		nixpkgs,
		flake-utils,
		rust-overlay,
		...
	}:
		flake-utils.lib.eachDefaultSystem (system:
			let
				pkgs = import nixpkgs {
					inherit system;
					overlays = [ (import rust-overlay) ];
				};
			in with pkgs; rec {
				devShell = mkShell rec {
					packages = [
						rust-bin.stable.latest.default
            #(rust-bin.selectLatestNightlyWith (toolchain:
            #  toolchain.default.override {
            #    extensions = [ "miri" "rust-src" ];
            #}))
            llvm_18
            libffi
            libxml2
            SDL2
					];
				};
			});
}
