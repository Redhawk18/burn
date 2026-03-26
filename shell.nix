{
  pkgs ? import <nixpkgs> {
    overlays = [
      (import (fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))
    ];
  },
}:

pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.pkg-config
    pkgs.clang
    pkgs.cmake
    (pkgs.rust-bin.nightly.latest.default.override {
      extensions = [
        "rust-src"
        "rust-analyzer"
      ];
    })
  ];

  buildInputs = [
    pkgs.openssl
    pkgs.libiconv
    pkgs.gnustep-libobjc
    # Common Burn/WGPU/Winit dependencies
    pkgs.xorg.libX11
    pkgs.xorg.libXcursor
    pkgs.xorg.libXrandr
    pkgs.xorg.libXi
    pkgs.libxkbcommon
    pkgs.vulkan-loader
  ];

  shellHook = ''
    export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig"

    # Force the C compiler to find the libobjc headers
    export C_INCLUDE_PATH="${pkgs.gnustep-libobjc}/include"
    export CPLUS_INCLUDE_PATH="${pkgs.gnustep-libobjc}/include"

    # Help Rust's cc-rs crate find the right tools
    export CC="clang"
    export CXX="clang++"

    # Runtime library paths
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.vulkan-loader}/lib:${pkgs.libxkbcommon}/lib:${pkgs.gnustep-libobjc}/lib"

    echo "✅ Nix shell loaded: Rust Nightly & Libobjc detected"
  '';
}
