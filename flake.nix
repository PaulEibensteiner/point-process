{
  description = "Development environment for point-process";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs {
          inherit system;
        }));
    in {
      devShells = forEachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            gcc
            gnumake
            pkg-config
            expat
          ];

          shellHook = ''
            export UV_PROJECT_ENVIRONMENT="''${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.expat ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            uv sync
          '';
        };
      });
    };
}
