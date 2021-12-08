{
  description = "Generate continuous time stationary stochastic processes from a given auto correlation function.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    poetry2nix.url = "github:nix-community/poetry2nix";
    flake-utils.url = "github:numtide/flake-utils";
    fcSpline.url = "github:vale981/fcSpline";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix, fcSpline }:
    (flake-utils.lib.eachDefaultSystem (system:
      let
        name = "stocproc";
        overlay = nixpkgs.lib.composeManyExtensions [
          poetry2nix.overlay

          (final: prev:
            let overrides = prev.poetry2nix.overrides.withDefaults
              (self: super: { });
            in
            {
              ${name} = (prev.poetry2nix.mkPoetryApplication {
                projectDir = ./.;
                preferWheels = true;
                overrides = overrides;
              });

              "${name}Shell" = (prev.poetry2nix.mkPoetryEnv {
                projectDir = ./.;
                overrides = overrides;
                preferWheels = true;
                editablePackageSources = {
                  ${name} = ./${name};
                };
              });
            })

        ];
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ overlay ];
          config.allowUnfree = true;
        };
      in
      rec {
        packages = {
          ${name} = pkgs.${name};
        };

        defaultPackage = packages.${name};
        devShell = pkgs."${name}Shell".env.overrideAttrs (oldAttrs: {
          buildInputs = [ pkgs.poetry pkgs.black pkgs.pyright ];
        });
      }));
}
