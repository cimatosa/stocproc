{
  description = "Generate continuous time stationary stochastic processes from a given auto correlation function.";


  inputs = {
    utils.url = "github:vale981/hiro-flake-utils";
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = { self, utils, nixpkgs, ... }:
    (utils.lib.poetry2nixWrapper nixpkgs {
      name = "stocproc";
      poetryArgs = {
        projectDir = ./.;
      };
    });
}
