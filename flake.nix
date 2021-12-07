{
  description = "Generate continuous time stationary stochastic processes from a given auto correlation function.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    mach-nix.url = "github:DavHau/mach-nix";
    flake-utils.url = "github:numtide/flake-utils";
    fcSpline.url = "github:vale981/fcSpline";
  };

   outputs = { self, nixpkgs, flake-utils, mach-nix, fcSpline }:
     let
       python = "python39";
       pypiDataRev = "master";
       pypiDataSha256 = "041rpjrwwa43hap167jy8blnxvpvbfil0ail4y4mar1q5f0q57xx";
       devShell = pkgs:
         pkgs.mkShell {
           buildInputs = [
             (pkgs.${python}.withPackages
               (ps: with ps; [ black mypy ]))
             pkgs.nodePackages.pyright
           ];
         };

     in flake-utils.lib.eachSystem ["x86_64-linux"] (system:
       let
         pkgs = nixpkgs.legacyPackages.${system};
         mach-nix-wrapper = import mach-nix { inherit pkgs python pypiDataRev pypiDataSha256; };

         fcSplinePkg = fcSpline.defaultPackage.${system};

         stocproc = (mach-nix-wrapper.buildPythonPackage rec {
           src = ./.;
           requirements = ''
            numpy>=1.20
            scipy>=1.6
            mpmath>=1.2.0
            cython
            '';
           pname = "stocproc";
           version = "1.0.1";
           propagatedBuildInputs = [fcSplinePkg];
         });

         pythonShell = mach-nix-wrapper.mkPythonShell {
           requirements = builtins.readFile ./requirements.txt;
         };

         mergeEnvs = envs:
           pkgs.mkShell (builtins.foldl' (a: v: {
             buildInputs = a.buildInputs ++ v.buildInputs;
             nativeBuildInputs = a.nativeBuildInputs ++ v.nativeBuildInputs;
           }) (pkgs.mkShell { }) envs);

       in {
         devShell = mergeEnvs [ (devShell pkgs) pythonShell ];
         defaultPackage = stocproc;
         packages.stocproc = stocproc;
       });
}
