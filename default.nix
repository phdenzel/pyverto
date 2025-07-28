{
  pkgs ? import <nixpkgs> {},
  src ? ./.,
  # subdir ? "",
}:
let 
  pythonPackage = pkgs.python312Packages.buildPythonApplication {
    pname = "pyverto";
    version = "0.1.10";
    format = "pyproject";
    build-system = with pkgs.python312Packages; [hatchling];
    propagatedBuildInputs = with pkgs.python312Packages; [
      gitpython
    ];
    src = src;
    # doCheck = false;
    meta = {
      description = "A python package version management tool.";
      meta.description.license = pkgs.lib.licenses.mit;
    };
  };
in

pythonPackage

