# pyverto

[![PyPI - Version](https://img.shields.io/pypi/v/pyverto.svg)](https://pypi.org/project/pyverto)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyverto.svg)](https://pypi.org/project/pyverto)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install pyverto
```

## Usage

```console
Usage:
  pyverto [command] [--commit] [-h]

Commands:
  version    Show current version
  release    Remove any pre-release/dev/post suffix (finalize version)
  major      Increment the major version
  minor      Increment the minor version
  micro      Increment the micro (patch) version
  alpha      Convert to or increment alpha pre-release
  beta       Convert to or increment beta pre-release
  pre        Convert to or increment rc (release candidate)
  rev        Increment post-release (+postN)
  dev        Convert to or increment dev release (-devN)
```
Examples:
  - `pyverto minor`: 0.0.1 → 0.1.0
  - `pyverto dev`: 0.1.0 → 0.1.0-dev0
  - `pyverto pre --commit`: 0.1.0-dev0 → 0.1.0-rc0
  
  
## Usage in GitHub Actions

```yaml
- uses: phdenzel/pyverto@v0.1.7
  with:
      bump-type: "minor"
```


## License

`pyverto` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
