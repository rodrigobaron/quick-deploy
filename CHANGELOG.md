# CHANGELOG

This file follows [semantic versioning 2.0.0](https://semver.org/). Given a version number MAJOR.MINOR.PATCH, increment
the:

- **MAJOR** version when you make incompatible API changes,
- **MINOR** version when you add functionality in a backwards compatible manner, and
- **PATCH** version when you make backwards compatible bug fixes.

As a heuristic:

- if you fix a bug, increment the PATCH
- if you add a feature (add keyword arguments with default values, add a new object, a new mechanism for parameter setup
  that is backwards compatible, etc.), increment the MINOR version
- if you introduce a breaking change (removing arguments, removing objects, restructuring code such that it affects
  imports, etc.), increment the MAJOR version

The general format is:

```

# VERSION - DATE (dd/mm/yyyy)
### Added
- A to B
### Changed
- B to C
### Removed
- C from D

```
# 0.2.1 - DATE (08/01/2022)

### Changed
- Fix minimal version imports 

# 0.2.0 - DATE (08/01/2022)

### Added
- Code coverage
- Tensorflow command

### Changed
- Code documentation

# 0.1.1 - DATE (02/01/2022)

### Changed

- rename package

# 0.1.0 - DATE (02/01/2022)

### Added

- initial release of quick-deploy
