# Contributing
---

To contribute to `Phoebus`, feel free to submit a pull request or open
an issue.

1.  Create a new fork of `main` where your changes can be made.
2.  After completing, create a pull request, describe the final approach
    and ensure the branch has no conflicts with current Regression Tests
    and Formatting Checks.
3.  Assign external reviewers (a minimum of 1, one of which should be a
    Maintainer, and which cannot be the contributor). Provide concise
    description of changes.
4.  Once comments/feedback is addressed, the PR will be merged into the
    main branch and changes will be added to list of changes in the next
    release.
5.  At present, Releases (with a git version tag) for the `main` branch
    of `Phoebus` will occur at a 6 to 12 month cadence or following
    implementation of a major enhancement or capability to the code
    base.

*Maintainers* of `Phoebus` will follow the same process for
contributions as above except for the following differences: they may
create branches directly within `Phoebus`, they will hold repository
admin privileges, and actively participate in the technical review of
contributions to `Phoebus`.

## Pull request protocol

When submitting a pull request, there is a default template that is
automatically populated. The pull request should sufficiently summarize
all changes. As necessary, tests should be added for new features of
bugs fixed.

Before a pull request will be merged, the code should be formatted. We
use clang-format for this, pinned to version 12. The script
`scripts/bash/format.sh` will apply `clang-format` to C++ source files
in the repository as well as `black` to python files, if available. The
script takes two CLI arguments that may be useful, `CFM`, which can be
set to the path for your clang-format binary, and `VERBOSE`, which if
set to `1` adds useful output. For example:

``` bash
CFM=clang-format-12 VERBOSE=1 ./scripts/bash/format.sh
```

## Test Suite

Several sets of tests are triggered on a pull request: a static format
check, a docs buld, and a suite of unit and regression tests. These are
run through github's CPU infrastructure. These tests help ensure that
modifications to the code do not break existing capabilities and ensure
a consistent code style.

## Becoming a Maintainer

For the purpose of our development model, we label *Contributors* or
*Maintainers* of `Phoebus`. Below we describe these labels and the
process for contributing to `Phoebus`.

We welcome contributions from scientists from a large variety of
relativistic astrophysics. New users are welcome to contributions to
`Phoebus` via the process above. Contributors with 6 merged
pull requests into the main branch (over a minimum of 6 months) will be
eligible to join as a *Maintainer* of `Phoebus` with additional
repository access and roles. However, final approval of *Maintainer*
status will require a approval by vote by existing
*Maintainers*, a necessary step to ensure the safety and integrity of
the code base for all users of `Phoebus`.

The *Maintainers* of `Phoebus` consist of the original developers of the
code and those that have a demonstrated history in the development of
`Phoebus` prior to the implementation of the *Development Model*
described here. Maintainers hold admin access, serve as reviewers for
pull requests, and are in charge of the maintaining, deployment, and
improvement of efforts towards: regression testing, documentation,
science test cases (gold standards), and continuous integration.

## List of Current Maintainers of Phoebus
---
| Name     | Handle       | Team       |
|----------|--------------|------------|
| Brandon Barker | [@AstroBarker](https://www.github.com/AstroBarker) | Los Alamos National Lab |
| Josh Dolence | [@jdolence](https://www.github.com/jdolence) | Los Alamos National Lab |
| Carl Fields | [@carlnotsagan](https://www.github.com/carlnotsagan) | University of Arizona |
| Mariam Gogilashvili | [@mari2895](https://www.github.com/mari2895) | Niels Bohr Institute |
| Jonah Miller | [@Yurlungur](https://www.github.com/Yurlungur) | Los Alamos National Lab |
| Jeremiah Murphy | [@curiousmiah](https://www.github.com/curiousmiah) | Florida State University |
| Ben Ryan | [@brryan](https://www.github.com/brryan) | Los Alamos National Lab |
