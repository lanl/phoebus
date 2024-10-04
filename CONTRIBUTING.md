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

# Pull request protocol

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

In order for a pull request to merge, we require:

-   Provide a thorough summary of changes, especially if they are
    breaking changes
-   Obey style guidleines (format with `clang-format` and pass the
    necessary test)
-   Pass the existing test suite
-   Have at least one approval from a Maintainer
-   If Applicable:
    -   Write new tests for new features or bugs
    -   Include or update documentation in `doc/`

---

# Test Suite

Several sets of tests are triggered on a pull request: a static format
check, a docs buld, and a suite of unit and regression tests. These are
run through github\'s CPU infrastructure. These tests help ensure that
modifications to the code do not break existing capabilities and ensure
a consistent code style.

## Adding Tests

There are two primary categories of tests written in `Phoebus`: unit
tests and regression tests.

### Unit

Unit tests live in `tst/unit/`. They are implemented using the
[Catch2](https://github.com/catchorg/Catch2) unit testing framework.
They are integrated with `cmake` and can be run, when enabled, with
`ctest`. There are a few necessary `cmake` configurations to beuild
tests:

+------------------------------+---------+------------------------------------------+
| Option                       | Default | Description                              |
+==============================+=========+==========================================+
| PHOEBUS\_ENABLE\_UNIT\_TESTS | OFF     | Enables Catch2 unit tests                |
+------------------------------+---------+------------------------------------------+
| PHOEBUS\_ENABLE\_DOWNLOADS   | OFF     | Enables unit tests using tabulated EOS   |
+------------------------------+---------+------------------------------------------+

### Regression

Regression tests run existing simulations and test against saved output
in order to verify sustained capabilities. They are implemented in
Python in `test/regression/`. To run the tests you will need a Python
environment with at least `numpy` and `h5py`. Tests can be ran manually
as, e.g.,

``` bash
python linear_modes.py
```

This will build Phoebus locally in `phoebus/tst/regression/build` and
run it in `phoebus/tst/regression/run`. Ensure that these directories do
not already exist. Each script `test.py` has a correspodning \"gold
file\" `test.gold`. The gold files contain the gold standard data that
the output of the regression test is compared against. To generate new
gold data, for example if a change is implemented that changes the
behavior of a test (not erroneously) or a new test is created, run the
test script with the `--upgold` option. This will create or update the
corresponding `.gold` file. To add a new test:

1.  Create a new test script.
    -   Update the `modified_inputs` struct to change any input deck
        options
    -   Set the `variables` list to contain the quantities to test
        against
2.  Run the script with the `--upgold` option
3.  Commit the test script and gold file
4.  Update the CI to include the new test
    (`phoebus/.github/workflows/tests.yml`)

---
See the [docs](https://lanl.github.io/phoebus) for further information
about contributing to `Phoebus`.
---

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
