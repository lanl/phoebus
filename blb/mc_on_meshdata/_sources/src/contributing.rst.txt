.. _singularity-eos: https://lanl.github.io/singularity-eos
.. _Catch2: https://github.com/catchorg/Catch2

Contributing
=============================

To contribute to ``Phoebus``, feel free to submit a pull
request or open an issue.

1. Create a new fork of ``main`` where your changes can be made.
2. After completing, create a pull request, describe the final approach
   and ensure the branch has no conflicts with current Regression Tests
   and Formatting Checks.
3. Assign external reviewers (a minimum of 1, one of which should be a
   Maintainer, and which cannot be the contributor). Provide concise
   description of changes.
4. Once comments/feedback is addressed, the PR will be merged into the
   main branch and changes will be added to list of changes in the next
   release.
5. At present, Releases (with a git version tag) for the ``main`` branch
   of ``Phoebus`` will occur at a 6 to 12 month cadence or following
   implementation of a major enhancement or capability to the code base.

*Maintainers* of ``Phoebus`` will follow the same process for
contributions as above except for the following differences: they may
create branches directly within ``Phoebus``, they will hold repository
admin privileges, and actively participate in the technical review of
contributions to ``Phoebus``.


Pull request protocol
----------------------

When submitting a pull request, there is a default template that is automatically
populated. The pull request should sufficiently summarize all changes.
As necessary, tests should be added for new features of bugs fixed.

Before a pull request will be merged, the code should be formatted. We
use clang-format for this, pinned to version 12.
The script ``scripts/bash/format.sh`` will apply ``clang-format``
to C++ source files in the repository as well as ``black`` to python files, if available.
The script takes two CLI arguments
that may be useful, ``CFM``, which can be set to the path for your
clang-format binary, and ``VERBOSE``, which if set to ``1`` adds
useful output. For example:

.. code-block:: bash

    CFM=clang-format-12 VERBOSE=1 ./scripts/bash/format.sh

In order for a pull request to merge, we require:

- Provide a thorough summary of changes, especially if they are breaking changes
- Obey style guidleines (format with ``clang-format`` and pass the necessary test)
- Pass the existing test suite
- Have at least one approval from a Maintainer
- If applicable:

  - Write new tests for new features or bugs
  - Include or update documentation in ``doc/``

Test Suite
----------

Several sets of tests are triggered on a pull request: a static format
check, a docs buld, and a suite of unit and regression tests.
These are run through github's CPU infrastructure. These tests
help ensure that modifications to the code do not break existing capabilities
and ensure a consistent code style.

Adding Tests
````````````

There are two primary categories of tests written in ``Phoebus``:
unit tests and regression tests.

Unit
^^^^

Unit tests live in ``tst/unit/``. They are implemented using the
`Catch2`_ unit testing framework. They are integrated with ``cmake``
and can be run, when enabled, with ``ctest``. There are a few necessary ``cmake``
configurations to beuild tests:

========================== ========= =======================================
    Option                  Default   Description
========================== ========= =======================================
PHOEBUS_ENABLE_UNIT_TESTS    OFF      Enables Catch2 unit tests
PHOEBUS_ENABLE_DOWNLOADS     OFF      Enables unit tests using tabulated EOS
========================== ========= =======================================

Regression
^^^^^^^^^^
Regression tests run existing simulations and test against saved output
in order to verify sustained capabilities.
They are implemented in Python in
``test/regression/``. To run the tests you will need a Python environment with
at least ``numpy`` and ``h5py``. Tests can be ran manually as, e.g.,

.. code-block:: bash

   python linear_modes.py

This will build Phoebus locally in ``phoebus/tst/regression/build`` and run it in
``phoebus/tst/regression/run``. Ensure that these directories do not already exist.
Each script ``test.py`` has a correspodning "gold file" ``test.gold``.
The gold files contain the gold standard data that the output of the regression test
is compared against. To generate new gold data, for example if a change is implemented
that changes the behavior of a test (not erroneously) or a new test is created, run the test
script with the ``--upgold`` option. This will create or update the corresponding ``.gold`` file.
To add a new test:

1. Create a new test script.

   - Update the ``modified_inputs`` struct to change any input deck options
   - Set the ``variables`` list to contain the quantities to test against
2. Run the script with the ``--upgold`` option
3. Commit the test script and gold file
4. Update the CI to include the new test (``phoebus/.github/workflows/tests.yml``)


Expectations for code review
-----------------------------

Much of what follows is adapted from `singularity-eos`_.

From the perspective of the contributor
````````````````````````````````````````

Code review is an integral part of the development process
for ``Phoebus``. You can expect at least one, perhaps many,
core developers to read your code and offer suggestions.
You should treat this much like scientific or academic peer review.
You should listen to suggestions but also feel entitled to push back
if you believe the suggestions or comments are incorrect or
are requesting too much effort.

Reviewers may offer conflicting advice, if this is the case, it's an
opportunity to open a discussion and communally arrive at a good
approach. You should feel empowered to argue for which of the
conflicting solutions you prefer or to suggest a compromise. If you
don't feel strongly, that's fine too, but it's best to say so to keep
the lines of communication open.

Big contributions may be difficult to review in one piece and you may
be requested to split your pull request into two or more separate
contributions. You may also receive many "nitpicky" comments about
code style or structure. These comments help keep a broad codebase,
with many contributors uniform in style and maintainable with
consistent expectations accross the code base. While there is no
formal style guide for now, the regular contributors have a sense for
the broad style of the project. You should take these stylistic and
"nitpicky" suggestions seriously, but you should also feel free to
push back.

As with any creative endeavor, we put a lot of ourselves into our
code. It can be painful to receive criticism on your contribution and
easy to take it personally. While you should resist the urge to take
offense, it is also partly code reviewer's responsiblity to create a
constructive environment, as discussed below.

Expectations of code reviewers
````````````````````````````````

A good code review builds a contribution up, rather than tearing it
down. Here are a few rules to keep code reviews constructive and
congenial:

* You should take the time needed to review a contribution and offer
  meaningful advice. Unless a contribution is very small, limit
  the times you simply click "approve" with a "looks good to me."

* You should keep your comments constructive. For example, rather than
  saying "this pattern is bad," try saying "at this point, you may
  want to try this other pattern."

* Avoid language that can be misconstrued, even if it's common
  notation in the commnunity. For example, avoid phrases like "code
  smell."

* Explain why you make a suggestion. In addition to saying "try X
  instead of Y" explain why you like pattern X more than pattern Y.

* A contributor may push back on your suggestion. Be open to the
  possibility that you're either asking too much or are incorrect in
  this instance. Code review is an opportunity for everyone to learn.

* Don't just highlight what you don't like. Also highlight the parts
  of the pull request you do like and thank the contributor for their
  effort.

General principle for everyone
```````````````````````````````

It's hard to convey tone in text correspondance. Try to read what
others write favorably and try to write in such a way that your tone
can't be mis-interpreted as malicious.

A Large Ecosystem
------------------------

``Phoebus`` depends on several other open-source, Los Alamos
maintained, projects. In particular, ``Parthenon``, ``singularity-eos``,
``singularity-opac``, and ``spiner``.
If you have issues with these projects, ideally
submit issues on the relevant github pages. However, if you can't
figure out where an issue belongs, no big deal. Submit where you can
and we'll engage with you to figure out how to proceed.

Becoming a Contributor
----------------------

For the purpose of our development model, we label *Contributors* or
*Maintainers* of ``Phoebus``. Below we describe these labels and the
process for contributing to ``Phoebus``.

We welcome contributions from scientists from a large variety of
relativistic astrophysics. New users are welcome to contributions to
``Phoebus`` via the *Contributors* process. Contributors with 6 merged
pull requests into the main branch (over a minimum of 6 months) will
be eligible to join as a *Maintainer* of ``Phoebus`` with additional
repository access and roles. However, final approval of *Maintainer*
status will require a approval by vote by existing
*Maintainers*, a necessary step to ensure the safety and integrity of
the code base for all users of ``Phoebus``.

The *Maintainers* of ``Phoebus`` consist of the original developers of
the code and those that have a demonstrated history in the development
of ``Phoebus`` prior to the implementation of the *Development Model*
described here. Maintainers hold admin access, serve as
reviewers for pull requests, and are in charge of the maintaining,
deployment, and improvement of efforts towards: regression testing,
documentation, science test cases (gold standards), and continuous
integration.

Maintainers are excepted to make a good faith effort to adhere to
:ref:`these suggestions <mou>` in order to maintain a supportive and
productive environment.

List of Current Maintainers of Phoebus
------------------------------------------

+-------------------+--------------------------------+-----------------------+
| Name              | Handle                         | Team                  |
+===================+================================+=======================+
| Brandon Barker    |                                | Los Alamos National   |
|                   | `@AstroBarker <https://www.g   | Lab                   |
|                   | ithub.com/AstroBarker>`__      |                       |
+-------------------+--------------------------------+-----------------------+
| Josh Dolence      | `@jdolence <https://ww         | Los Alamos National   |
|                   | w.github.com/jdolence>`__      | Lab                   |
+-------------------+--------------------------------+-----------------------+
| Carl Fields       |                                | University of Arizona |
|                   | `@carlnotsagan <https://www.gi |                       |
|                   | thub.com/carlnotsagan>`__      |                       |
+-------------------+--------------------------------+-----------------------+
| Mariam            | `@mari2895 <https://ww         | Niels Bohr Institute  |
| Gogilashvili      | w.github.com/mari2895>`__      |                       |
+-------------------+--------------------------------+-----------------------+
| Jonah Miller      | `@Yurlungur <https://www       | Los Alamos National   |
|                   | .github.com/Yurlungur>`__      | Lab                   |
+-------------------+--------------------------------+-----------------------+
| Jeremiah Murphy   |                                | Florida State         |
|                   | `@curiousmiah <https://www.g   | University            |
|                   | ithub.com/curiousmiah>`__      |                       |
+-------------------+--------------------------------+-----------------------+
| Ben Ryan          | `@brryan <https://             | Los Alamos National   |
|                   | www.github.com/brryan>`__      | Lab                   |
+-------------------+--------------------------------+-----------------------+
