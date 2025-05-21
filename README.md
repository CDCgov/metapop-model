# A stochastic SEIR metapopulation model in python

⚠️ This is a work in progress

## Model features

* Stochastic SEIRV model
* Flexible numbers of groups (e.g., age classes or connected populations) that is changeable using the config
* Intervention strategies common for measles: pre-introduction vaccination, active vaccination, and isolation of infectious populations
* An interactive widget built in streamlit

## Getting started

* Enable [poetry](https://python-poetry.org/) with `poetry install` and then start a poetry environment by activating the virtual environment: `source $(poetry env info --path)/bin/activate`
* Run the example in `scripts/connectivity` exploring the impact of changing connectivity patterns and initial vaccine coverage with `python scripts/connectivity/simulate.py`. This will produce output in the `output` folder. Visualization and summary statistic tables for these results can be made with `Rscript scripts/connectivity/make_plots.R`.
* We also provide examples exploring other features in the model. The subfolder `scripts/interventions` contains an example exploring the impact of active vaccination and isolation in 3 connection populations. The subfolder `scripts/onepop` contains an example exploring these interventions in a single population.


## Running with a flexible number of compartments

* The `metapop` package can currently support the modeling of infectious disease transmission in 1 or 3 groups. Groups represent populations with different epidemiologically relevant characteristics such as age classes or connected populations. To model a single population or 3 connected populations, you can change the number of groups by modifying the config file (`scripts/config.yaml`).
* For example, if we wanted 3 groups, we could specify `n_groups: 3` and then modify other parameters defining conditions for the populations: `popsizes: [300, 200, 100]` for population sizes, `I0: [0, 5, 2]` to specify the number of initial infected people in each population, `k_i: [10, 15, 10]` for the average degree per person in each population, etc. The parameters `k_g1`, `k_g2`, `k_21` define connectivity rates between groups: `k_ij` is the number of contacts the average person in group j makes with others in group i. These parameters define the contact matrix between populations (constructed through the method `get_per_capita_contact_matrix`).
* Similarly for a single population, we can build the model by setting `n_groups: 1`, `popsizes: [5000]` for the population size, `I0: 5`, `k_i: [10]`, etc. In this case, the contact matrix is equivalent to `k_i`.
* As a work in progress, we are working to add functionality that will allow users to give a contact matrix of their choice and model a flexible number of groups that mapped to the contact matrix.


## Local app
You can run the app locally using Streamlit with `make run_app`. To run the advanced app with 3 connected populations, use `make run_advanced_app`.


## Project Admin

* Paige Miller, yub1@cdc.gov (CDC/IOD/ORR/CFA)
* Theresa Sheets, utg8@cdc.gov (CDC/IOD/ORR/CFA)
* Will Koval, ad71@cdc.gov (CDC/IOD/ORR/CFA)
* Beau Bruce, lue7@cdc.gov (CDC/IOD/ORR/CFA)
* Dina Mistry, uqx8@cdc.gov (CDC/IOD/ORR/CFA)

## General Disclaimer
This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).
