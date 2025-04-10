# A stochastic SEIR metapopulation model in python

⚠️ This is a work in progress

## Model features

* Stochastic SEIRV model
* Flexible numbers of groups (e.g., age classes or connected populations) that is changeable using the config
* Intervention strategies common for measles: pre-introduction vaccination, reactive vaccination, isolation of infectious populations
* Basic example of how to do a calibration
* An interactive widget built in streamlit

## Getting started

* Enable [poetry](https://python-poetry.org/) with `poetry install` and then start a poetry environment by activating the virtual environment: `source $(poetry env info --path)/bin/activate`
* Run the example with `python scripts/simulate.py`, this will produce output in output folder.  Plots for this can be made with `python scripts/make_plots.py`
* Run basic calibration in `run_abc.py`, note that `make_plots.qmd` created and saved a fake dataset to use in `run_abc.csv` (included in repo to make it easier to get started here), also the calibration is for a value of beta matrix (with a uniform prior from 0.1, 0.5), can easily change code to calibrate another parameter
* Make plots and visualizations with `make_plots.qmd`
* Data for `scripts/make_plots.py` are in Sharepoint > Crosscutting > response data folder > ABM inputs

## Running with flexible number of compartments

* Instead of two groups (e.g., age classes), you can easily change the number of groups by changing the config file (`scripts/config.yaml`)
* For example, if we wanted 6 groups, you could specify   `n_groups: 6`  and then change `N: [100, 200, 300, 400, 500, 600]` (i.e., a list of 6 population sizes), same for `I0: [1, 0, 0, 0, 0, 0]`, and then the beta_matrix would have to be a matrix 6x6. In the future, it would be good to have funcitonality to read beta matrix from a csv.

## Local app
You can run the app locally using Streamlit with `streamlit run metapop/launch_app.py`


## Project Admin

Paige Miller, yub1@cdc.gov (CDC/IOD/ORR/CFA)

Theresa Sheets, utg8@cdc.gov (CDC/IOD/ORR/CFA)

Dina Mistry, uqx8@cdc.gov (CDC/IOD/ORR/CFA)

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
