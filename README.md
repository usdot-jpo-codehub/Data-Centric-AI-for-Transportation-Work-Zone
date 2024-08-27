# codehub-readme-template

## Summary of what a README file is:

A README file contains important information about the associated source code, files, dependencies and related details that make up an application. In order to leverage applications and their associated source code, members of the open source community rely on clear and concise READMEs.

## Purpose of this template

The purpose of this README template is to provide project teams an easy template to copy-paste into their own README file for source code funded, either fully or partially, by the United States Department of Transportation (U.S. DOT) Intelligent Transportation Systems Joint Program Office (ITS JPO). This template contains elements for following the [ITS JPO Source Code Guidelines](https://its.dot.gov/code/#/source-code-guidelines) and README best practices.

The goal is for project READMEs to be compliant, clear, concise, and informative for source code users. Please note that examples of the [CONTRIBUTING.md](https://github.com/usdot-jpo-codehub/codehub-readme-template/blob/master/Contributing.MD) and [LICENSE.md](https://github.com/usdot-jpo-codehub/codehub-readme-template/blob/master/LICENSE) files required by the ITS JPO Source Code Guidelines are nested within this template. See the License and Contributions sections of this template for more details. **Please feel free to copy and paste this template into your own repository's README space.**

For a real-world example of an ITS JPO-funded project using this template, see the [STOL-AMS/TO-22-Improved-CACC](https://github.com/STOL-AMS/TO-22-Improved-CACC) repository.

# README Outline:
* Project Description
* Prerequisites
* Usage
	* Building
	* Testing
	* Execution
* Additional Notes
* Version History and Retention
* License
* Contributions
* Contact Information
* Acknowledgements

# Project Description

*Insert a description of your project, including the project's title, purpose and goals of the project, purpose of the source code, how the source code relates to the overall goals of the project, whether this source code relates to other source code in the project, and length of the project.*

Example:

README Template

This project is a README template for users to copy-paste into their project's README file and fill out with their project's information in order to provide useful and relevant information to users of a repository. It is accompanied by a LICENSE and CONTRIBUTING file, linked to in the License and Contributions sections below. This project will continue indefinitely as long as the ITS JPO Source Code Guidelines are in effect.

# Prerequisites

*Detail what actions users need to take before they can stand up the project, including instructions for different environments users might have. This might include instructions and examples for installing additional software.*

Example:

Requires:
- Java 8 (or higher)
- Maven 3.5.4
- Docker

# Usage
*Provide users with detailed instructions for how to use your software. The specifics of this section will vary between projects, but should adhere to the following minimum outline:*

## Building
*Specifics for how to build/compile this code should be outlined here. If your code does not require any type of build/compilation, specify that here.*

Example: 

Step 1: Build Docker image:
```
docker build myproject
```

Step 2: Run Docker image:
```
docker run myproject
```
## Testing
*Specifics for how to run any test cases your project may have. If your test cases are automatically run as part of the build process, or you don't include any testing, specify that here.*

Example:

Run unit tests:
```
/test/runUnitTests.sh
```

## Execution
*Explain how to use the final product of your code. If your code is meant to be run as is (ex: scripts), specify how to run those scripts. If your code is meant to be deployed (ex: AWS Lambda), specify how to do that.*

Example:

Run the script:
```
python /scripts/myScript.py
```

Deploy to Lambda:
```
aws lambda create-function --function-name my-function ... 
```

# Additional Notes
*Optional - Any additional information that would be useful to someone wanting to use this code. If there are datasets and associated Data Management Plans that interact with this source code, please provide them here*

Example:

**Known Issues:**
* Script fails when run on Tuesdays

**Associated datasets:**
* [Associated Dataset](https://its.dot.gov/data/)

# Version History and Retention
*A statement of the status of the source code (prototype, alpha, beta, release, etc.), how often users can expect activity on this repository, and a version/release history in the form of a CHANGELOG file. Additionally, include a retention statement that specifies how long this repository will remain publicly accessible*

Example:

**Status:** This project is in the release phase.

**Release Frequency:** This project is updated approximately once every 2-3 weeks

**Release History: See [CHANGELOG.md](CHANGELOG.md)**

**Retention:** This project will remain publicly accessible for a minimum of five years (until at least 06/15/2025).

# License
*Create a file named "LICENSE.md" that contains at least the licensing status of the code and the full text of the open source license or a link to where the license is officially maintained. If for some reason the source code does not use an open license, explain why. See [LICENSE.md](https://github.com/usdot-jpo-codehub/codehub-readme-template/blob/master/LICENSE) for an example of this file.*

Example:

This project is licensed under the Creative Commons 1.0 Universal (CC0 1.0) License - see the [License.MD](https://github.com/usdot-jpo-codehub/codehub-readme-template/blob/master/LICENSE) for more details. 

# Contributions
*Create a file named "CONTRIBUTING.md" explaining how users can interact with this project's repository, your expectations for their conduct, and how contributions by users will be released (e.g. whether they will be released under the same license and whether those contributors waive their rights accordingly). See [CONTRIBUTING.md](https://github.com/usdot-jpo-codehub/codehub-readme-template/blob/master/Contributing.MD) for an example of this file.*

Example:

Please read [CONTRIBUTING.md](https://github.com/usdot-jpo-codehub/codehub-readme-template/blob/master/Contributing.MD) for details on our Code of Conduct, the process for submitting pull requests to us, and how contributions will be released.

# Contact Information
*Provide a primary contact and associated contact information (e.g. email and phone number) for users to contact with questions about this repository.*

Example:

Contact Name: ITS JPO
Contact Information: data.itsjpo@dot.gov, (888)-888-8888

# Acknowledgements
*Describe how users should reference your code if they use it to build additional software, list the Digital Object Identifier for this project, list the sample citations for associated report/paper and for this source code, and (optional) list if you have a 3rd party or any specific contributor to give credit.*

*Sample citation should be in the below format, with the `formatted fields` replaced with details of your source code*

_`author_surname_or_organization`, `first_initial`. (`year`)._ `program_or_source_code_title` _(`code_version`) [Source code]. Provided by ITS CodeHub through GitHub.com. Accessed YYYY-MM-DD from `doi_url`._

Example:

## Citing this code
To track how this government-funded code is used, we request that if you decide to build additional software using this code please acknowledge its Digital Object Identifier in your software's README/documentation.

> Digital Object Identifier: https://doi.org/xxx.xxx/xxxx

To cite this code in a publication or report, please cite our associated report/paper and/or our source code. Below is a sample citation for this code:

> ITS CodeHub Team. (2021). _ITS CodeHub README Template_ (0.1) [Source code]. Provided by ITS CodeHub through GitHub.com. Accessed 2021-01-27 from https://doi.org/xxx.xxx/xxxx.

When you copy or adapt from this code, please include the original URL you copied the source code from and date of retrieval as a comment in your code. Additional information on how to cite can be found in the [ITS CodeHub FAQ](https://its.dot.gov/code/#/faqs).

## Contributors
Shout out to [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) for their README template.
