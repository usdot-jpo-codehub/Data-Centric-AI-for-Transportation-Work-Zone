# Summary:

This repository contains the source code of the work zone data curation pipline and the AMCNN-ED work zone traffic impact prediction model. The data curation pipeline for this project employs a sophisticated integration of diverse traffic-related data sources, significantly enhancing the quality and utility of the data post-integration.The traffic prediction model predicts the traffic speed conditions on roadway segments of a planned work zone before its deployment. The source code was developed based on Pytorch 2.0. The data curation code generates an integrated time series traffic data with work zone attributes, and the AMCNN-ED code generate a trained pytorch model for traffic condition prediction. The source code was programmed in Jupyter Lab with Python.


# README Outline:
* Project Description
* Prerequisites
* Usage
	* Building
	* Testing
* Authors
* License
* Acknowledgements

# Project Description


In alignment with the Data Centric AI concept, this project was funded under the Intelligent Transportation Systems Joint Program Office (ITS JPO) of the U.S. Department of Transportation (U.S. DOT) and collaborated with the Maryland Department of Transportation (MDOT) to leverage advanced AI methodologies for work zone-related impact prediction. This project prioritizes the creation of a comprehensive work zone data set as its initial goal, followed by the development of the work zone impact prediction model. This endeavor involves a meticulous process of collecting, integrating, and organizing diverse traffic data to address existing gaps and enhance the richness of work zone data sets. Machine learning, a subfield of AI, focuses on the development of algorithms capable of learning from data. Subsequently, the project focuses on accessing and predicting the traffic impact of work zones through the deployment of machine learning technologies, thereby encapsulating both the challenges and innovations in this field.

# Prerequisites

Requires:
- Python 3.9 (or higher)
- Pytorch 2.0
- Jupyter Lab


# Usage
*Provide users with detailed instructions for how to use your software. The specifics of this section will vary between projects, but should adhere to the following minimum outline:*

## Building

Step 1: Create a virtual environment and navigate to project folder.

Step 2: Run data curation model:
```
python data_curation.py
```
Step 3: Run model training module:
```
python model_train.py
```

## Testing

Run model tests:
```
python model_test.py
```

# Authors

Qinhua Jiang, PhD Candidate, University of California, Los Angeles. Email: qhjiang93@ucla.edu.
Yaofa Gong, research assistant, University of California, Los Angeles. Email: gongyaofa0211@g.ucla.edu


# License

This project is licensed under the Creative Commons 1.0 Universal (CC0 1.0) License - see the [License.MD](https://github.com/usdot-jpo-codehub/codehub-readme-template/blob/master/LICENSE) for more details. 


# Acknowledgements
This research is supported by the Intelligent Transportation Systems Joint Program Office (ITS JPO) of the U.S. Department of Transportation (U.S. DOT) 


## Contributors
Shout out to [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) for their README template.
