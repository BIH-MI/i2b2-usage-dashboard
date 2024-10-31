# i2b2-usage-dashboard

## About the project

We developed an interactive dashboard that analyzes usage metrics to provide data stewards and administrators with a comprehensive understanding of how users at their institution are interacting with the Informatics for Integrating Biology and the Bedside (i2b2) platform. This can help to identify strengths and limitations of the platform in general, as well as the current data representation supporting the local use case. It can thus serve as a communication aid for the dialog with users to identify potential areas for improvement in supporting efficient access to health data.

We first identified data usage dimensions and relevant visualizations, which were then implemented by assessing and analyzing the metadata of previous user queries. The dashboard provides an overview of the number of queries and users over time, frequently queried concepts, and query complexity. The visualizations were developed in Python and integrated into a Dash application to create an interactive dashboard.

## Deployment

The dashboard itself is a Python application, where the dependencies are listed in the `requirements.txt`. We additionally provided example configuration files for docker compose in order to run the dashboard in a docker container. 

## Contact

If you have questions or problems, please open an issue here on GitHub or contact us directly.

You can find out more about our work at [mi.bihealth.org](https://mi.bihealth.org).

## License

&copy; 2023-2024 Berlin Institute of Health

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
