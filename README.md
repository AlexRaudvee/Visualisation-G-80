# SharkSight

SharkSight is a Streamlit-based data visualization tool designed to analyze shark attack incidents in Australia. This application aims to assist Australian beach safeguards and related authorities in enhancing beach safety by providing interactive visualizations and insights into shark attack data.

## Features
- Interactive Visualizations: Utilize various visualization idioms, including parallel coordinates plots and bar charts, to explore and analyze shark attack data.
- Data Interactivity: Engage with the data through multiple interactions, allowing for in-depth analysis and understanding.
- User-Friendly Interface: Built on the Streamlit framework, the application offers an intuitive and accessible interface for users.

## Installation
### Prerequisites
1. Python 3.7 or higher
2. pip (Python package installer)
### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/AlexRaudvee/SharkSight.git
    cd SharkSight
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the dataloader - `dataloader.py`:

    ```bash
    python dataloader.py
    ```
4. Run the Streamlit Application:

    ```bash
    streamlit run main.py --theme='light'
    ```
5. Access the application:
    Open your web browser and navigate to http://localhost:8501/ to start using the tool.

## Usage

### Data Loading

Ensure that the shark attack dataset is properly loaded into the application. The dataset should include relevant information such as incident dates, locations, shark species involved, and outcomes.

### Exploring Visualizations

- Parallel Coordinates Plot: Select multiple attributes to visualize relationships and patterns across different dimensions of the data.
- Bar Charts: Compare various attributes, such as the frequency of attacks over different years or regions, to identify trends.
Interactivity:
- Map: Select the region of interests, as well there is possibility to select two regions by turning ON this function in the Controls Menu (Side Menu)

Utilize the interactive features to filter and highlight specific data points, aiding in focused analysis and decision-making.

### Contributing
Contributions to enhance the functionality or usability of SharkSight are welcome. To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add your feature description'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request detailing your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For inquiries or feedback, please contact the repository owner: AlexRaudvee

## Acknowledgments
- Data Source: The shark attack data utilized in this application is sourced from the **Australian Shark-Incident Database** (ASID), which quantifies temporal and spatial patterns of shark-human interactions in Australia.
- Framework: This application is built using Streamlit, an open-source app framework for Machine Learning and Data Science teams.