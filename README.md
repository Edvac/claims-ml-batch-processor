# Claims ML Batch Processor

**Production-ready ML pipeline for insurance claims prediction**

Built from Alex's proven notebook analysis into a scalable batch processing system.

## Prerequisites

- Python 3.13+ (recommended)
- Git

## Dependencies

- matplotlib
- numpy
- pandas
- pillow
- pytest
- scikit-learn
- scipy
- xgboost

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Edvac/claims-ml-batch-processor.git
    cd claims-ml-batch-processor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create
    python -m venv .venv
    # Activate (macOS/Linux)
    source .venv/bin/activate
    # Activate (Windows)
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


##  **Quick Start**

1.  **Train the Model (Required once):**
    ```bash
    python main.py --train
    ```

2.  **Run Predictions:**

    *   **Process a sample file:**
        ```bash
        python main.py
        ```
    *   **Process all CSV files in a folder (e.g., `data/`):**
        ```bash
        python main.py --batch data/
        ```
    *   **Process a specific file:**
        ```bash
        python main.py --batch data/your_file_name.csv
        ```
    *   **Process files matching a pattern:**
        ```bash
        python main.py --batch data/ --pattern "claims_*.csv"
        ```

## Output

Prediction results will be saved as CSV files in the `outputs/` directory.

## Documentation

- [Future Additions and Improvements](docs/Enchancements.md)
- [Project Questions](docs/Project_questions.md)
- [Notebook transforamtion journey](docs/TransformationPipelineOverview.md)
