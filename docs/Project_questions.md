# Project Questions & Answers

## 1. What are the assumptions you have made for this service and why?

*   **Data Format:** Daily applications (~1200) available as CSV from ETL at predictable location/time. **Why:** Aligns with batch processing and existing code structure.
*   **Data Quality:** ETL provides clean data with correct types. **Why:** ML pipeline focuses on ML preprocessing, not basic data validation.
*   **Model Performance:** Alex's model performance is acceptable for production. **Why:** Model improvement explicitly out of scope.
*   **Infrastructure:** Sufficient compute available for ~1200 daily records. **Why:** Volume is moderate for typical batch processing.
