# Project Questions & Answers

## 1. What are the assumptions you have made for this service and why?

*   **Data Format:** Daily applications (~1200) available as CSV from ETL at predictable location/time. **Why:** Aligns with batch processing and existing code structure.
*   **Data Quality:** ETL provides clean data with correct types. **Why:** ML pipeline focuses on ML preprocessing, not basic data validation.
*   **Model Performance:** Alex's model performance is acceptable for production. **Why:** Model improvement explicitly out of scope.
*   **Infrastructure:** Sufficient compute available for ~1200 daily records. **Why:** Volume is moderate for typical batch processing.

## 2. What considerations are there to ensure the business can leverage this service?

*   **Clear Output:** CSV with application IDs and risk scores consumable by downstream systems.
*   **Actionable Insights:** Claims team needs guidance on score thresholds and workflows.
*   **Timeliness:** Predictions available before policy binding decisions.
*   **Monitoring:** Pipeline health monitoring and model performance tracking.
*   **Feedback Loop:** Process to link predictions to actual outcomes for evaluation.

## 3. Which traditional teams within the business would you need to talk to and why?

*   **Claims Department (Trevor's Team):** Primary stakeholders - understand needs, output format, thresholds.
*   **Data Scientists/DevOps:** ETL process coordination, data access, infrastructure, scheduling.
*   **Underwriting/Policy Operations:** Integration into policy binding workflow and timelines.
*   **Compliance/Legal:** Regulatory adherence, data privacy, production service standards.

## 4. What is in and out of scope for your responsibility?

*   **In Scope:**
    *   Extract Alex's logic into production Python pipeline
    *   Daily batch prediction service
    *   Model retraining capability
    *   Production-ready code structure and documentation
    *   Deployment framework description

*   **Out of Scope:**
    *   Model performance improvement
    *   Real-time prediction endpoints
    *   Actual live deployment
    *   Advanced CI/CD or monitoring systems
    *   New feature development