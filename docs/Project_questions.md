# Project Questions & Answers

## 1. What are the assumptions you have made for this service and why?

*   **Data Format:** Daily applications (~1200) available as CSV from ETL at predictable location/time. **Why:** Aligns with batch processing and existing code structure.
*   **Data Quality:** ETL provides clean data with correct types. **Why:** ML pipeline focuses on ML preprocessing, not basic data validation.
*   **Model Performance:** Alex's model performance is acceptable for production. **Why:** Model improvement explicitly out of scope.
*   **Infrastructure:** Sufficient compute available for ~1200 daily records. **Why:** Volume is moderate for typical batch processing.

## 2. What considerations are there to ensure the business can leverage this service?

*   **Decision Workflows:** Claims team needs clear processes - what to do with HIGH vs MEDIUM vs LOW risk predictions.
*   **Threshold Calibration:** Business must define acceptable risk levels and adjust score thresholds based on actual outcomes.
*   **Staff Training:** Trevor's team needs training on interpreting risk scores and integrating predictions into underwriting decisions.
*   **Integration Planning:** How predictions fit into existing policy approval workflows and systems.
*   **Success Metrics:** Business must define how to measure if predictions are actually improving claims outcomes.


## 3. Which traditional teams within the business would you need to talk to and why?

### Most Likely Required:
*   **Claims Department (Trevor's Team):** Primary stakeholders - understand business needs, output format, risk thresholds.
*   **Underwriting Department:** Integration into policy approval workflow and decision timelines.

### Quite Likely Required:
*   **Finance/Actuarial:** Reserve calculation methodology, risk categorization impact on financial planning.
*   **Compliance/Legal:** Regulatory adherence, data privacy, audit trail requirements for automated decisions.
*   **Operations/Policy Administration:** Integration with existing application processing systems.



## 4. What is in and out of scope for your responsibility?

*   **In Scope:**
    *   Extract Alex's logic into production-ready Python pipeline
    *   Daily batch prediction service
    *   Model training capability
    *   Production-ready code structure and documentation
    *   Deployment strategy and architecture recommendation

*   **Out of Scope:**
*   Model performance improvement or algorithm changes
    *   Real-time/streaming prediction endpoints
    *   Actual infrastructure provisioning and live deployment
    *   Advanced monitoring, alerting, or CI/CD implementation
    *   New feature engineering or business logic development
