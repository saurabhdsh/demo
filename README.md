# QA Failure Analysis Dashboard

A comprehensive Streamlit application for analyzing QA execution data, providing insights, and AI-powered recommendations for test failure patterns.

## Features

- **File Upload & Data Processing**
  - CSV file upload with automatic validation
  - Summary statistics and data preview
  - Interactive filtering and sorting

- **Multi-Tab Analysis**
  1. **Failure Analysis**
     - Overall failure rate visualization
     - LOB-wise failure breakdown
     - Defect category analysis
  
  2. **Failure Trends**
     - Time-series analysis
     - Interactive date range and LOB filters
     - Daily failure trend visualization
  
  3. **Gap Analysis**
     - Frequently failing test cases
     - Pattern detection
     - Test coverage insights
  
  4. **LOB-Wise Analysis**
     - Failure distribution by LOB
     - Comparative analysis
     - Failure rate metrics
  
  5. **Predictive Analysis**
     - Future failure predictions
     - Trend forecasting
     - 30-day failure probability
  
  6. **AI Recommendations**
     - OpenAI-powered insights
     - Root cause analysis
     - Actionable recommendations

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`
- OpenAI API key for AI recommendations

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd qa-failure-analysis
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload your QA execution data CSV file with the following required columns:
   - TestCase: Test case identifier
   - LOB: Line of Business
   - ExecutionStatus: Pass/Fail status
   - ExecutionDate: Date of execution
   - DefectType: Type of defect (for failures)

4. Navigate through the different tabs to explore various analyses and insights

## CSV Format Example

```csv
TestCase,LOB,ExecutionStatus,ExecutionDate,DefectType
TC001,Banking,Pass,2024-01-01,None
TC002,Insurance,Fail,2024-01-01,Automation
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 