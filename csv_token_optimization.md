 oij # CSV Token Optimization Guide

This document outlines the optimizations implemented to handle large CSV files without exceeding token limits when using AI analysis. The approach focuses on data summarization, selective truncation, and intelligent prompt management.

## Table of Contents

1. [Overview](#overview)
2. [Data Truncation](#data-truncation)
3. [Prompt Formatting](#prompt-formatting)
4. [Token Management](#token-management)
5. [Implementation Details](#implementation-details)

## Overview

When sending CSV data to AI models, token limits can be quickly reached, especially with large datasets. The implemented optimizations address this challenge through:

- **Data summarization** - Converting raw data into statistical summaries
- **Selective truncation** - Limiting the number of items in distributions and recent issues
- **Compact formatting** - Using more token-efficient formatting for metrics and distributions
- **Intelligent prompt truncation** - Preserving essential sections while reducing less important ones
- **Token estimation** - Dynamically adjusting the prompt based on estimated token usage
- **Optimized API parameters** - Using parameters that maximize the quality of responses while managing token usage

## Data Truncation

### `truncate_data_for_context` Function (Lines 1758-1823)

This function prepares and truncates data from the CSV to fit within token limits:

```python
def truncate_data_for_context(df: pd.DataFrame) -> dict:
    """Prepare and truncate data for OpenAI context with optimized token usage"""
    if df is None or len(df) == 0:
        return {}
        
    # Get only failed test cases for analysis
    failed_df = df[df['Execution Status'] == 'Fail'].copy()
    
    # Get the most recent failures (limit to 10 instead of 20)
    recent_failures = failed_df.nlargest(10, 'Execution Date')
    
    # Calculate key metrics
    total_tests = len(df)
    total_failures = len(failed_df)
    failure_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0
    
    # Get top N items for each distribution (limit to top 5 for each category)
    def get_top_n_dict(series, n=5):
        counts = series.value_counts()
        if len(counts) <= n:
            return counts.to_dict()
        else:
            top_n = counts.nlargest(n)
            result = top_n.to_dict()
            if len(counts) > n:
                # Add an "Others" category with the sum of remaining values
                result['Others'] = counts[n:].sum()
            return result
    
    # Create optimized metrics dictionary
    metrics = {
        'total_tests': total_tests,
        'total_failures': total_failures,
        'failure_rate': round(failure_rate, 2),  # Round to 2 decimal places to save tokens
        'recent_failures': len(recent_failures),
        'defect_types': get_top_n_dict(failed_df['Defect Type']),
        'status_dist': get_top_n_dict(failed_df['Defect Status']),
        'lob_dist': get_top_n_dict(failed_df['LOB'])
    }
    
    # Only include optional columns if they exist
    if 'Severity' in failed_df.columns:
        metrics['severity_dist'] = get_top_n_dict(failed_df['Severity'])
    
    if 'Priority' in failed_df.columns:
        metrics['priority_dist'] = get_top_n_dict(failed_df['Priority'])
    
    # For recent issues, only include essential fields and limit descriptions
    recent_issues = []
    for _, row in recent_failures.head(3).iterrows():  # Limit to top 3 most recent issues
        issue = {
            'Test Case ID': row['Test Case ID'],
            'Defect Type': row['Defect Type'],
            'Defect Status': row['Defect Status']
        }
        
        # Truncate description to save tokens
        if 'Defect Description' in row and isinstance(row['Defect Description'], str):
            desc = row['Defect Description']
            issue['Defect Description'] = desc[:100] + '...' if len(desc) > 100 else desc
        
        recent_issues.append(issue)
    
    metrics['recent_issues'] = recent_issues
    
    return metrics
```

**Key Optimizations:**
- Limits recent failures to 10 (down from 20)
- Uses `get_top_n_dict` to limit each distribution to top 5 items
- Adds an "Others" category to summarize remaining items
- Only includes optional columns if they exist
- Limits recent issues to 3 (down from 5)
- Truncates description text to 100 characters

## Prompt Formatting

### `create_analysis_prompt` Function (Lines 1824-1883)

This function creates a token-efficient prompt for the AI analysis:

```python
def create_analysis_prompt(context: str, metrics: dict, user_query: str) -> str:
    """Create a focused prompt for OpenAI analysis with optimized token usage"""
    context_focus = {
        "failure": "failure patterns and root causes",
        "trend": "trends and patterns over time",
        "gap": "testing gaps and areas needing attention",
        "lob": "LOB-specific performance and issues",
        "predictive": "predictions and future risks",
        "root_cause": "root causes and solutions"
    }.get(context, "general analysis")
    
    # Format distributions in a more compact way
    def format_distribution(dist_dict):
        if not dist_dict:
            return "None"
        return ", ".join([f"{k}: {v}" for k, v in dist_dict.items()])
    
    # Format recent issues in a more compact way
    def format_recent_issues(issues):
        if not issues:
            return "None"
        
        formatted = []
        for i, issue in enumerate(issues, 1):
            issue_str = f"Issue {i}: {issue.get('Test Case ID', 'N/A')} - {issue.get('Defect Type', 'N/A')} - {issue.get('Defect Status', 'N/A')}"
            if 'Defect Description' in issue:
                issue_str += f" - {issue.get('Defect Description', 'N/A')}"
            formatted.append(issue_str)
        
        return "\n".join(formatted)
    
    base_prompt = f"""Analyze the following test failure data focusing on {context_focus}.

METRICS:
Tests: {metrics.get('total_tests', 0)} | Failures: {metrics.get('total_failures', 0)} | Rate: {metrics.get('failure_rate', 0)}% | Recent: {metrics.get('recent_failures', 0)}

DISTRIBUTIONS:
- Defect Types: {format_distribution(metrics.get('defect_types', {}))}
- Status: {format_distribution(metrics.get('status_dist', {}))}
- LOB: {format_distribution(metrics.get('lob_dist', {}))}"""

    # Only include optional distributions if they exist
    if 'severity_dist' in metrics:
        base_prompt += f"\n- Severity: {format_distribution(metrics.get('severity_dist', {}))}"
    
    if 'priority_dist' in metrics:
        base_prompt += f"\n- Priority: {format_distribution(metrics.get('priority_dist', {}))}"

    base_prompt += f"""

RECENT ISSUES:
{format_recent_issues(metrics.get('recent_issues', []))}

USER QUESTION: {user_query}

Provide a concise analysis with:
1. Direct answer to the question
2. Key insights from metrics
3. Patterns identified
4. Recommendations
5. Risk assessment"""

    return base_prompt
```

**Key Optimizations:**
- Uses helper functions for compact formatting
- Presents metrics on a single line with separators
- Formats distributions as comma-separated key-value pairs
- Only includes optional distributions if they exist
- Formats recent issues in a structured, readable way
- Removes verbose language and unnecessary words

## Token Management

### `get_ai_analysis` Function (Lines 83-110)

This function handles the AI analysis with token optimization:

```python
def get_ai_analysis(prompt):
    """Get AI analysis using Azure OpenAI API with optimized token usage"""
    if ai_client is not None and st.session_state.data is not None:
        try:
            # Create a more concise system message
            system_message = "You are a QA expert analyzing test failure data. Provide concise, actionable insights."
            
            # Estimate token count (rough approximation)
            estimated_tokens = len(prompt.split()) * 1.3  # Rough estimate: 1 token ≈ 0.75 words
            
            # If prompt is too long, truncate it further
            if estimated_tokens > 3000:  # Leave room for system message and response
                prompt = truncate_prompt_for_token_limit(prompt, 3000)
            
            # Get AI response with optimized parameters
            response = ai_client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800,  # Increased from 500 to allow for more detailed responses
                top_p=0.95,      # Focus on more likely tokens
                frequency_penalty=0.5  # Reduce repetition
            )
           
            # Return the response content
            return response.choices[0].message.content
           
        except Exception as e:
            st.error(f"Error generating AI analysis: {str(e)}")
            return None
    else:
        st.warning("Azure OpenAI client not initialized or no data loaded. Please check your configuration and upload data.")
        return None
```

**Key Optimizations:**
- Uses a concise system message
- Estimates token count to detect when truncation is needed
- Dynamically truncates the prompt if it exceeds token limits
- Optimizes API parameters for better responses

### `truncate_prompt_for_token_limit` Function (Lines 111-204)

This function intelligently truncates prompts to fit within token limits:

```python
def truncate_prompt_for_token_limit(prompt, target_tokens):
    """Intelligently truncate a prompt to fit within token limits"""
    lines = prompt.split('\n')
    
    # Identify sections in the prompt
    sections = []
    current_section = []
    current_section_name = "header"
    
    for line in lines:
        if line.strip().startswith('METRICS:'):
            # Save previous section and start a new one
            if current_section:
                sections.append((current_section_name, current_section))
            current_section = [line]
            current_section_name = "metrics"
        elif line.strip().startswith('DISTRIBUTIONS:'):
            if current_section:
                sections.append((current_section_name, current_section))
            current_section = [line]
            current_section_name = "distributions"
        elif line.strip().startswith('RECENT ISSUES:'):
            if current_section:
                sections.append((current_section_name, current_section))
            current_section = [line]
            current_section_name = "issues"
        elif line.strip().startswith('USER QUESTION:'):
            if current_section:
                sections.append((current_section_name, current_section))
            current_section = [line]
            current_section_name = "question"
        elif line.strip().startswith('Provide a concise analysis'):
            if current_section:
                sections.append((current_section_name, current_section))
            current_section = [line]
            current_section_name = "instructions"
        else:
            current_section.append(line)
    
    # Add the last section
    if current_section:
        sections.append((current_section_name, current_section))
    
    # Prioritize sections (keep question, metrics, and instructions intact)
    essential_sections = ["header", "metrics", "question", "instructions"]
    truncatable_sections = ["distributions", "issues"]
    
    # Calculate tokens for essential sections
    essential_text = ""
    for name, section in sections:
        if name in essential_sections:
            essential_text += "\n".join(section) + "\n"
    
    essential_tokens = len(essential_text.split()) * 1.3
    remaining_tokens = target_tokens - essential_tokens
    
    # If essential sections already exceed the limit, truncate the prompt drastically
    if remaining_tokens <= 0:
        return f"""Analyze the following test data.

USER QUESTION: {prompt.split('USER QUESTION:')[1].split('Provide')[0].strip()}

Provide a concise analysis."""
    
    # Truncate non-essential sections
    truncated_sections = {}
    for name, section in sections:
        if name in truncatable_sections:
            section_text = "\n".join(section)
            section_tokens = len(section_text.split()) * 1.3
            
            if name == "distributions":
                # Keep only the first 3 distribution lines
                truncated_sections[name] = [section[0]] + section[1:4]
            elif name == "issues":
                # Keep only the first 2 issues
                issue_lines = []
                issue_count = 0
                for line in section:
                    if line.strip().startswith("Issue"):
                        issue_count += 1
                        if issue_count <= 2:
                            issue_lines.append(line)
                    else:
                        issue_lines.append(line)
                truncated_sections[name] = issue_lines
            else:
                truncated_sections[name] = section
    
    # Reconstruct the prompt
    truncated_prompt = ""
    for name, section in sections:
        if name in essential_sections:
            truncated_prompt += "\n".join(section) + "\n"
        elif name in truncatable_sections:
            truncated_prompt += "\n".join(truncated_sections[name]) + "\n"
    
    return truncated_prompt.strip()
```

**Key Optimizations:**
- Identifies different sections of the prompt
- Prioritizes essential sections (question, metrics, instructions)
- Selectively truncates less important sections (distributions, issues)
- For distributions, keeps only the first 3 distribution lines
- For issues, keeps only the first 2 issues
- Has a fallback for extreme cases where even essential sections exceed limits

## Implementation Details

### Session State Initialization (Lines 74-80)

```python
# Initialize session state variables
if 'prompt' not in st.session_state:
    st.session_state.prompt = False
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = False
if 'data' not in st.session_state:
    st.session_state.data = None
```

### Data Loading and Validation (Lines 167-190)

```python
def load_and_validate_data(file):
    """Load and validate the uploaded CSV file or file path"""
    try:
        # Check if file is a string (file path) or an uploaded file
        if isinstance(file, str):
            # Load from file path
            df = pd.read_csv(file)
        else:
            # Load from uploaded file
            df = pd.read_csv(file)
            
        required_columns = ['Execution Date', 'Application', 'Test Case ID', 'LOB',
                          'Execution Status', 'Defect ID', 'Defect Description',
                          'Defect Type', 'Defect Status']
       
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
           
        # Convert Execution Date to datetime
        df['Execution Date'] = pd.to_datetime(df['Execution Date'])
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
```

### Beautiful Prompt Section (Lines 1667-1757)

```python
def beautiful_prompt_section(context: Optional[str] = None, analysis_function: Optional[Callable] = None):
    # ... (code omitted for brevity)
    
    # Process the input
    if user_input and analysis_function and st.session_state.data is not None:
        with st.spinner(""):
            try:
                # Store the current input in session state to preserve it
                if f"prev_input_{context}" not in st.session_state:
                    st.session_state[f"prev_input_{context}"] = ""
                st.session_state[f"prev_input_{context}"] = user_input
                
                # Prepare data for context
                metrics = truncate_data_for_context(st.session_state.data)
                
                # Create focused prompt
                prompt = create_analysis_prompt(context or "general", metrics, user_input)
                
                # Get AI response
                response = analysis_function(prompt)
                
                if response:
                    st.markdown('<div class="chat-response">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("Unable to generate analysis. Please try again.")
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
```

## Conclusion

These optimizations work together to ensure that you can analyze larger CSV files without hitting token limits, while still maintaining high-quality AI analysis. The approach is scalable and can be adapted to different types of data and analysis requirements.

Key benefits:
- Handles larger datasets without token limit errors
- Maintains analysis quality by prioritizing important information
- Provides graceful degradation when token limits are approached
- Optimizes token usage through intelligent formatting and truncation 