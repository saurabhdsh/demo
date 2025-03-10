import streamlit as st
from typing import Optional, Callable
import pandas as pd
 
def create_send_button_html():
    """Create HTML for the send button with arrow icon"""
    return """<svg class="arrow-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
    </svg>"""

def floating_prompt_section(context: Optional[str] = None, analysis_function: Optional[Callable] = None):
    """Create floating prompt section with optional context"""
   
    # Add floating prompt CSS
    st.markdown("""
    <style>
    .floating-chat {
        position: fixed;
        left: 20px;
        bottom: 20px;
        width: 350px;
        background-color: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
        z-index: 1000;
        border: 1px solid #f0f0f0;
        font-family: 'Arial', sans-serif;
    }
    
    .chat-response {
        margin-top: 15px;
        padding: 15px;
        border-radius: 12px;
        background-color: #f8f9fa;
        max-height: 250px;
        overflow-y: auto;
        font-size: 14px;
        line-height: 1.5;
        color: #212529;
        border: 1px solid #e9ecef;
    }
    
    /* Custom styling for chat input */
    .stChatInput {
        margin-top: 10px;
    }
    
    .stChatInput > div {
        border-radius: 50px !important;
        border: 1px solid #e9ecef !important;
        background-color: #f8f9fa !important;
        position: relative !important;
        display: flex !important;
        align-items: center !important;
    }
    
    .stChatInput input {
        border-radius: 50px !important;
        padding-right: 40px !important;
        width: calc(100% - 40px) !important;
    }
    
    .stChatInput button {
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #f8f9fa !important;
        border: none !important;
        color: #2e6fdb !important;
        position: absolute !important;
        right: 4px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
    }
    
    .stChatInput button:hover {
        background-color: #e9ecef !important;
    }
    </style>
    """, unsafe_allow_html=True)
   
    with st.container():
        st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
        
        # Add context-specific placeholder text
        placeholder = "Ask about test failures, defects, or trends..."
        if context == "failure":
            placeholder = "Ask about failure patterns, defect types, or severity distribution..."
        elif context == "trend":
            placeholder = "Ask about failure trends, historical patterns, or predictions..."
        elif context == "gap":
            placeholder = "Ask about testing gaps, coverage issues, or improvement areas..."
        elif context == "lob":
            placeholder = "Ask about specific LOB performance, issues, or comparisons..."
        elif context == "predictive":
            placeholder = "Ask about future trends, risk areas, or potential issues..."
        elif context == "root_cause":
            placeholder = "Ask about root causes and solutions..."
        
        # Add JavaScript to preserve input after submission
        st.markdown(
            """
            <script>
            // Store the input value in session storage when it changes
            document.addEventListener('DOMContentLoaded', function() {
                // Wait for Streamlit to fully load
                setTimeout(function() {
                    const chatInputs = document.querySelectorAll('.stChatInput input');
                    chatInputs.forEach(input => {
                        // Set initial value from session storage if exists
                        const storedValue = sessionStorage.getItem(input.id);
                        if (storedValue) {
                            input.value = storedValue;
                        }
                        
                        // Store value when it changes
                        input.addEventListener('input', function() {
                            sessionStorage.setItem(this.id, this.value);
                        });
                        
                        // Clear storage when form is submitted
                        const form = input.closest('form');
                        if (form) {
                            form.addEventListener('submit', function() {
                                // Don't clear the input value
                                setTimeout(function() {
                                    input.value = sessionStorage.getItem(input.id);
                                }, 100);
                            });
                        }
                    });
                }, 1000);
            });
            </script>
            """,
            unsafe_allow_html=True
        )
        
        # Use Streamlit's chat_input
        user_input = st.chat_input(placeholder=placeholder, key=f"chat_input_{context if context else 'general'}")
        
        # Process the input
        if user_input and analysis_function:
            with st.spinner("Analyzing..."):
                # Store the current input in session state to preserve it
                if f"prev_input_{context}" not in st.session_state:
                    st.session_state[f"prev_input_{context}"] = ""
                st.session_state[f"prev_input_{context}"] = user_input
                
                try:
                    # Check if data is available
                    if st.session_state.data is not None:
                        # Prepare data for context
                        metrics = truncate_data_for_context(st.session_state.data)
                        
                        # Create focused prompt
                        prompt = create_analysis_prompt(context or "general", metrics, user_input)
                        
                        # Get AI response with the formatted prompt
                        response = analysis_function(prompt)
                    else:
                        # If no data is available, just pass the user input directly
                        response = analysis_function(user_input)
                    
                    if response:
                        st.markdown(f"<div class='chat-response'>{response}</div>", unsafe_allow_html=True)
                    else:
                        st.error("Unable to generate analysis. Please try again.")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
       
        st.markdown('</div>', unsafe_allow_html=True)
 
def add_floating_prompt_to_tab(tab_name: str, analysis_function: Optional[Callable] = None):
    """Add floating prompt with context to a specific tab"""
    context_map = {
        "failure": "failure",
        "trend": "trend",
        "gap": "gap",
        "lob": "lob",
        "predictive": "predictive",
        "root_cause": "root_cause"
    }
    floating_prompt_section(context_map.get(tab_name.lower()), analysis_function)
 
def truncate_data_for_context(df: pd.DataFrame) -> dict:
    """Prepare and truncate data for OpenAI context"""
    if df is None or len(df) == 0:
        return {}
 
    # Get only failed test cases for analysis
    failed_df = df[df['Execution Status'] == 'Fail'].copy()
    # Get the most recent 20 failures - This limits the data volume
    recent_failures = failed_df.nlargest(20, 'Execution Date')
 
    # Calculate key metrics - This summarizes data instead of sending raw data
    total_tests = len(df)
    total_failures = len(failed_df)
 
    metrics = {
        'total_tests': total_tests,
        'total_failures': total_failures,
        'failure_rate': (total_failures / total_tests * 100) if total_tests > 0 else 0,
        'recent_failures': len(recent_failures),
 
        # Using value_counts().to_dict() creates summaries instead of raw data
        'defect_types': failed_df['Defect Type'].value_counts().to_dict(),
        'severity_dist': failed_df['Severity'].value_counts().to_dict() if 'Severity' in failed_df.columns else {},
        'priority_dist': failed_df['Priority'].value_counts().to_dict() if 'Priority' in failed_df.columns else {},
        'status_dist': failed_df['Defect Status'].value_counts().to_dict(),
        'lob_dist': failed_df['LOB'].value_counts().to_dict(),
 
        # Only sending essential columns and limiting to recent issues
        'recent_issues': recent_failures[[
            'Test Case ID',
            'Defect Type',
            'Defect Status',
            'Defect Description'
        ]].to_dict('records')[:5]  # Further limiting to only 5 recent issues
    }
    
    return metrics

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