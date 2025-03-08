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
        
        # Initialize session state for this context if it doesn't exist
        input_key = f"input_{context if context else 'general'}"
        if input_key not in st.session_state:
            st.session_state[input_key] = ""
        
        # Create a form to handle the input submission
        with st.form(key=f"chat_form_{context if context else 'general'}", clear_on_submit=False):
            # Use text_input instead of chat_input to have more control
            user_input = st.text_input(
                "",
                value=st.session_state[input_key],
                placeholder=placeholder,
                key=f"text_{input_key}",
                label_visibility="collapsed"
            )
            
            # Create a custom submit button that looks like a send button
            col1, col2 = st.columns([6, 1])
            with col2:
                submitted = st.form_submit_button("Send")
        
        # Apply custom CSS to make it look like our design
        st.markdown(
            """
            <style>
            /* Style the form to look like chat input */
            .stForm {
                background-color: transparent !important;
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            
            /* Hide the form border */
            .stForm > div {
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            
            /* Style the text input */
            .stTextInput > div > div > input {
                border-radius: 50px !important;
                border: 1px solid #e9ecef !important;
                background-color: #f8f9fa !important;
                padding: 8px 16px !important;
            }
            
            /* Style the submit button */
            .stForm button {
                border-radius: 50% !important;
                width: 40px !important;
                height: 40px !important;
                padding: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                background-color: #f8f9fa !important;
                border: 1px solid #e9ecef !important;
                color: #2e6fdb !important;
                font-size: 20px !important;
                line-height: 1 !important;
            }
            
            .stForm button:hover {
                background-color: #e9ecef !important;
            }
            
            /* Hide the label */
            .stTextInput label {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Process the input when the form is submitted
        if submitted and user_input:
            # Store the input in session state
            st.session_state[input_key] = user_input
            
            if analysis_function:
                with st.spinner("Analyzing..."):
                    response = analysis_function(user_input)
                    if response:
                        st.markdown(f"<div class='chat-response'>{response}</div>", unsafe_allow_html=True)
       
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