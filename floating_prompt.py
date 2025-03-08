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
    
    /* Input container styling - Oval shape */
    .input-container {
        display: flex;
        align-items: center;
        background-color: #f8f9fa;
        border-radius: 50px;
        padding: 8px 16px;
        margin-top: 10px;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .input-container:hover {
        border-color: #adb5bd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Input field styling */
    .chat-input {
        flex-grow: 1;
        border: none;
        background: transparent;
        padding: 8px 40px 8px 8px;
        font-size: 14px;
        color: #212529;
        outline: none;
        width: calc(100% - 48px);
    }
    
    /* Send button styling - Positioned inside the input */
    .send-button {
        background: none;
        border: none;
        padding: 8px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s ease;
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
    }
    
    .send-button:hover {
        transform: translateY(-50%) scale(1.1);
    }
    
    /* Arrow icon styling */
    .arrow-icon {
        width: 20px;
        height: 20px;
        fill: #2e6fdb;
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
        
        # Create a hidden text input to capture the user's query
        input_key = f"hidden_input_{context if context else 'general'}"
        user_input = st.text_input("", key=input_key, label_visibility="collapsed")
        
        # Create input container with send button
        st.markdown(
            f'''
            <div class="input-container">
                <input type="text" class="chat-input" 
                    placeholder="{placeholder}" 
                    id="chat_input_{context if context else 'general'}"
                    onkeyup="updateHiddenInput(this.value, '{input_key}')"
                />
                <button class="send-button" onclick="submitForm('{input_key}')">
                    {create_send_button_html()}
                </button>
            </div>
            ''',
            unsafe_allow_html=True
        )
        
        # Add JavaScript for handling input and button click
        st.markdown(
            f'''
            <script>
            // Function to update the hidden Streamlit input
            function updateHiddenInput(value, inputKey) {{
                // Find the Streamlit input element
                const streamlitDoc = window.parent.document;
                const hiddenInput = streamlitDoc.querySelector('input[aria-label="{input_key}"]');
                if (hiddenInput) {{
                    hiddenInput.value = value;
                }}
                
                // Handle Enter key
                if (event && event.key === 'Enter') {{
                    submitForm('{input_key}');
                }}
            }}
            
            // Function to submit the form
            function submitForm(inputKey) {{
                // Get the value from the visible input
                const chatInput = document.querySelector('.chat-input');
                const value = chatInput.value;
                
                if (value.trim() !== '') {{
                    // Find the Streamlit input element
                    const streamlitDoc = window.parent.document;
                    const hiddenInput = streamlitDoc.querySelector('input[aria-label="{input_key}"]');
                    
                    if (hiddenInput) {{
                        // Set the value
                        hiddenInput.value = value;
                        
                        // Find and click the submit button
                        const submitButton = hiddenInput.nextElementSibling;
                        if (submitButton) {{
                            submitButton.click();
                            
                            // Clear the visible input
                            chatInput.value = '';
                        }}
                    }}
                }}
            }}
            </script>
            ''',
            unsafe_allow_html=True
        )
        
        # Process the input
        if user_input and analysis_function:
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