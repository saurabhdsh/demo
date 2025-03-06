import streamlit as st
from typing import Optional, Callable
import pandas as pd

def truncate_data_for_context(df: pd.DataFrame) -> dict:
    """Prepare and truncate data for OpenAI context"""
    if df is None or len(df) == 0:
        return {}
        
    # Get only failed test cases for analysis
    failed_df = df[df['Execution Status'] == 'Fail'].copy()
    
    # Get the most recent 20 failures
    recent_failures = failed_df.nlargest(20, 'Execution Date')
    
    # Calculate key metrics
    metrics = {
        'total_failures': len(failed_df),
        'recent_failures': len(recent_failures),
        'defect_types': failed_df['Defect Type'].value_counts().to_dict(),
        'severity_dist': failed_df['Severity'].value_counts().to_dict(),
        'priority_dist': failed_df['Priority'].value_counts().to_dict(),
        'status_dist': failed_df['Defect Status'].value_counts().to_dict(),
        'lob_dist': failed_df['LOB'].value_counts().to_dict(),
        # Include recent failures sample with test case ID instead of name
        'recent_issues': recent_failures[[
            'Test Case ID', 
            'Defect Type', 
            'Severity', 
            'Priority', 
            'Defect Status',
            'Defect Description'
        ]].to_dict('records')
    }
    
    return metrics

def create_analysis_prompt(context: str, metrics: dict, user_query: str) -> str:
    """Create a focused prompt for OpenAI analysis"""
    base_prompt = f"""As a QA expert, analyze the following test failure data focusing on {context}.
Current metrics summary:
- Total Failures: {metrics.get('total_failures', 0)}
- Recent Failures: {metrics.get('recent_failures', 0)}

Distribution Summary:
- Defect Types: {', '.join([f'{k}: {v}' for k, v in metrics.get('defect_types', {}).items()])}
- Severity: {', '.join([f'{k}: {v}' for k, v in metrics.get('severity_dist', {}).items()])}
- Priority: {', '.join([f'{k}: {v}' for k, v in metrics.get('priority_dist', {}).items()])}
- Status: {', '.join([f'{k}: {v}' for k, v in metrics.get('status_dist', {}).items()])}

Recent Issues Sample (up to 5 most critical):
{str(metrics.get('recent_issues', [])[:5])}

User Question: {user_query}

Provide a concise analysis focusing on:
1. Direct answer to the user's question
2. Key insights from the relevant metrics
3. Specific recommendations if applicable
4. If the question is about specific test cases, include their failure details and current status"""

    return base_prompt

def floating_prompt_section(context: Optional[str] = None, analysis_function: Optional[Callable] = None):
    """Create floating prompt section with optional context"""
    
    # Add floating prompt CSS
    st.markdown("""
    <style>
    .floating-chat {
        position: fixed;
        left: 20px;
        bottom: 20px;
        width: 300px;
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        border: 1px solid #e0e0e0;
    }
    .chat-response {
        margin-top: 10px;
        padding: 10px;
        border-radius: 8px;
        background-color: #f7f7f7;
        max-height: 200px;
        overflow-y: auto;
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
        
        user_input = st.text_input(
            "",
            placeholder=placeholder,
            key=f"chat_input_{context}" if context else "chat_input"
        )
        
        if st.button("Ask", key=f"ask_button_{context}" if context else "ask_button"):
            if user_input and analysis_function and st.session_state.data is not None:
                with st.spinner("Analyzing..."):
                    # Prepare truncated data for context
                    metrics = truncate_data_for_context(st.session_state.data)
                    
                    # Create focused prompt
                    prompt = create_analysis_prompt(context or "general analysis", metrics, user_input)
                    
                    # Get AI response
                    response = analysis_function(prompt)
                    
                    if response:
                        st.markdown('<div class="chat-response">', unsafe_allow_html=True)
                        st.markdown(response)
                        st.markdown('</div>', unsafe_allow_html=True)
            elif st.session_state.data is None:
                st.warning("Please upload data first to use the AI analysis feature.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def add_floating_prompt_to_tab(tab_name: str, analysis_function: Optional[Callable] = None):
    """Add floating prompt with context to a specific tab"""
    context_map = {
        "failure": "failure",
        "trend": "trend",
        "gap": "gap",
        "lob": "lob",
        "predictive": "predictive"
    }
    floating_prompt_section(context_map.get(tab_name.lower()), analysis_function) 