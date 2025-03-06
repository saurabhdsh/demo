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
    total_tests = len(df)
    total_failures = len(failed_df)
    failure_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0
    
    metrics = {
        'total_tests': total_tests,
        'total_failures': total_failures,
        'failure_rate': failure_rate,
        'recent_failures': len(recent_failures),
        'defect_types': failed_df['Defect Type'].value_counts().to_dict(),
        'severity_dist': failed_df['Severity'].value_counts().to_dict(),
        'priority_dist': failed_df['Priority'].value_counts().to_dict(),
        'status_dist': failed_df['Defect Status'].value_counts().to_dict(),
        'lob_dist': failed_df['LOB'].value_counts().to_dict(),
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
    # Create context-specific focus
    context_focus = {
        "failure": "failure patterns and root causes",
        "trend": "trends and patterns over time",
        "gap": "testing gaps and areas needing attention",
        "lob": "LOB-specific performance and issues",
        "predictive": "predictions and future risks"
    }.get(context, "general analysis")
    
    base_prompt = f"""As a QA expert, analyze the following test failure data focusing on {context_focus}.

Current Metrics:
- Total Tests: {metrics.get('total_tests', 0)}
- Total Failures: {metrics.get('total_failures', 0)}
- Failure Rate: {metrics.get('failure_rate', 0):.2f}%
- Recent Failures: {metrics.get('recent_failures', 0)}

Distribution Summary:
- Defect Types: {metrics.get('defect_types', {})}
- Severity: {metrics.get('severity_dist', {})}
- Priority: {metrics.get('priority_dist', {})}
- Status: {metrics.get('status_dist', {})}
- LOB Distribution: {metrics.get('lob_dist', {})}

Recent Critical Issues:
{str(metrics.get('recent_issues', [])[:5])}

User Question: {user_query}

Please provide a detailed analysis including:
1. Direct answer to the user's question
2. Key insights from the relevant metrics
3. Specific patterns or trends identified
4. Actionable recommendations
5. Risk assessment if applicable

Format your response in markdown with appropriate headers and bullet points."""

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
    .chat-input {
        margin-bottom: 10px;
    }
    .chat-button {
        width: 100%;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
        
        # Add context-specific placeholder text
        placeholder_text = {
            "failure": "Ask about failure patterns, defect types, or specific issues...",
            "trend": "Ask about trends, patterns over time, or specific date ranges...",
            "gap": "Ask about gaps in testing, coverage, or specific areas...",
            "lob": "Ask about specific LOB performance, issues, or comparisons...",
            "predictive": "Ask about predictions, future trends, or risk areas..."
        }.get(context, "Ask a question about the analysis...")
        
        user_input = st.text_input(
            "",
            placeholder=placeholder_text,
            key=f"chat_input_{context}" if context else "chat_input",
            help="Type your question and press Enter or click Ask"
        )
        
        if st.button("Ask", key=f"ask_button_{context}" if context else "ask_button", use_container_width=True):
            if user_input and analysis_function and st.session_state.data is not None:
                with st.spinner("Analyzing..."):
                    try:
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
            elif st.session_state.data is None:
                st.warning("Please upload data first to use the AI analysis feature.")
            elif not user_input:
                st.warning("Please enter a question to analyze.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def add_floating_prompt_to_tab(tab_name: str, analysis_function: Optional[Callable] = None):
    """Add floating prompt with context to a specific tab"""
    floating_prompt_section(tab_name.lower(), analysis_function) 