import streamlit as st
from typing import Optional, Callable
import pandas as pd

def create_beautiful_prompt_css():
    """Add custom CSS for beautiful prompt styling"""
    st.markdown("""
    <style>
    /* Main container styling */
    .beautiful-chat {
        position: fixed;
        left: 20px;
        bottom: 20px;
        width: 350px;
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
        z-index: 1000;
        border: 1px solid #f0f0f0;
        font-family: 'Arial', sans-serif;
    }
    
    /* Input container styling */
    .input-container {
        display: flex;
        align-items: center;
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 8px 16px;
        margin-top: 10px;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
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
        padding: 8px;
        font-size: 14px;
        color: #212529;
        outline: none;
    }
    
    /* Send button styling */
    .send-button {
        background: none;
        border: none;
        padding: 8px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s ease;
    }
    
    .send-button:hover {
        transform: scale(1.1);
    }
    
    /* Arrow icon styling */
    .arrow-icon {
        width: 20px;
        height: 20px;
        fill: #212529;
    }
    
    /* Response container styling */
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
    
    /* Scrollbar styling */
    .chat-response::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-response::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    .chat-response::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }
    
    .chat-response::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Loading animation */
    .loading-dots {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
    }
    
    .dot {
        width: 8px;
        height: 8px;
        margin: 0 4px;
        background-color: #212529;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes pulse {
        0%, 100% { transform: scale(0.8); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

def create_send_button_html():
    """Create HTML for the send button with arrow icon"""
    return """
    <svg class="arrow-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
    </svg>
    """

def beautiful_prompt_section(context: Optional[str] = None, analysis_function: Optional[Callable] = None):
    """Create beautiful floating prompt section with arrow send button"""
    
    # Add custom CSS
    create_beautiful_prompt_css()
    
    # Create container
    with st.container():
        st.markdown('<div class="beautiful-chat">', unsafe_allow_html=True)
        
        # Add context-specific placeholder text
        placeholder_text = {
            "failure": "Ask about failure patterns or specific issues...",
            "trend": "Ask about trends and patterns over time...",
            "gap": "Ask about testing gaps and coverage...",
            "lob": "Ask about LOB performance and issues...",
            "predictive": "Ask about predictions and risks..."
        }.get(context, "Ask a question about the analysis...")
        
        # Create input container with send button
        st.markdown(
            f'''
            <div class="input-container">
                <input type="text" class="chat-input" 
                    placeholder="{placeholder_text}" 
                    id="chat_input_{context if context else 'general'}"
                />
                <button class="send-button" onclick="send_message()">
                    {create_send_button_html()}
                </button>
            </div>
            ''',
            unsafe_allow_html=True
        )
        
        # Add JavaScript for handling input and button click
        st.markdown(
            '''
            <script>
            function send_message() {
                const input = document.querySelector('.chat-input');
                const message = input.value;
                if (message) {
                    // Clear input
                    input.value = '';
                    // Send message to Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:message',
                        data: message
                    }, '*');
                }
            }
            
            // Handle Enter key
            document.querySelector('.chat-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    send_message();
                }
            });
            </script>
            ''',
            unsafe_allow_html=True
        )
        
        # Handle message processing
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        user_input = st.text_input(
            "",
            key=f"hidden_input_{context if context else 'general'}",
            label_visibility="collapsed"
        )
        
        if user_input and analysis_function and st.session_state.data is not None:
            with st.spinner(""):
                try:
                    # Prepare data for context
                    metrics = truncate_data_for_context(st.session_state.data)
                    
                    # Create focused prompt
                    prompt = create_analysis_prompt(context or "general", metrics, user_input)
                    
                    # Show loading animation
                    st.markdown(
                        '''
                        <div class="loading-dots">
                            <div class="dot"></div>
                            <div class="dot"></div>
                            <div class="dot"></div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                    
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
        
        st.markdown('</div>', unsafe_allow_html=True)

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
        'severity_dist': failed_df['Severity'].value_counts().to_dict() if 'Severity' in failed_df.columns else {},
        'priority_dist': failed_df['Priority'].value_counts().to_dict() if 'Priority' in failed_df.columns else {},
        'status_dist': failed_df['Defect Status'].value_counts().to_dict(),
        'lob_dist': failed_df['LOB'].value_counts().to_dict(),
        'recent_issues': recent_failures[[
            'Test Case ID', 
            'Defect Type', 
            'Defect Status',
            'Defect Description'
        ]].to_dict('records')[:5]  # Limit to 5 most recent issues
    }
    
    return metrics

def create_analysis_prompt(context: str, metrics: dict, user_query: str) -> str:
    """Create a focused prompt for OpenAI analysis"""
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
{str(metrics.get('recent_issues', []))}

User Question: {user_query}

Please provide a detailed analysis including:
1. Direct answer to the user's question
2. Key insights from the relevant metrics
3. Specific patterns or trends identified
4. Actionable recommendations
5. Risk assessment if applicable

Format your response in markdown with appropriate headers and bullet points."""

    return base_prompt

def add_beautiful_prompt_to_tab(tab_name: str, analysis_function: Optional[Callable] = None):
    """Add beautiful prompt with context to a specific tab"""
    beautiful_prompt_section(tab_name.lower(), analysis_function) 