import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv
import openai
from functools import reduce

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="QA Failure Analysis Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for floating prompt
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

/* Defect Item Styling */
.defect-item {
    padding: 12px;
    margin: 10px 0;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 8px;
    border: 1px solid #eaecef;
}

.defect-item p {
    margin: 5px 0;
}

.status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 500;
    background: #e9ecef;
    color: #495057;
}

.test-case {
    font-size: 0.9em;
    color: #6c757d;
}

/* Enhanced Card Styles */
.response-card {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    max-width: 800px;
    margin: 20px auto;
    padding: 0;
}

.title-section {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 12px;
}

.title-section h1 {
    font-size: 1.1em;
    margin: 0;
    font-weight: 600;
}

.title-section h2 {
    font-size: 0.9em;
    opacity: 0.9;
    margin: 4px 0 0 0;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}

.metric-card {
    background: white;
    padding: 12px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #f0f0f0;
}

.metric-label {
    color: #64748b;
    font-size: 0.75em;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.1em;
    font-weight: 600;
    background: linear-gradient(45deg, #2563eb, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.distribution-section {
    background: white;
    padding: 16px;
    border-radius: 8px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.distribution-title {
    color: #1e293b;
    font-size: 0.9em;
    font-weight: 600;
    margin-bottom: 12px;
}

.distribution-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px;
    margin: 6px 0;
    background: #f8fafc;
    border-radius: 6px;
    font-size: 0.8em;
}

.distribution-item span:last-child {
    font-weight: 500;
    color: #2563eb;
}

.recommendations-section {
    background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
    color: white;
    padding: 16px;
    border-radius: 8px;
    margin-top: 16px;
}

.recommendations-title {
    font-size: 0.9em;
    font-weight: 600;
    margin-bottom: 12px;
    opacity: 0.9;
}

.recommendations-list {
    margin: 0;
    padding-left: 16px;
}

.recommendations-list li {
    margin: 8px 0;
    font-size: 0.8em;
    line-height: 1.4;
    opacity: 0.9;
}

/* Statistics Styling */
.metric-value {
    background: linear-gradient(45deg, #3498db, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Emoji Styling */
.response-card h1 emoji, .response-card h2 emoji {
    font-size: 1.2em;
    margin-right: 8px;
}

/* Distribution Values */
.distribution-value {
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
    color: #6c757d;
    background: #f8f9fa;
    padding: 2px 6px;
    border-radius: 4px;
}

/* LOB Analysis Specific Styles */
.lob-stats {
    padding: 15px;
    background: rgba(255, 255, 255, 0.7);
    border-radius: 8px;
    margin: 10px 0;
}

.lob-stats p {
    font-size: 1.1em;
    margin-bottom: 15px;
}

.defect-breakdown {
    margin-top: 15px;
}

.defect-breakdown h3 {
    color: #2c3e50;
    font-size: 1em;
    margin: 15px 0 10px 0;
    padding-bottom: 5px;
    border-bottom: 1px solid #eaecef;
}

.defect-breakdown ul {
    list-style: none;
    padding-left: 0;
    margin-bottom: 20px;
}

.defect-breakdown li {
    margin: 8px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 10px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 6px;
}

.defect-type {
    font-weight: 500;
    color: #2c3e50;
    padding: 2px 8px;
    background: #e9ecef;
    border-radius: 12px;
    font-size: 0.9em;
}

/* Enhanced Card Styles for LOB Analysis */
.success-card {
    border-left: 4px solid #2ecc71;
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(255, 255, 255, 0.9));
}

.info-card {
    border-left: 4px solid #3498db;
    background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(255, 255, 255, 0.9));
}

.warning-card {
    border-left: 4px solid #f1c40f;
    background: linear-gradient(135deg, rgba(241, 196, 15, 0.1), rgba(255, 255, 255, 0.9));
}

/* Metric Value Enhancement */
.metric-value {
    font-size: 1.4em;
    font-weight: 700;
    background: linear-gradient(45deg, #2c3e50, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-right: 5px;
}

/* Status Badge Enhancement */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 500;
    background: #e9ecef;
    color: #495057;
    margin-right: 8px;
}

.status-badge.open {
    background: #fee2e2;
    color: #dc2626;
}

.status-badge.closed {
    background: #dcfce7;
    color: #16a34a;
}

.status-badge.in-progress {
    background: #fef3c7;
    color: #d97706;
}

/* Key Findings Enhancement */
.warning-card ul {
    padding-left: 20px;
}

.warning-card li {
    margin: 10px 0;
    line-height: 1.6;
    color: #2c3e50;
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    .lob-stats {
        padding: 10px;
    }
    
    .metric-value {
        font-size: 1.2em;
    }
    
    .defect-breakdown li {
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = False

# Initialize OpenAI client at the start of the script
if 'openai_client' not in st.session_state:
    try:
        st.session_state.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
    except Exception as e:
        st.warning(f"OpenAI client initialization failed: {str(e)}")
        st.session_state.openai_client = None

def get_ai_analysis(prompt):
    """Get AI analysis using Azure OpenAI API"""
    if st.session_state.openai_client is not None and st.session_state.data is not None:
        try:
            # Calculate actual distributions from the data
            df = st.session_state.data
            failed_df = df[df['Execution Status'] == 'Fail']
            
            # Calculate defect type distribution
            defect_type_dist = failed_df['Defect Type'].value_counts()
            defect_type_percentages = (defect_type_dist / defect_type_dist.sum() * 100).round(1)
            
            # Calculate severity distribution
            severity_dist = failed_df['Severity'].value_counts()
            severity_percentages = (severity_dist / severity_dist.sum() * 100).round(1)
            
            # Calculate priority distribution
            priority_dist = failed_df['Priority'].value_counts()
            priority_percentages = (priority_dist / priority_dist.sum() * 100).round(1)
            
            # Calculate status distribution
            status_dist = failed_df['Defect Status'].value_counts()
            status_percentages = (status_dist / status_dist.sum() * 100).round(1)
            
            # Format distributions for the system message
            defect_type_metrics = "\n".join([f"              • {level}: {count}" for level, count in defect_type_dist.items()])
            severity_metrics = "\n".join([f"              • {level}: {count}" for level, count in severity_dist.items()])
            priority_metrics = "\n".join([f"              • {level}: {count}" for level, count in priority_dist.items()])
            
            defect_type_analysis = "\n".join([f"              • {level}: {count} defects ({percentage}%)" 
                                         for level, count, percentage in zip(defect_type_dist.index, 
                                                                          defect_type_dist.values, 
                                                                          defect_type_percentages)])
            
            severity_analysis = "\n".join([f"              • {level}: {count} defects ({percentage}%)" 
                                         for level, count, percentage in zip(severity_dist.index, 
                                                                          severity_dist.values, 
                                                                          severity_percentages)])
            
            priority_analysis = "\n".join([f"              • {level}: {count} defects ({percentage}%)" 
                                         for level, count, percentage in zip(priority_dist.index, 
                                                                          priority_dist.values, 
                                                                          priority_percentages)])
            
            status_analysis = "\n".join([f"              • {status}: {count} defects ({percentage}%)" 
                                       for status, count, percentage in zip(status_dist.index, 
                                                                         status_dist.values, 
                                                                         status_percentages)])
            
            # Format system message with actual data
            system_message = f"""You are a QA expert specializing in test failure analysis. 
            Your response should be clear and concise, focusing on key metrics and actual values.
            Format your response using the smallest possible heading levels:
            
            ##### 📊 [Brief Title]
            ###### [One-line Context]
            
            **Key Metrics:**
            - Total Defects: {len(failed_df)}
            - Defect Type Distribution:
{defect_type_metrics}
            - Severity Distribution:
{severity_metrics}
            - Priority Distribution:
{priority_metrics}
            
            **Distribution Analysis:**
            - By Defect Type:
{defect_type_analysis}
            - By Severity:
{severity_analysis}
            - By Priority:
{priority_analysis}
            - By Status:
{status_analysis}
            
            Note: Use these actual values in your analysis. For queries about specific defect types, provide the exact count and percentage.
            When asked about specific types of issues (e.g., Environment issues), filter the data accordingly and provide precise numbers."""

            # Get AI response
            response = st.session_state.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Get the response content and render it
            content = response.choices[0].message.content
            st.markdown(content)
            
        except Exception as e:
            if "context_length_exceeded" in str(e):
                st.error("Analysis contains too much data. Trying with summarized information...")
                try:
                    truncated_prompt = truncate_prompt(prompt)
                    response = st.session_state.openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": truncated_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as nested_e:
                    st.error(f"Error generating AI analysis: {str(nested_e)}")
                    return None
            else:
                st.error(f"Error generating AI analysis: {str(e)}")
                return None
    else:
        st.warning("Azure OpenAI client not initialized. Please check your configuration.")
        return None

def truncate_prompt(prompt):
    """Truncate the prompt to handle token limits"""
    # Split the prompt into lines
    lines = prompt.split('\n')
    header_lines = [line for line in lines if not line.strip().startswith('{') and not line.strip().isdigit()]
    data_lines = [line for line in lines if line.strip().startswith('{') or line.strip().isdigit()]
    
    # Take only the first 10 most relevant data points
    truncated_data = data_lines[:10] if len(data_lines) > 10 else data_lines
    
    # Combine header and truncated data
    return '\n'.join(header_lines + ['', 'Summary of top issues:'] + truncated_data)

def load_and_validate_data(file):
    """Load and validate the uploaded CSV file"""
    try:
        df = pd.read_csv(file)
        required_columns = ['Execution Date', 'User Story', 'Test Case ID', 'LOB', 
                          'Execution Status', 'Defect ID', 'Defect Description', 
                          'Defect Type', 'Defect Status', 'Severity', 'Priority']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
            
        # Convert Execution Date to datetime
        df['Execution Date'] = pd.to_datetime(df['Execution Date'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_summary_stats(df):
    """Calculate summary statistics"""
    total_tests = len(df)
    failed_df = df[df['Execution Status'] == 'Fail']
    
    # Calculate basic metrics
    pass_rate = (df['Execution Status'].value_counts().get('Pass', 0) / total_tests) * 100
    fail_rate = (df['Execution Status'].value_counts().get('Fail', 0) / total_tests) * 100
    total_defects = len(failed_df)
    unique_defects = failed_df['Defect ID'].nunique()
    
    # Calculate defect type distribution
    defect_type_dist = failed_df['Defect Type'].value_counts()
    
    # Calculate severity distribution
    severity_dist = failed_df['Severity'].value_counts()
    
    # Calculate priority distribution
    priority_dist = failed_df['Priority'].value_counts()
    
    # Calculate LOB distribution
    lob_dist = failed_df['LOB'].value_counts()
    
    return {
        'total_tests': total_tests,
        'pass_rate': pass_rate,
        'fail_rate': fail_rate,
        'total_defects': total_defects,
        'unique_defects': unique_defects,
        'defect_type_dist': defect_type_dist,
        'severity_dist': severity_dist,
        'priority_dist': priority_dist,
        'lob_dist': lob_dist
    }

def get_ai_response(prompt, context):
    """Get AI analysis for specific prompts"""
    try:
        if st.session_state.openai_client is None:
            return "OpenAI client not initialized. Please check your environment variables."
            
        analysis_prompt = f"""
        Based on the following context about test failures:
        {context}
        
        User Question: {prompt}
        
        Please provide a detailed analysis focusing specifically on the user's question.
        Include:
        1. Direct answer to the question
        2. Supporting evidence from the data
        3. Recommendations if applicable
        """
        
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a QA expert analyzing test failure data."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def floating_prompt_section():
    """Create floating prompt section"""
    with st.container():
        st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
        
        user_input = st.text_input(
            "",
            placeholder="Ask questions like: 'What are the open defects?' or 'Show defects in C&I module'",
            key="chat_input"
        )
        
        if st.button("Ask", key="ask_button"):
            if user_input:
                with st.spinner("Analyzing..."):
                    get_ai_analysis(user_input)
        
        st.markdown('</div>', unsafe_allow_html=True)

def failure_analysis_tab(df):
    """Tab 1: Failure Analysis with Interactive Prompting"""
    st.header("🔍 Failure Analysis")
    
    # Add floating prompt section
    floating_prompt_section()
    
    # Store the current section for context-aware prompting
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 'overview'
    
    # Modern styling configuration
    chart_config = {
        'template': 'plotly_white',
        'font': dict(family="Arial, sans-serif"),
        'title_font_size': 20,
        'showlegend': True
    }
    
    # Pass/Fail Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced Pass/Fail Distribution
        status_counts = df['Execution Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig_status = px.pie(
            status_counts, 
            values='Count',
            names='Status',
            title='Overall Pass/Fail Distribution',
            color_discrete_sequence=['#00CC96', '#EF553B'],
            hole=0.4
        )
        fig_status.update_layout(**chart_config)
        fig_status.update_traces(textposition='outside', textinfo='percent+label+value')
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        # Enhanced LOB-wise Failure Breakdown
        failed_df = df[df['Execution Status'] == 'Fail']
        lob_failures = failed_df['LOB'].value_counts().reset_index()
        lob_failures.columns = ['LOB', 'Count']
        
        fig_lob = px.pie(
            lob_failures,
            values='Count',
            names='LOB',
            title='LOB-wise Failure Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_lob.update_layout(**chart_config)
        fig_lob.update_traces(textposition='outside', textinfo='percent+label+value')
        st.plotly_chart(fig_lob, use_container_width=True)

    # Add detailed failure pattern analysis
    st.subheader("📊 Detailed Failure Pattern Analysis")
    
    # Calculate defect type distribution
    defect_type_dist = failed_df.groupby(['Defect Type', 'Severity']).size().reset_index(name='Count')
    
    # Create pattern analysis columns
    col3, col4 = st.columns(2)
    
    with col3:
        # Defect type distribution with severity breakdown
        fig_defect_type = px.bar(
            defect_type_dist,
            x='Defect Type',
            y='Count',
            color='Severity',
            title='Defect Type Distribution by Severity',
            labels={'Count': 'Number of Failures', 'Defect Type': 'Type of Defect'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_defect_type.update_layout(
            template='plotly_white',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_defect_type, use_container_width=True)
    
    with col4:
        # Priority distribution
        priority_dist = failed_df.groupby(['Priority', 'Defect Status']).size().reset_index(name='Count')
        fig_priority = px.bar(
            priority_dist,
            x='Priority',
            y='Count',
            color='Defect Status',
            title='Priority Distribution by Status',
            labels={'Count': 'Number of Defects'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_priority.update_layout(
            template='plotly_white',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_priority, use_container_width=True)

    # Add defect status breakdown
    st.subheader("🎯 Defect Status Analysis")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Status distribution by LOB
        status_by_lob = failed_df.groupby(['LOB', 'Defect Status']).size().reset_index(name='Count')
        fig_status_lob = px.bar(
            status_by_lob,
            x='LOB',
            y='Count',
            color='Defect Status',
            title='Defect Status Distribution by LOB',
            labels={'Count': 'Number of Defects'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_status_lob.update_layout(template='plotly_white')
        st.plotly_chart(fig_status_lob, use_container_width=True)
    
    with col6:
        # Severity distribution by status
        severity_by_status = failed_df.groupby(['Severity', 'Defect Status']).size().reset_index(name='Count')
        fig_severity_status = px.bar(
            severity_by_status,
            x='Severity',
            y='Count',
            color='Defect Status',
            title='Severity Distribution by Status',
            labels={'Count': 'Number of Defects'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_severity_status.update_layout(
            template='plotly_white',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_severity_status, use_container_width=True)

    # Add heatmap for defect distribution
    st.subheader("🔥 Defect Distribution Matrix")
    
    # Create heatmap data
    heatmap_data = pd.crosstab(
        failed_df['Defect Type'],
        failed_df['LOB'],
        values=failed_df['Defect ID'],
        aggfunc='count',
        normalize='columns'
    ).fillna(0) * 100
    
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="LOB", y="Defect Type", color="Percentage"),
        title="Defect Type Distribution Across LOBs (%)",
        color_continuous_scale="RdYlBu_r",
        aspect="auto"
    )
    fig_heatmap.update_traces(text=heatmap_data.round(1), texttemplate="%{text}%")
    fig_heatmap.update_layout(template='plotly_white')
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

def trend_analysis_tab(df):
    """Tab 2: Failure Trends Over Time"""
    st.header("📈 Failure Trends Over Time")
    
    # Date range filter
    date_range = st.date_input(
        "Select Date Range",
        [df['Execution Date'].min(), df['Execution Date'].max()]
    )
    
    # LOB filter
    selected_lob = st.multiselect(
        "Select LOB(s)",
        options=df['LOB'].unique(),
        default=df['LOB'].unique()
    )
    
    filtered_df = df[
        (df['Execution Date'].dt.date >= date_range[0]) &
        (df['Execution Date'].dt.date <= date_range[1]) &
        (df['LOB'].isin(selected_lob))
    ]
    
    # Calculate daily metrics
    daily_stats = filtered_df.groupby('Execution Date').agg({
        'Execution Status': lambda x: (x == 'Fail').sum(),
        'Test Case ID': 'count'
    }).reset_index()
    daily_stats['Failure Rate'] = (daily_stats['Execution Status'] / daily_stats['Test Case ID'] * 100).round(2)
    
    # Create trend analysis columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Failure count trend
        fig_count = px.line(
            daily_stats,
            x='Execution Date',
            y='Execution Status',
            title='Daily Failure Count Trend',
            labels={'Execution Status': 'Number of Failures', 'Execution Date': 'Date'}
        )
        fig_count.update_layout(template='plotly_white')
        st.plotly_chart(fig_count, use_container_width=True)
    
    with col2:
        # Failure rate trend
        fig_rate = px.line(
            daily_stats,
            x='Execution Date',
            y='Failure Rate',
            title='Daily Failure Rate Trend (%)',
            labels={'Failure Rate': 'Failure Rate (%)', 'Execution Date': 'Date'}
        )
        fig_rate.update_layout(template='plotly_white')
        st.plotly_chart(fig_rate, use_container_width=True)
    
    # Defect type trends
    st.subheader("📊 Defect Type Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Defect type trend
        defect_trend = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['Execution Date', 'Defect Type']
        ).size().reset_index(name='Count')
        
        fig_defect = px.line(
            defect_trend,
            x='Execution Date',
            y='Count',
            color='Defect Type',
            title='Defect Type Trends',
            labels={'Count': 'Number of Defects'}
        )
        fig_defect.update_layout(template='plotly_white')
        st.plotly_chart(fig_defect, use_container_width=True)
    
    with col4:
        # Severity trend
        severity_trend = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['Execution Date', 'Severity']
        ).size().reset_index(name='Count')
        
        fig_severity = px.line(
            severity_trend,
            x='Execution Date',
            y='Count',
            color='Severity',
            title='Severity Trends',
            labels={'Count': 'Number of Defects'}
        )
        fig_severity.update_layout(template='plotly_white')
        st.plotly_chart(fig_severity, use_container_width=True)
    
    # Status and Priority Analysis
    st.subheader("📈 Status and Priority Trends")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Status trend
        status_trend = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['Execution Date', 'Defect Status']
        ).size().reset_index(name='Count')
        
        fig_status = px.line(
            status_trend,
            x='Execution Date',
            y='Count',
            color='Defect Status',
            title='Defect Status Trends',
            labels={'Count': 'Number of Defects'}
        )
        fig_status.update_layout(template='plotly_white')
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col6:
        # Priority trend
        priority_trend = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['Execution Date', 'Priority']
        ).size().reset_index(name='Count')
        
        fig_priority = px.line(
            priority_trend,
            x='Execution Date',
            y='Count',
            color='Priority',
            title='Priority Trends',
            labels={'Count': 'Number of Defects'}
        )
        fig_priority.update_layout(template='plotly_white')
        st.plotly_chart(fig_priority, use_container_width=True)
    
    # LOB Analysis
    st.subheader("🏢 LOB-wise Trend Analysis")
    
    # LOB failure trends
    lob_trend = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
        ['Execution Date', 'LOB']
    ).size().reset_index(name='Count')
    
    fig_lob = px.line(
        lob_trend,
        x='Execution Date',
        y='Count',
        color='LOB',
        title='LOB-wise Failure Trends',
        labels={'Count': 'Number of Failures'}
    )
    fig_lob.update_layout(template='plotly_white')
    st.plotly_chart(fig_lob, use_container_width=True)

def gap_analysis_tab(df):
    """Tab 3: Gap Analysis"""
    st.header("🔍 Gap Analysis")
    
    # Filter failed test cases
    failed_df = df[df['Execution Status'] == 'Fail']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top failing test cases with defect types
        test_case_failures = failed_df.groupby(['Test Case ID', 'Defect Type']).size().reset_index(name='Count')
        test_case_failures = test_case_failures.sort_values('Count', ascending=False).head(10)
        
        fig_test_cases = px.bar(
            test_case_failures,
            x='Test Case ID',
            y='Count',
            color='Defect Type',
            title='Top 10 Frequently Failing Test Cases by Defect Type',
            labels={'Count': 'Failure Count'}
        )
        fig_test_cases.update_layout(
            template='plotly_white',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_test_cases, use_container_width=True)
    
    with col2:
        # User Story analysis with severity
        story_failures = failed_df.groupby(['User Story', 'Severity']).size().reset_index(name='Count')
        story_failures = story_failures.sort_values('Count', ascending=False).head(10)
        
        fig_stories = px.bar(
            story_failures,
            x='User Story',
            y='Count',
            color='Severity',
            title='Top 10 User Stories with Most Failures by Severity',
            labels={'Count': 'Failure Count'}
        )
        fig_stories.update_layout(
            template='plotly_white',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_stories, use_container_width=True)
    
    # Defect Resolution Analysis
    st.subheader("📊 Defect Resolution Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Status distribution by severity
        status_severity = failed_df.groupby(['Defect Status', 'Severity']).size().reset_index(name='Count')
        
        fig_status_severity = px.bar(
            status_severity,
            x='Defect Status',
            y='Count',
            color='Severity',
            title='Defect Status Distribution by Severity',
            labels={'Count': 'Number of Defects'}
        )
        fig_status_severity.update_layout(template='plotly_white')
        st.plotly_chart(fig_status_severity, use_container_width=True)
    
    with col4:
        # Priority distribution by status
        priority_status = failed_df.groupby(['Priority', 'Defect Status']).size().reset_index(name='Count')
        
        fig_priority_status = px.bar(
            priority_status,
            x='Priority',
            y='Count',
            color='Defect Status',
            title='Priority Distribution by Status',
            labels={'Count': 'Number of Defects'}
        )
        fig_priority_status.update_layout(
            template='plotly_white',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_priority_status, use_container_width=True)
    
    # LOB Analysis
    st.subheader("🏢 LOB-wise Analysis")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # LOB distribution by defect type
        lob_defect = failed_df.groupby(['LOB', 'Defect Type']).size().reset_index(name='Count')
        
        fig_lob_defect = px.bar(
            lob_defect,
            x='LOB',
            y='Count',
            color='Defect Type',
            title='LOB Distribution by Defect Type',
            labels={'Count': 'Number of Defects'}
        )
        fig_lob_defect.update_layout(template='plotly_white')
        st.plotly_chart(fig_lob_defect, use_container_width=True)
    
    with col6:
        # LOB distribution by severity
        lob_severity = failed_df.groupby(['LOB', 'Severity']).size().reset_index(name='Count')
        
        fig_lob_severity = px.bar(
            lob_severity,
            x='LOB',
            y='Count',
            color='Severity',
            title='LOB Distribution by Severity',
            labels={'Count': 'Number of Defects'}
        )
        fig_lob_severity.update_layout(template='plotly_white')
        st.plotly_chart(fig_lob_severity, use_container_width=True)
    
    # Gap Matrix
    st.subheader("📉 Gap Analysis Matrix")
    
    # Create gap matrix
    gap_matrix = pd.crosstab(
        [failed_df['LOB'], failed_df['Severity']],
        [failed_df['Defect Type'], failed_df['Defect Status']]
    ).fillna(0)
    
    # Flatten the matrix for visualization
    gap_matrix_flat = gap_matrix.reset_index()
    
    # Create heatmap
    fig_heatmap = px.imshow(
        gap_matrix,
        labels=dict(x="Defect Type & Status", y="LOB & Severity", color="Count"),
        title="Comprehensive Gap Analysis Matrix",
        aspect="auto"
    )
    fig_heatmap.update_layout(
        template='plotly_white',
        height=600
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

def lob_analysis_tab(df):
    """Tab 4: LOB-Wise Failure Analysis"""
    st.header("📊 LOB-Wise Failure Analysis")
    
    # Add weekend vs weekday analysis
    df['IsWeekend'] = df['Execution Date'].dt.dayofweek.isin([5, 6])
    df['DayOfWeek'] = df['Execution Date'].dt.day_name()
    
    # LOB selector
    selected_lob = st.selectbox("Select LOB for Detailed Analysis", df['LOB'].unique())
    
    lob_df = df[df['LOB'] == selected_lob]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced failure rate trend with dotted line and hover data
        lob_daily_stats = lob_df.groupby(['Execution Date', 'DayOfWeek']).agg({
            'Execution Status': lambda x: (x == 'Fail').mean() * 100,
            'Defect Description': lambda x: '<br>'.join(x[x.notna()].unique())
        }).reset_index()
        
        fig_lob_trend = go.Figure()
        fig_lob_trend.add_trace(go.Scatter(
            x=lob_daily_stats['Execution Date'],
            y=lob_daily_stats['Execution Status'],
            mode='lines+markers',
            line=dict(dash='dot', color='#1f77b4'),
            name='Failure Rate',
            hovertemplate="<b>Date:</b> %{x}<br>" +
                         "<b>Failure Rate:</b> %{y:.1f}%<br>" +
                         "<b>Day:</b> %{customdata[0]}<br>" +
                         "<b>Issues:</b> %{customdata[1]}<extra></extra>",
            customdata=lob_daily_stats[['DayOfWeek', 'Defect Description']].values
        ))
        
        fig_lob_trend.update_layout(
            title=f'Failure Rate Trend for {selected_lob}',
            xaxis_title='Date',
            yaxis_title='Failure Rate (%)',
            template='plotly_white',
            hovermode='x unified'
        )
        st.plotly_chart(fig_lob_trend)
    
    with col2:
        # Enhanced Weekend vs Weekday Analysis with defect types
        weekend_stats = lob_df.groupby(['IsWeekend', 'Defect Type']).agg({
            'Execution Status': lambda x: (x == 'Fail').mean() * 100
        }).reset_index()
        weekend_stats['Day Type'] = weekend_stats['IsWeekend'].map({True: 'Weekend', False: 'Weekday'})
        
        fig_weekend = px.bar(
            weekend_stats,
            x='Day Type',
            y='Execution Status',
            color='Defect Type',
            title=f'Weekend vs Weekday Failure Rate by Defect Type for {selected_lob}',
            labels={'Execution Status': 'Failure Rate (%)', 'Defect Type': 'Type of Defect'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_weekend.update_layout(template='plotly_white', barmode='group')
        st.plotly_chart(fig_weekend)

    # Add detailed defect analysis
    st.subheader("Detailed Defect Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        # Most common defect patterns
        defect_patterns = lob_df[lob_df['Execution Status'] == 'Fail'].groupby(['Defect Type', 'Defect Description']).size()
        defect_patterns = defect_patterns.sort_values(ascending=False).head(5)
        
        fig_patterns = px.bar(
            x=defect_patterns.index.get_level_values(0),
            y=defect_patterns.values,
            title=f'Top 5 Defect Patterns in {selected_lob}',
            labels={'x': 'Defect Type', 'y': 'Occurrence Count'},
            color=defect_patterns.index.get_level_values(0),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_patterns)
    
    with col4:
        # Test case vulnerability analysis
        tc_vulnerability = lob_df[lob_df['Execution Status'] == 'Fail'].groupby(['Test Case ID', 'Defect Type']).size()
        tc_vulnerability = tc_vulnerability.sort_values(ascending=False).head(5)
        
        fig_tc = px.bar(
            x=tc_vulnerability.index.get_level_values(0),
            y=tc_vulnerability.values,
            title=f'Most Vulnerable Test Cases in {selected_lob}',
            labels={'x': 'Test Case ID', 'y': 'Failure Count'},
            color=tc_vulnerability.index.get_level_values(1),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_tc)

    # Enhanced AI Analysis Section
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI LOB Analysis")
        
        # Get detailed metrics for analysis
        weekend_defect_details = lob_df[lob_df['Execution Status'] == 'Fail'].groupby(['IsWeekend', 'Defect Type', 'Defect Description']).size().nlargest(5)
        
        # Get specific test case failure patterns
        tc_failure_patterns = lob_df[lob_df['Execution Status'] == 'Fail'].groupby(['Test Case ID', 'Defect Description']).size().nlargest(5)
        
        # Calculate stability metrics
        stability_score = (1 - lob_df['Execution Status'].eq('Fail').mean()) * 100
        
        # Get trend of defect types over time
        recent_defect_trend = lob_df[lob_df['Execution Status'] == 'Fail'].groupby(['Execution Date', 'Defect Type']).size().tail(5)
        
        analysis_prompt = f"""
        Detailed analysis for {selected_lob}:

        Stability Score: {stability_score:.1f}%

        Top Weekend vs Weekday Issues with Descriptions:
        {weekend_defect_details.to_string()}

        Most Problematic Test Cases and Their Issues:
        {tc_failure_patterns.to_string()}

        Recent Defect Type Trends:
        {recent_defect_trend.to_string()}

        Please provide:
        1. Specific analysis of the most critical issues in {selected_lob}
        2. Detailed weekend vs weekday vulnerability patterns with root causes
        3. Test case stability analysis with specific failure reasons
        4. Concrete recommendations based on the actual defect descriptions
        5. Risk assessment for each identified pattern
        6. Actionable mitigation strategies for the top issues
        7. Specific areas needing immediate attention based on recent trends
        """
        
        with st.spinner(f"Generating detailed analysis for {selected_lob}..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)

def predictive_analysis_tab(df):
    """Tab 5: Predictive Analysis"""
    st.header("🔮 Predictive Analysis")
    
    # Prepare data for prediction
    df['DaysSinceStart'] = (df['Execution Date'] - df['Execution Date'].min()).dt.days
    df['FailureFlag'] = (df['Execution Status'] == 'Fail').astype(int)
    
    # Group by date and calculate failure rate
    daily_stats = df.groupby('Execution Date').agg({
        'FailureFlag': 'mean'
    }).reset_index()
    
    X = np.array(range(len(daily_stats))).reshape(-1, 1)
    y = daily_stats['FailureFlag'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions for next 30 days
    future_days = np.array(range(len(X), len(X) + 30)).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    # Calculate confidence metrics
    prediction_std = np.std(predictions)
    confidence_scores = {}
    trend_indicators = {}
    
    for lob in df['LOB'].unique():
        lob_df = df[df['LOB'] == lob]
        if len(lob_df) > 0:
            # Calculate recent trend
            recent_failures = lob_df.tail(7)['FailureFlag'].mean()
            historical_failures = lob_df['FailureFlag'].mean()
            trend = recent_failures - historical_failures
            
            # Calculate confidence score (0-100)
            data_points = len(lob_df)
            consistency = 1 - np.std(lob_df['FailureFlag'])
            confidence = (0.4 * data_points/len(df) + 0.6 * consistency) * 100
            
            confidence_scores[lob] = confidence
            trend_indicators[lob] = trend
    
    # Plot actual vs predicted with confidence band
    fig_pred = go.Figure()
    
    # Actual data
    fig_pred.add_trace(go.Scatter(
        x=daily_stats['Execution Date'],
        y=daily_stats['FailureFlag'],
        name='Actual Failure Rate',
        mode='markers+lines'
    ))
    
    # Predictions with confidence band
    future_dates = pd.date_range(
        start=daily_stats['Execution Date'].max(),
        periods=30,
        freq='D'
    )
    
    fig_pred.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name='Predicted Failure Rate',
        line=dict(dash='dash')
    ))
    
    # Add confidence bands
    fig_pred.add_trace(go.Scatter(
        x=future_dates,
        y=predictions + 2*prediction_std,
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0)',
        showlegend=False
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=future_dates,
        y=predictions - 2*prediction_std,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0)',
        name='95% Confidence Interval'
    ))
    
    fig_pred.update_layout(
        title='Failure Rate Prediction for Next 30 Days',
        xaxis_title='Date',
        yaxis_title='Failure Rate',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Create confidence assessment visualization
    st.subheader("Prediction Confidence Assessment")
    
    confidence_df = pd.DataFrame({
        'LOB': list(confidence_scores.keys()),
        'Confidence': list(confidence_scores.values()),
        'Trend': list(trend_indicators.values())
    })
    
    # Add trend indicators
    confidence_df['TrendIndicator'] = confidence_df['Trend'].apply(
        lambda x: "↑" if x > 0.05 else "↓" if x < -0.05 else "→"
    )
    confidence_df['TrendColor'] = confidence_df['Trend'].apply(
        lambda x: 'red' if x > 0.05 else 'green' if x < -0.05 else 'grey'
    )
    
    fig_confidence = go.Figure()
    
    # Add confidence bars
    fig_confidence.add_trace(go.Bar(
        x=confidence_df['LOB'],
        y=confidence_df['Confidence'],
        text=confidence_df.apply(lambda row: f"{row['Confidence']:.1f}% {row['TrendIndicator']}", axis=1),
        textposition='auto',
        marker_color=confidence_df['Confidence'].apply(
            lambda x: 'rgb(44, 160, 44)' if x >= 70 
            else 'rgb(255, 127, 14)' if x >= 40 
            else 'rgb(214, 39, 40)'
        )
    ))
    
    fig_confidence.update_layout(
        title='Prediction Confidence by LOB with Trend Indicators',
        xaxis_title='Line of Business',
        yaxis_title='Confidence Score (%)',
        yaxis_range=[0, 100],
        template='plotly_white'
    )
    
    st.plotly_chart(fig_confidence, use_container_width=True)
    
    # Failure probability by LOB with enhanced visualization
    st.subheader("Failure Probability by LOB")
    lob_predictions = {}
    
    for lob in df['LOB'].unique():
        lob_df = df[df['LOB'] == lob]
        if len(lob_df) > 0:
            lob_stats = lob_df.groupby('Execution Date')['FailureFlag'].mean().reset_index()
            X_lob = np.array(range(len(lob_stats))).reshape(-1, 1)
            y_lob = lob_stats['FailureFlag'].values
            
            model_lob = LinearRegression()
            model_lob.fit(X_lob, y_lob)
            
            # Predict next day
            next_day_pred = model_lob.predict([[len(X_lob)]])[0]
            lob_predictions[lob] = next_day_pred * 100
    
    # Display LOB predictions with trend indicators
    lob_pred_df = pd.DataFrame(list(lob_predictions.items()), columns=['LOB', 'Failure Probability'])
    lob_pred_df['Previous'] = [df[df['LOB'] == lob]['FailureFlag'].mean() * 100 for lob in lob_pred_df['LOB']]
    lob_pred_df['Change'] = lob_pred_df['Failure Probability'] - lob_pred_df['Previous']
    lob_pred_df['TrendIndicator'] = lob_pred_df['Change'].apply(
        lambda x: "↑" if x > 1 else "↓" if x < -1 else "→"
    )
    
    fig_lob_pred = go.Figure()
    fig_lob_pred.add_trace(go.Bar(
        x=lob_pred_df['LOB'],
        y=lob_pred_df['Failure Probability'],
        text=lob_pred_df.apply(
            lambda row: f"{row['Failure Probability']:.1f}% {row['TrendIndicator']}", 
            axis=1
        ),
        textposition='auto',
        marker_color='rgba(58, 71, 80, 0.6)'
    ))
    
    fig_lob_pred.update_layout(
        title='Predicted Failure Probability by LOB with Trend Indicators',
        xaxis_title='Line of Business',
        yaxis_title='Failure Probability (%)',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_lob_pred, use_container_width=True)

    # AI Analysis Section
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI Predictive Insights")
        
        # Get top 5 recent trends
        recent_trends = df.tail(5).groupby('Execution Date').agg({
            'Execution Status': lambda x: (x == 'Fail').mean() * 100
        })
        
        analysis_prompt = f"""
        Recent Failure Trends:
        {recent_trends.to_string()}
        
        Predicted Failure Probabilities by LOB:
        {lob_pred_df.head().to_string()}
        
        Please provide:
        1. Analysis of recent trends
        2. Key risk factors
        3. Prediction confidence assessment
        4. Recommended actions
        """
        
        with st.spinner("Generating predictive insights..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)

# Main application
st.title("🔍 QA Failure Analysis Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your QA execution data (CSV)", type=['csv'])

if uploaded_file is not None:
    # Load and validate data
    df = load_and_validate_data(uploaded_file)
    
    if df is not None:
        st.session_state.data = df
        
        # Display summary statistics with enhanced styling
        st.markdown("### 📊 Summary Statistics")
        stats = get_summary_stats(df)
        
        # Create metrics with custom styling and arrows
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Test Cases", stats['total_tests'])
        col2.metric(
            "Pass Rate", 
            f"{stats['pass_rate']:.1f}%",
            f"↑ {stats['pass_rate']:.1f}%",
            delta_color="normal"
        )
        col3.metric(
            "Fail Rate", 
            f"{stats['fail_rate']:.1f}%",
            f"↓ {stats['fail_rate']:.1f}%",
            delta_color="inverse"
        )
        col4.metric("Total Defects", stats['total_defects'])
        col5.metric("Unique Defects", stats['unique_defects'])
        
        # Add custom CSS for metric colors
        st.markdown("""
            <style>
            [data-testid="stMetricDelta"] {
                background-color: transparent !important;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Add spacing
        st.write("")
        
        # AI Analysis Button centered below metrics
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🤖 Analyze with AI", type="primary", use_container_width=True):
                st.session_state.ai_analysis = True
        
        # Add spacing after button
        st.write("")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Failure Analysis",
            "Failure Trends",
            "Gap Analysis",
            "LOB Analysis",
            "Predictive Analysis"
        ])
        
        with tab1:
            failure_analysis_tab(df)
        
        with tab2:
            trend_analysis_tab(df)
            
        with tab3:
            gap_analysis_tab(df)
            
        with tab4:
            lob_analysis_tab(df)
            
        with tab5:
            predictive_analysis_tab(df)
else:
    st.info("Please upload your qa_failure_analysis.csv file to begin the analysis.")
    
    # Display sample CSV format
    st.subheader("Required CSV Format:")
    sample_data = {
        'Execution Date': ['2024-01-01', '2024-01-01'],
        'User Story': ['Story 1', 'Story 2'],
        'Test Case ID': ['TC001', 'TC002'],
        'LOB': ['Banking', 'Insurance'],
        'Execution Status': ['Pass', 'Fail'],
        'Defect ID': ['', 'D_001'],
        'Defect Description': ['', 'Sample defect'],
        'Defect Type': ['', 'Automation'],
        'Defect Status': ['', 'Open'],
        'Severity': ['', 'High'],
        'Priority': ['', 'P1']
    }
    st.dataframe(pd.DataFrame(sample_data)) 