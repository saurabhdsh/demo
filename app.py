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
from azure.identity import ClientSecretCredential, get_bearer_token_provider
from azure.keyvault.secrets import SecretClient
from floating_prompt import add_floating_prompt_to_tab

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
try:
    # Connect to Key Vault
    KEY_VAULT_NAME = "kv-uais-nonprod"
    KV_URI = f"https://{KEY_VAULT_NAME}.vault.azure.net/"
    
    # Use environment variables for initial connection
    credential = ClientSecretCredential(
        tenant_id=os.getenv('APP_TENANT_ID'),
        client_id=os.getenv('APP_CLIENT_ID'),
        client_secret=os.getenv('APP_CLIENT_SECRET'),
        additionally_allowed_tenants=["*"]
    )
    
    # Create Key Vault client
    secret_client = SecretClient(
        vault_url=KV_URI,
        credential=credential
    )
    
    # Get Azure credentials from vault
    az_cred = ClientSecretCredential(
        tenant_id=secret_client.get_secret("secret-uais-tenant-id").value,
        client_id=secret_client.get_secret("secret-client-id-uais").value,
        client_secret=secret_client.get_secret("secret-client-secret-uais").value
    )
    
    # Get the bearer token provider
    token_provider = get_bearer_token_provider(az_cred, "https://cognitiveservices.azure.com/.default")
    
    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT = "https://prod-1.services.unitedaistudio.uhg.com/aoai-shared-openai-prod-1"
    DEPLOYMENT_NAME = "gpt-4o_2024-05-13"
    
    # Initialize the OpenAI client
    ai_client = openai.AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2024-06-01",
        azure_deployment=DEPLOYMENT_NAME,
        azure_ad_token_provider=token_provider,
        default_headers={
            "projectId": secret_client.get_secret("secret-client-project-uais").value
        }
    )
except Exception as e:
    st.error(f"Error initializing Azure OpenAI client: {str(e)}")
    ai_client = None

# Configure page settings
st.set_page_config(
    page_title="QA Failure Analysis Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = False

def get_ai_analysis(prompt):
    """Get AI analysis using Azure OpenAI API"""
    if ai_client is not None:
        try:
            # Get AI response with optimized system message
            response = ai_client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a QA expert analyzing test failure data. Provide clear, concise insights based on the metrics and data provided. Focus on actionable insights and specific recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500  # Reduced tokens for more focused responses
            )
            
            # Return the response content
            return response.choices[0].message.content
            
        except Exception as e:
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
        df['Execution Date'] = pd.to_datetime(df['Execution Date'], format='%d/%m/%y')
        
        # Fill NaN values appropriately
        df['Severity'].fillna('Medium', inplace=True)
        df['Priority'].fillna('P3', inplace=True)
        df['Defect Type'].fillna('Unknown', inplace=True)
        df['Defect Status'].fillna('Not Applicable', inplace=True)
        
        # For failed test cases without defect info, add placeholder
        failed_mask = (df['Execution Status'] == 'Fail') & (df['Defect ID'].isna())
        df.loc[failed_mask, 'Defect ID'] = df.loc[failed_mask, 'Test Case ID'].apply(lambda x: f'D_{x}')
        df.loc[failed_mask, 'Defect Description'] = 'Failure under investigation'
        df.loc[failed_mask, 'Defect Type'] = 'Unknown'
        df.loc[failed_mask, 'Defect Status'] = 'Open'
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_summary_stats(df):
    """Calculate summary statistics"""
    total_tests = len(df)
    pass_rate = (df['Execution Status'].value_counts().get('Pass', 0) / total_tests) * 100
    fail_rate = (df['Execution Status'].value_counts().get('Fail', 0) / total_tests) * 100
    total_defects = len(df[df['Execution Status'] == 'Fail'])
    unique_defects = df['Defect ID'].nunique()
    
    return {
        'total_tests': total_tests,
        'pass_rate': pass_rate,
        'fail_rate': fail_rate,
        'total_defects': total_defects,
        'unique_defects': unique_defects
    }

def failure_analysis_tab(df):
    """Tab 1: Failure Analysis"""
    st.header("🔍 Failure Analysis")
    
    # Add floating prompt
    add_floating_prompt_to_tab("failure", get_ai_analysis)
    
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
        lob_failures = failed_df.groupby('LOB').agg({
            'Test Case ID': 'count',
            'Defect Type': lambda x: ', '.join(x.unique()),
            'Severity': lambda x: ', '.join(x.unique())
        }).reset_index()
        lob_failures.columns = ['LOB', 'Count', 'Defect Types', 'Severity Levels']
        
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
    
    # Add floating prompt
    add_floating_prompt_to_tab("trend", get_ai_analysis)
    
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
        'Test Case ID': lambda x: ', '.join(x[x == 'Fail'].unique()[:5]),
        'Defect Description': lambda x: ', '.join(x[x.notna()].unique()[:3])
    }).reset_index()
    
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
        # Calculate and plot failure rate
        daily_stats['Total Tests'] = filtered_df.groupby('Execution Date').size().values
        daily_stats['Failure Rate'] = (daily_stats['Execution Status'] / daily_stats['Total Tests'] * 100).round(2)
        
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
    
    # Add floating prompt
    add_floating_prompt_to_tab("gap", get_ai_analysis)
    
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
    """Tab 4: LOB Analysis"""
    st.header("📊 LOB Analysis")
    
    # Add floating prompt
    add_floating_prompt_to_tab("lob", get_ai_analysis)
    
    # Add weekend vs weekday analysis
    df['DayOfWeek'] = df['Execution Date'].dt.day_name()
    df['IsWeekend'] = df['Execution Date'].dt.dayofweek.isin([5, 6])
    
    # LOB selector
    selected_lob = st.selectbox("Select LOB for Detailed Analysis", df['LOB'].unique())
    
    lob_df = df[df['LOB'] == selected_lob]
    failed_lob_df = lob_df[lob_df['Execution Status'] == 'Fail']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Failure rate trend
        daily_stats = lob_df.groupby('Execution Date').agg({
            'Execution Status': lambda x: (x == 'Fail').mean() * 100,
            'Test Case ID': lambda x: x[x == 'Fail'].nunique(),
            'Defect Type': lambda x: ', '.join(x[x != ''].unique())
        }).reset_index()
        
        fig_trend = px.line(
            daily_stats,
            x='Execution Date',
            y='Execution Status',
            title=f'Daily Failure Rate Trend for {selected_lob}',
            labels={'Execution Status': 'Failure Rate (%)', 'Execution Date': 'Date'}
        )
        fig_trend.update_layout(template='plotly_white')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Weekend vs Weekday Analysis
        weekend_stats = failed_lob_df.groupby(['IsWeekend', 'Defect Type']).size().reset_index(name='Count')
        weekend_stats['Day Type'] = weekend_stats['IsWeekend'].map({True: 'Weekend', False: 'Weekday'})
        
        fig_weekend = px.bar(
            weekend_stats,
            x='Day Type',
            y='Count',
            color='Defect Type',
            title=f'Weekend vs Weekday Analysis for {selected_lob}',
            labels={'Count': 'Number of Failures'}
        )
        fig_weekend.update_layout(template='plotly_white')
        st.plotly_chart(fig_weekend, use_container_width=True)

    # Defect Analysis
    st.subheader("📊 Defect Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Defect type distribution
        defect_dist = failed_lob_df.groupby(['Defect Type', 'Severity']).size().reset_index(name='Count')
        
        fig_defect = px.bar(
            defect_dist,
            x='Defect Type',
            y='Count',
            color='Severity',
            title=f'Defect Distribution for {selected_lob}',
            labels={'Count': 'Number of Defects'}
        )
        fig_defect.update_layout(template='plotly_white')
        st.plotly_chart(fig_defect, use_container_width=True)
    
    with col4:
        # Status distribution
        status_dist = failed_lob_df.groupby(['Defect Status', 'Priority']).size().reset_index(name='Count')
        
        fig_status = px.bar(
            status_dist,
            x='Defect Status',
            y='Count',
            color='Priority',
            title=f'Defect Status Distribution for {selected_lob}',
            labels={'Count': 'Number of Defects'}
        )
        fig_status.update_layout(template='plotly_white')
        st.plotly_chart(fig_status, use_container_width=True)

    # Test Case Analysis
    st.subheader("🔍 Test Case Analysis")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Top failing test cases
        test_failures = failed_lob_df.groupby(['Test Case ID', 'Severity']).size().reset_index(name='Count')
        test_failures = test_failures.nlargest(10, 'Count')
        
        fig_test = px.bar(
            test_failures,
            x='Test Case ID',
            y='Count',
            color='Severity',
            title=f'Top 10 Failing Test Cases in {selected_lob}',
            labels={'Count': 'Number of Failures'}
        )
        fig_test.update_layout(template='plotly_white')
        st.plotly_chart(fig_test, use_container_width=True)
    
    with col6:
        # Failure patterns by day
        day_patterns = failed_lob_df.groupby(['DayOfWeek', 'Defect Type']).size().reset_index(name='Count')
        
        fig_day = px.bar(
            day_patterns,
            x='DayOfWeek',
            y='Count',
            color='Defect Type',
            title=f'Failure Patterns by Day for {selected_lob}',
            labels={'Count': 'Number of Failures'}
        )
        fig_day.update_layout(template='plotly_white')
        st.plotly_chart(fig_day, use_container_width=True)

    # Severity and Priority Analysis
    st.subheader("⚠️ Severity and Priority Analysis")
    
    col7, col8 = st.columns(2)
    
    with col7:
        # Severity trends
        severity_trend = failed_lob_df.groupby(['Execution Date', 'Severity']).size().reset_index(name='Count')
        
        fig_severity = px.line(
            severity_trend,
            x='Execution Date',
            y='Count',
            color='Severity',
            title=f'Severity Trends for {selected_lob}',
            labels={'Count': 'Number of Defects'}
        )
        fig_severity.update_layout(template='plotly_white')
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col8:
        # Priority distribution
        priority_dist = failed_lob_df.groupby(['Priority', 'Severity']).size().reset_index(name='Count')
        
        fig_priority = px.bar(
            priority_dist,
            x='Priority',
            y='Count',
            color='Severity',
            title=f'Priority Distribution for {selected_lob}',
            labels={'Count': 'Number of Defects'}
        )
        fig_priority.update_layout(template='plotly_white')
        st.plotly_chart(fig_priority, use_container_width=True)

def predictive_analysis_tab(df):
    """Tab 5: Predictive Analysis"""
    st.header("🔮 Predictive Analysis")
    
    # Add floating prompt
    add_floating_prompt_to_tab("predictive", get_ai_analysis)
    
    # Prepare data for prediction
    df['DaysSinceStart'] = (df['Execution Date'] - df['Execution Date'].min()).dt.days
    df['FailureFlag'] = (df['Execution Status'] == 'Fail').astype(int)
    
    # Group by date and calculate failure rate
    daily_stats = df.groupby('Execution Date').agg({
        'FailureFlag': 'mean',
        'Test Case ID': lambda x: x[df['Execution Status'] == 'Fail'].nunique(),
        'Defect Type': lambda x: x[df['Execution Status'] == 'Fail'].unique()
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
    
    # Failure probability by LOB
    st.subheader("Failure Probability by LOB")
    
    # Calculate failure probabilities for each LOB
    lob_predictions = {}
    lob_details = {}
    
    for lob in df['LOB'].unique():
        lob_df = df[df['LOB'] == lob]
        if len(lob_df) > 0:
            # Calculate daily stats for this LOB
            lob_stats = lob_df.groupby('Execution Date').agg({
                'FailureFlag': 'mean',
                'Test Case ID': lambda x: x[lob_df['Execution Status'] == 'Fail'].nunique(),
                'Defect Type': lambda x: x[lob_df['Execution Status'] == 'Fail'].unique()
            }).reset_index()
            
            # Prepare data for prediction
            X_lob = np.array(range(len(lob_stats))).reshape(-1, 1)
            y_lob = lob_stats['FailureFlag'].values
            
            # Train model for this LOB
            model_lob = LinearRegression()
            model_lob.fit(X_lob, y_lob)
            
            # Predict next day
            next_day_pred = model_lob.predict([[len(X_lob)]])[0]
            lob_predictions[lob] = next_day_pred * 100
            
            # Store additional details
            lob_details[lob] = {
                'recent_failures': len(lob_df[lob_df['Execution Status'] == 'Fail'].tail(7)),
                'total_failures': len(lob_df[lob_df['Execution Status'] == 'Fail']),
                'defect_types': lob_df[lob_df['Execution Status'] == 'Fail']['Defect Type'].value_counts().to_dict()
            }
    
    # Create LOB predictions visualization
    lob_pred_df = pd.DataFrame(list(lob_predictions.items()), columns=['LOB', 'Failure Probability'])
    lob_pred_df['Previous'] = [df[df['LOB'] == lob]['FailureFlag'].mean() * 100 for lob in lob_pred_df['LOB']]
    lob_pred_df['Change'] = lob_pred_df['Failure Probability'] - lob_pred_df['Previous']
    lob_pred_df['TrendIndicator'] = lob_pred_df['Change'].apply(
        lambda x: "↑" if x > 1 else "↓" if x < -1 else "→"
    )
    
    # Create visualization
    fig_lob_pred = go.Figure()
    
    # Add bars for each LOB
    fig_lob_pred.add_trace(go.Bar(
        x=lob_pred_df['LOB'],
        y=lob_pred_df['Failure Probability'],
        text=lob_pred_df.apply(
            lambda row: f"{row['Failure Probability']:.1f}% {row['TrendIndicator']}", 
            axis=1
        ),
        textposition='auto',
        marker_color=lob_pred_df['Change'].apply(
            lambda x: 'rgba(255, 99, 71, 0.7)' if x > 0 
            else 'rgba(60, 179, 113, 0.7)' if x < 0 
            else 'rgba(128, 128, 128, 0.7)'
        )
    ))
    
    fig_lob_pred.update_layout(
        title='Predicted Failure Probability by LOB with Trend Indicators',
        xaxis_title='Line of Business',
        yaxis_title='Failure Probability (%)',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_lob_pred, use_container_width=True)
    
    # Display detailed predictions
    st.subheader("Detailed Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Defect type prediction
        defect_trend = df[df['Execution Status'] == 'Fail'].groupby(['Execution Date', 'Defect Type']).size().reset_index(name='Count')
        
        fig_defect = px.line(
            defect_trend,
            x='Execution Date',
            y='Count',
            color='Defect Type',
            title='Defect Type Trends and Predictions',
            labels={'Count': 'Number of Defects'}
        )
        fig_defect.update_layout(template='plotly_white')
        st.plotly_chart(fig_defect, use_container_width=True)
    
    with col2:
        # Severity prediction
        severity_trend = df[df['Execution Status'] == 'Fail'].groupby(['Execution Date', 'Severity']).size().reset_index(name='Count')
        
        fig_severity = px.line(
            severity_trend,
            x='Execution Date',
            y='Count',
            color='Severity',
            title='Severity Trends and Predictions',
            labels={'Count': 'Number of Defects'}
        )
        fig_severity.update_layout(template='plotly_white')
        st.plotly_chart(fig_severity, use_container_width=True)

def root_cause_analysis_tab(df):
    """Tab 6: Root Cause Analysis"""
    st.header("🔍 Root Cause Analysis")
    
    # Group defects by type and description for analysis
    defect_analysis = df[df['Execution Status'] == 'Fail'].groupby(
        ['Defect Type', 'Test Case ID', 'Test Case Name', 'Defect Description', 'LOB']
    ).size().reset_index(name='occurrence_count')
    
    # Create defect type filter
    selected_defect_type = st.selectbox(
        "Select Defect Type for Analysis",
        options=sorted(defect_analysis['Defect Type'].unique())
    )
    
    # Filter data by selected defect type
    filtered_defects = defect_analysis[defect_analysis['Defect Type'] == selected_defect_type]
    
    # Display defect distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Test case distribution for selected defect type
        test_case_dist = filtered_defects.groupby(['Test Case ID', 'Test Case Name']).agg({
            'occurrence_count': 'sum',
            'Defect Description': lambda x: '<br>'.join(x.unique())
        }).reset_index()
        
        fig_test_cases = px.bar(
            test_case_dist,
            x='Test Case ID',
            y='occurrence_count',
            title=f'Test Cases Affected by {selected_defect_type}',
            labels={'occurrence_count': 'Number of Occurrences', 'Test Case ID': 'Test Case'},
            custom_data=['Test Case Name', 'Defect Description']
        )
        fig_test_cases.update_layout(template='plotly_white')
        fig_test_cases.update_traces(
            hovertemplate="<b>Test Case:</b> %{x}<br>" +
                         "<b>Name:</b> %{customdata[0]}<br>" +
                         "<b>Occurrences:</b> %{y}<br>" +
                         "<b>Defect Details:</b><br>%{customdata[1]}"
        )
        st.plotly_chart(fig_test_cases)
    
    with col2:
        # LOB distribution for selected defect type
        lob_dist = filtered_defects.groupby('LOB').agg({
            'occurrence_count': 'sum',
            'Test Case Name': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else '')
        }).reset_index()
        
        fig_lob = px.pie(
            lob_dist,
            values='occurrence_count',
            names='LOB',
            title=f'LOB Distribution for {selected_defect_type}',
            custom_data=['Test Case Name']
        )
        fig_lob.update_layout(template='plotly_white')
        fig_lob.update_traces(
            textposition='outside',
            textinfo='percent+label',
            hovertemplate="<b>LOB:</b> %{label}<br>" +
                         "<b>Count:</b> %{value}<br>" +
                         "<b>Affected Test Cases:</b><br>%{customdata[0]}"
        )
        st.plotly_chart(fig_lob)
    
    # Detailed defect pattern analysis
    st.subheader("📋 Detailed Defect Pattern Analysis")
    
    # Create a table of unique defect descriptions and their impact
    defect_patterns = filtered_defects.groupby('Defect Description').agg({
        'Test Case ID': lambda x: ', '.join(x.unique()),
        'Test Case Name': lambda x: ', '.join(x.unique()),
        'LOB': lambda x: ', '.join(x.unique()),
        'occurrence_count': 'sum'
    }).reset_index()
    
    # Sort by occurrence count
    defect_patterns = defect_patterns.sort_values('occurrence_count', ascending=False)
    
    # Display the patterns in an expandable section
    with st.expander("View Detailed Defect Patterns", expanded=True):
        # Create a formatted display of the patterns without using style
        st.dataframe(
            defect_patterns.sort_values('occurrence_count', ascending=False),
            use_container_width=True
        )
    
    # AI Root Cause Analysis
    st.subheader("🤖 AI Root Cause Analysis")
    
    # Prepare data for AI analysis
    defect_context = filtered_defects.to_dict('records')
    
    analysis_prompt = f"""
    Analyze the following defect patterns for {selected_defect_type} issues:
    
    Defect Details:
    {defect_patterns.to_string()}
    
    Test Case Context:
    {filtered_defects[['Test Case ID', 'Test Case Name', 'Defect Description']].to_string()}
    
    Please provide a detailed root cause analysis including:
    1. Common patterns and underlying causes
    2. Technical dependencies and environmental factors
    3. Impact assessment on different LOBs
    4. Systemic issues vs isolated incidents
    5. Potential preventive measures
    6. Specific recommendations for each pattern
    7. Priority assessment for fixes
    8. Risk of recurrence
    9. Long-term mitigation strategies
    10. Best practices to prevent similar issues
    
    Focus on providing actionable insights and specific technical recommendations.
    """
    
    if st.button("Generate Root Cause Analysis", type="primary"):
        with st.spinner("Analyzing defect patterns..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)
                
                # Add a download button for the analysis
                st.download_button(
                    label="Download Analysis Report",
                    data=ai_insights,
                    file_name=f"root_cause_analysis_{selected_defect_type}.md",
                    mime="text/markdown"
                )

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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Failure Analysis",
            "Failure Trends",
            "Gap Analysis",
            "LOB Analysis",
            "Predictive Analysis",
            "Root Cause Analysis"
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
            
        with tab6:
            root_cause_analysis_tab(df)
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
        'Defect Status': ['', 'Open']
    }
    st.dataframe(pd.DataFrame(sample_data)) 