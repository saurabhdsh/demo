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
from typing import Optional, Callable

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

# Initialize session state variables
if 'prompt' not in st.session_state:
    st.session_state.prompt = False
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = False
if 'data' not in st.session_state:
    st.session_state.data = None

def get_ai_analysis(prompt):
    """Get AI analysis using Azure OpenAI API"""
    if ai_client is not None and st.session_state.data is not None:
        try:
            # Create context for the AI
            if st.session_state.prompt:
                # Get the current data
                df = st.session_state.data
                failed_df = df[df['Execution Status'] == 'Fail']
               
                # Calculate key metrics for context
                total_tests = len(df)
                total_failures = len(failed_df)
                failure_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0
               
                # Calculate distributions
                defect_dist = failed_df['Defect Type'].value_counts()
                severity_dist = failed_df['Severity'].value_counts()
                priority_dist = failed_df['Priority'].value_counts()
                status_dist = failed_df['Defect Status'].value_counts()
                lob_dist = failed_df['LOB'].value_counts()
                app_dist=failed_df['Application'].value_counts()
                context = f"""Current metrics:
                - Total Tests: {total_tests}
                - Failed Tests: {total_failures}
                - Failure Rate: {failure_rate:.2f}%
               
                Defect Distribution:
                {defect_dist.to_string()}
               
                Severity Distribution:
                {severity_dist.to_string()}
               
                Priority Distribution:
                {priority_dist.to_string()}
               
                Status Distribution:
                {status_dist.to_string()}
               
                LOB Distribution:
                {lob_dist.to_string()}
 
                Application Distribution:
                {app_dist.to_string()}
               
                Based the data provided, answer the user question to best of your ability. Keep in mind the """
                st.session_state.prompt=False
            else:
                context="You are a QA expert analyzing test failure data"
           
            # Get AI response with optimized system message
            response = ai_client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
           
            # Return the response content
            return response.choices[0].message.content
           
        except Exception as e:
            st.error(f"Error generating AI analysis: {str(e)}")
            return None
    else:
        st.warning("Azure OpenAI client not initialized or no data loaded. Please check your configuration and upload data.")
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
    """Tab 1: Failure Analysis with Interactive Prompting"""
    st.header("🔍 Failure Analysis")
   
    # Add floating prompt
    st.session_state.prompt=True
    add_floating_prompt_to_tab("failure", get_ai_analysis)
   
    # Add IsWeekend column at the start
    df['IsWeekend'] = df['Execution Date'].dt.dayofweek.isin([5, 6])
    df['DayOfWeek'] = df['Execution Date'].dt.day_name()
   
    # Modern styling configuration
    chart_config = {
        'template': 'plotly_white',
        'font': dict(family="Arial, sans-serif"),
        'title_font_size': 20,
        'showlegend': True
    }
   
    col1, col2 = st.columns(2)
   
    with col1:
        # Enhanced Pass/Fail Distribution with hover data
        status_counts = df['Execution Status'].value_counts()
        total_tests = len(df)
       
        # Create a DataFrame for the pie chart
        status_df = pd.DataFrame({
            'Status': status_counts.index,
            'Count': status_counts.values,
            'Percentage': [f"{count} tests ({count/total_tests*100:.1f}%)" for count in status_counts.values]
        })
       
        fig_status = px.pie(
            data_frame=status_df,
            values='Count',
            names='Status',
            title='Overall Pass/Fail Distribution',
            color_discrete_sequence=['#00CC96', '#EF553B'],
            hole=0.4,
            custom_data=['Percentage']
        )
        fig_status.update_layout(**chart_config)
        fig_status.update_traces(
            textposition='outside',
            textinfo='percent+label',
            hovertemplate="Status: %{label}<br>%{customdata[0]}"
        )
        st.plotly_chart(fig_status)
   
    with col2:
        # Enhanced LOB-wise Failure Breakdown with hover data
        lob_failures = df[df['Execution Status'] == 'Fail'].groupby('LOB').agg({
            'Test Case ID': 'count',
            'Test Case Name': lambda x: '<br>'.join(x.unique()[:5]) + ('...' if len(x.unique()) > 5 else '')
        }).reset_index()
       
        fig_lob = px.pie(
            lob_failures,
            values='Test Case ID',
            names='LOB',
            title='LOB-wise Failure Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4,
            custom_data=[lob_failures['Test Case Name']]
        )
        fig_lob.update_layout(**chart_config)
        fig_lob.update_traces(
            textposition='outside',
            textinfo='percent+label',
            hovertemplate="LOB: %{label}<br>Failed Tests: %{value}<br>Sample Test Cases:<br>%{customdata[0]}"
        )
        st.plotly_chart(fig_lob)
 
    # Add detailed failure pattern analysis
    st.subheader("📊 Detailed Failure Pattern Analysis")
   
    # Calculate defect type distribution with test case details
    defect_type_dist = df[df['Execution Status'] == 'Fail'].groupby('Defect Type').agg({
        'Test Case ID': 'count',
        'Test Case Name': lambda x: '<br>'.join(x.unique()[:5]) + ('...' if len(x.unique()) > 5 else ''),
        'Defect Description': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else '')
    }).reset_index()
   
    # Create pattern analysis columns
    col3, col4 = st.columns(2)
   
    with col3:
        # Enhanced defect type distribution with hover data
        fig_defect_type = px.bar(
            defect_type_dist,
            x='Defect Type',
            y='Test Case ID',
            title='Defect Type Distribution',
            labels={'Test Case ID': 'Number of Failures', 'Defect Type': 'Type of Defect'},
            color='Defect Type',
            color_discrete_sequence=px.colors.qualitative.Set3,
            custom_data=[defect_type_dist['Test Case Name'], defect_type_dist['Defect Description']]
        )
        fig_defect_type.update_layout(template='plotly_white')
        fig_defect_type.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "Failures: %{y}<br>" +
                         "<b>Sample Test Cases:</b><br>%{customdata[0]}<br>" +
                         "<b>Sample Defects:</b><br>%{customdata[1]}"
        )
        st.plotly_chart(fig_defect_type)
   
    with col4:
        # Enhanced top failing test cases by defect type
        top_failing_cases = df[df['Execution Status'] == 'Fail'].groupby(
            ['Test Case ID', 'Test Case Name', 'Defect Type']
        ).size().reset_index(name='count')
        top_failing_cases = top_failing_cases.nlargest(10, 'count')
       
        fig_top_cases = px.bar(
            top_failing_cases,
            x='Test Case ID',
            y='count',
            color='Defect Type',
            title='Top 10 Failing Test Cases by Defect Type',
            labels={'count': 'Number of Failures', 'Test Case ID': 'Test Case'},
            color_discrete_sequence=px.colors.qualitative.Set3,
            custom_data=[top_failing_cases['Test Case Name']]
        )
        fig_top_cases.update_layout(template='plotly_white')
        fig_top_cases.update_traces(
            hovertemplate="<b>Test Case:</b> %{x}<br>" +
                         "<b>Name:</b> %{customdata[0]}<br>" +
                         "Failures: %{y}<br>" +
                         "Type: %{color}"
        )
        st.plotly_chart(fig_top_cases)
   
    # Create heatmap data
    issue_types = sorted(df[df['Execution Status'] == 'Fail']['Defect Type'].unique())
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['DayOfWeek'] = df['Execution Date'].dt.day_name()
    heatmap_data = []
    for issue_type in issue_types:
        for day in days:
            # Get failed test cases for this day and issue type
            failed_tests = df[
                (df['Execution Status'] == 'Fail') &
                (df['Defect Type'] == issue_type) &
                (df['DayOfWeek'] == day)
            ]
            count = len(failed_tests)
            test_cases = '<br>'.join(failed_tests['Test Case Name'].unique())
            heatmap_data.append({
                'Issue Type': issue_type,
                'Day': day,
                'Count': count,
                'Test Cases': test_cases
            })
   
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot(
        index='Issue Type',
        columns='Day',
        values='Count'
    ).fillna(0)
   
    test_cases_pivot = heatmap_df.pivot(
        index='Issue Type',
        columns='Day',
        values='Test Cases'
    ).fillna('')
   
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=days,
        y=issue_types,
        colorscale='RdYlBu_r',
        text=heatmap_pivot.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate="<b>Issue Type:</b> %{y}<br>" +
                     "<b>Day:</b> %{x}<br>" +
                     "<b>Count:</b> %{z}<br>" +
                     "<b>Failed Test Cases:</b><br>%{customdata}<extra></extra>",
        customdata=test_cases_pivot.values
    ))
   
    fig_heatmap.update_layout(
        title={'text': '🔥 Failure Distribution by Day and Issue Type','font': {'size': 24}},
        xaxis_title='Day of Week',
        yaxis_title='Issue Type',
        template='plotly_white'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
 
    # Add Test Cases vs Issue Types heatmap
    failed_tests = df[df['Execution Status'] == 'Fail']
   
    # Create pivot table for test cases vs issue types
    test_issue_pivot = pd.crosstab(
        failed_tests['Test Case Name'],
        failed_tests['Defect Type']
    )
   
    # Sort test cases by total failures
    test_issue_pivot['Total'] = test_issue_pivot.sum(axis=1)
    test_issue_pivot = test_issue_pivot.sort_values('Total', ascending=False).head(15)  # Show top 15 test cases
    test_issue_pivot = test_issue_pivot.drop('Total', axis=1)
   
    # Get defect descriptions for hover data
    defect_desc_pivot = pd.pivot_table(
        failed_tests,
        index='Test Case Name',
        columns='Defect Type',
        values='Defect Description',
        aggfunc=lambda x: '<br>'.join(x.unique())
    ).fillna('')
   
    # Filter to match test_issue_pivot
    defect_desc_pivot = defect_desc_pivot.loc[test_issue_pivot.index]
   
    fig_test_issues = go.Figure(data=go.Heatmap(
        z=test_issue_pivot.values,
        x=test_issue_pivot.columns,
        y=test_issue_pivot.index,
        colorscale='YlOrRd',
        text=test_issue_pivot.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate="<b>Test Case:</b> %{y}<br>" +
                     "<b>Issue Type:</b> %{x}<br>" +
                     "<b>Count:</b> %{z}<br>" +
                     "<b>Defect Details:</b><br>%{customdata}<extra></extra>",
        customdata=defect_desc_pivot.values
    ))
   
    fig_test_issues.update_layout(
        title={'text': '🎯 Top 15 Failing Test Cases by Issue Type','font': {'size': 24}},
        xaxis_title='Issue Type',
        yaxis_title='Test Case Name',
        template='plotly_white',
        height=600,  # Make it taller to accommodate more test cases
        xaxis={'tickangle': -45}  # Angle the x-axis labels for better readability
    )
    st.plotly_chart(fig_test_issues, use_container_width=True)
 
    # AI Analysis Section with standardized template
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI Analysis")
       
        # Get detailed failure patterns
        automation_issues = df[
            (df['Execution Status'] == 'Fail') &
            (df['Defect Type'].str.contains('Data|Environment', na=False))
        ].groupby(['Test Case Name', 'Defect Description', 'DayOfWeek']).size().nlargest(5)
       
        manual_issues = df[
            (df['Execution Status'] == 'Fail') &
            (df['Defect Type'].str.contains('UI|Functional', na=False))
        ].groupby(['Test Case Name', 'Defect Description', 'DayOfWeek']).size().nlargest(5)
       
        # Get LOB-wise failure distribution
        lob_failures = df[df['Execution Status'] == 'Fail'].groupby('LOB').size()
        most_affected_lob = lob_failures.idxmax()
       
        # Get defect type distribution
        defect_type_dist = df[df['Execution Status'] == 'Fail'].groupby('Defect Type').size()
       
        # Calculate test case stability
        test_stability = df.groupby('Test Case Name').agg({
            'Execution Status': lambda x: (x == 'Pass').mean() * 100
        }).sort_values('Execution Status')
       
        # Get defect status distribution
        defect_status_dist = df[df['Execution Status'] == 'Fail'].groupby('Defect Status').size()
       
        analysis_prompt = f"""
        Based on the actual test execution data, provide a detailed analysis:
 
        Test Execution Summary:
        - Total Test Cases: {len(df)}
        - Failed Test Cases: {len(df[df['Execution Status'] == 'Fail'])}
        - Most Affected LOB: {most_affected_lob}
       
        Defect Distribution:
        {defect_type_dist.to_string()}
       
        Top 5 Most Critical Issues:
        {automation_issues.to_string()}
       
        Top 5 UI/Functional Issues:
        {manual_issues.to_string()}
       
        Defect Status Overview:
        {defect_status_dist.to_string()}
       
        Least Stable Test Cases (Bottom 5):
        {test_stability.head().to_string()}
 
        Please provide:
        1. Critical Issue Analysis
           - Identify patterns in the most frequent failures
           - Analyze root causes based on defect descriptions
           - Assess impact on different LOBs
       
        2. Test Case Stability Assessment
           - Evaluate patterns in unstable test cases
           - Identify common factors in failing scenarios
           - Suggest improvements for test reliability
       
        3. Defect Management Insights
           - Analyze defect resolution patterns
           - Identify bottlenecks in defect lifecycle
           - Recommend process improvements
       
        4. Risk Assessment
           - Evaluate impact on business functionality
           - Identify high-risk areas needing immediate attention
           - Suggest preventive measures
       
        5. Actionable Recommendations
           - Specific steps for improving test stability
           - Process improvements for defect management
           - Test case enhancement suggestions
           - Resource allocation recommendations
       
        Focus on providing data-driven insights and specific, actionable recommendations based on the actual test results.
        """
       
        with st.spinner("Generating comprehensive analysis..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)
 
def trend_analysis_tab(df):
    """Tab 2: Failure Trends Over Time"""
    st.header("📈 Failure Trends Over Time")
   
    # Add floating prompt
    st.session_state.prompt=True
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
   
    # Enhanced time series of failures with hover data
    daily_stats = filtered_df.groupby(['Execution Date']).agg({
        'Execution Status': lambda x: (x == 'Fail').sum(),
        'Test Case Name': lambda x: '<br>'.join(x[x == 'Fail'].unique()[:5]) + ('...' if len(x[x == 'Fail'].unique()) > 5 else ''),
        'Defect Description': lambda x: '<br>'.join(x[x == 'Fail'].unique()[:3]) + ('...' if len(x[x == 'Fail'].unique()) > 3 else '')
    }).reset_index()
   
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_stats['Execution Date'],
        y=daily_stats['Execution Status'],
        mode='lines+markers',
        name='Failures',
        line=dict(dash='dot'),
        hovertemplate="<b>Date:</b> %{x}<br>" +
                     "<b>Failures:</b> %{y}<br>" +
                     "<b>Failed Test Cases:</b><br>%{customdata[0]}<br>" +
                     "<b>Defects:</b><br>%{customdata[1]}"
    ))
   
    fig_trend.update_layout(
        title='Daily Failure Trend with Test Case Details',
        xaxis_title='Date',
        yaxis_title='Number of Failures',
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)
   
    # Enhanced defect type trends with details
    col1, col2 = st.columns(2)
   
    with col1:
        defect_type_trend = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['Execution Date', 'Defect Type']
        ).agg({
            'Test Case ID': 'count',
            'Test Case Name': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else ''),
            'Defect Description': lambda x: '<br>'.join(x.unique()[:2]) + ('...' if len(x.unique()) > 2 else '')
        }).reset_index()
       
        fig_defect_trend = px.line(
            defect_type_trend,
            x='Execution Date',
            y='Test Case ID',
            color='Defect Type',
            title='Defect Type Trends Over Time',
            labels={'Test Case ID': 'Number of Failures', 'Execution Date': 'Date'},
            custom_data=['Test Case Name', 'Defect Description']
        )
        fig_defect_trend.update_layout(template='plotly_white')
        fig_defect_trend.update_traces(
            hovertemplate="<b>Date:</b> %{x}<br>" +
                         "<b>Failures:</b> %{y}<br>" +
                         "<b>Failed Test Cases:</b><br>%{customdata[0]}<br>" +
                         "<b>Defects:</b><br>%{customdata[1]}"
        )
        st.plotly_chart(fig_defect_trend)
   
    with col2:
        # Weekly pattern analysis with test case details
        filtered_df['DayOfWeek'] = filtered_df['Execution Date'].dt.day_name()
        weekly_pattern = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['DayOfWeek', 'Defect Type']
        ).agg({
            'Test Case ID': 'count',
            'Test Case Name': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else '')
        }).reset_index()
       
        fig_weekly = px.bar(
            weekly_pattern,
            x='DayOfWeek',
            y='Test Case ID',
            color='Defect Type',
            title='Weekly Failure Patterns by Defect Type',
            labels={'Test Case ID': 'Number of Failures', 'DayOfWeek': 'Day of Week'},
            custom_data=['Test Case Name']
        )
        fig_weekly.update_layout(template='plotly_white')
        fig_weekly.update_traces(
            hovertemplate="<b>Day:</b> %{x}<br>" +
                         "<b>Failures:</b> %{y}<br>" +
                         "<b>Failed Test Cases:</b><br>%{customdata[0]}"
        )
        st.plotly_chart(fig_weekly)
 
    # AI Analysis Section
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI Trend Analysis")
       
        # Calculate trend metrics
        total_failures = len(filtered_df[filtered_df['Execution Status'] == 'Fail'])
        failure_by_type = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby('Defect Type').size()
       
        # Get top recurring defects with test case names
        recurring_defects = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['Test Case Name', 'Defect Type', 'Defect Description']
        ).size().nlargest(5)
       
        # Calculate weekly patterns with test case details
        weekly_stats = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['DayOfWeek', 'Test Case Name', 'Defect Type', 'Defect Description']
        ).size().nlargest(5)
       
        # Get recent trend (last 7 days) with test case details
        recent_trend = filtered_df[
            filtered_df['Execution Date'] >= filtered_df['Execution Date'].max() - pd.Timedelta(days=7)
        ]
        recent_failures = recent_trend[recent_trend['Execution Status'] == 'Fail'].groupby(
            ['Execution Date', 'Test Case Name', 'Defect Type', 'Defect Description']
        ).size()
       
        # Calculate failure rate trends
        daily_failure_rates = filtered_df.groupby('Execution Date').agg({
            'Execution Status': lambda x: (x == 'Fail').mean() * 100
        }).sort_index()
       
        # Calculate trend direction
        trend_direction = 'increasing' if daily_failure_rates['Execution Status'].iloc[-1] > daily_failure_rates['Execution Status'].iloc[0] else 'decreasing'
       
        analysis_prompt = f"""
        Based on the actual test execution data for the selected period, provide a detailed trend analysis:
 
        Overall Trend Summary:
        - Total Failures: {total_failures}
        - Trend Direction: {trend_direction}
        - Latest Failure Rate: {daily_failure_rates['Execution Status'].iloc[-1]:.1f}%
       
        Failure Distribution by Type:
        {failure_by_type.to_string()}
       
        Top 5 Most Recurring Issues:
        {recurring_defects.to_string()}
       
        Critical Weekly Patterns:
        {weekly_stats.to_string()}
       
        Recent 7-Day Trend:
        {recent_failures.to_string()}
       
        Please provide:
        1. Trend Pattern Analysis
           - Analyze the overall failure rate trend
           - Identify specific test cases showing deteriorating performance
           - Highlight any cyclical patterns in failures
           - Correlate failures with specific days/times
       
        2. Defect Evolution Analysis
           - Track how defect patterns have changed over time
           - Identify persistent vs. newly emerging issues
           - Analyze defect resolution velocity
           - Highlight recurring patterns in specific test cases
       
        3. Impact Assessment
           - Evaluate the business impact of identified trends
           - Analyze the effectiveness of recent fixes
           - Identify areas showing improvement vs. degradation
           - Assess the stability of critical test cases
       
        4. Root Cause Analysis
           - Identify common factors in recurring failures
           - Analyze environmental or timing-related patterns
           - Evaluate test data dependencies
           - Assess infrastructure-related trends
       
        5. Predictive Insights
           - Forecast potential future issues based on trends
           - Identify test cases at risk of failure
           - Suggest preventive measures
           - Recommend monitoring focus areas
       
        6. Actionable Recommendations
           - Specific steps to address deteriorating trends
           - Test case improvement suggestions
           - Process enhancement recommendations
           - Resource allocation guidance
       
        Focus on providing data-driven insights and specific, actionable recommendations based on the actual trend data.
        Highlight any patterns that require immediate attention and suggest proactive measures to prevent future failures.
        """
       
        with st.spinner("Generating trend analysis..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)
 
def gap_analysis_tab(df):
    """Tab 3: Gap Analysis"""
    st.header("🔍 Gap Analysis")
   
    # Add floating prompt
    st.session_state.prompt=True
    add_floating_prompt_to_tab("gap", get_ai_analysis)
   
    col1, col2 = st.columns(2)
   
    with col1:
        # Enhanced top failing test cases with names
        test_case_failures = df[df['Execution Status'] == 'Fail'].groupby(
            ['Test Case ID', 'Test Case Name']
        ).size().reset_index(name='failure_count')
        test_case_failures = test_case_failures.nlargest(10, 'failure_count')
       
        fig_test_cases = px.bar(
            test_case_failures,
            x='Test Case ID',
            y='failure_count',
            title='Top 10 Frequently Failing Test Cases',
            labels={'failure_count': 'Failure Count', 'Test Case ID': 'Test Case'},
            custom_data=['Test Case Name']
        )
        fig_test_cases.update_layout(template='plotly_white')
        fig_test_cases.update_traces(
            hovertemplate="<b>Test Case:</b> %{x}<br>" +
                         "<b>Name:</b> %{customdata[0]}<br>" +
                         "<b>Failures:</b> %{y}"
        )
        st.plotly_chart(fig_test_cases)
   
    with col2:
        # Enhanced Application analysis with test case details
        story_failures = df[df['Execution Status'] == 'Fail'].groupby(
            ['Application']
        ).agg({
            'Test Case ID': 'count',
            'Test Case Name': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else '')
        }).reset_index()
        story_failures = story_failures.nlargest(10, 'Test Case ID')
       
        fig_stories = px.bar(
            story_failures,
            x='Application',
            y='Test Case ID',
            title='Top 10 Applications with Most Failures',
            labels={'Test Case ID': 'Failure Count', 'Application': 'Application'},
            custom_data=['Test Case Name']
        )
        fig_stories.update_layout(template='plotly_white')
        fig_stories.update_traces(
            hovertemplate="<b>Application:</b> %{x}<br>" +
                         "<b>Failures:</b> %{y}<br>" +
                         "<b>Failed Test Cases:</b><br>%{customdata[0]}"
        )
        st.plotly_chart(fig_stories)
   
    # Enhanced Defect Status Analysis with test case details
    st.subheader("Defect Resolution Analysis")
   
    defect_status_data = df[df['Execution Status'] == 'Fail'].groupby(
        ['LOB', 'Defect Status']
    ).agg({
        'Test Case ID': 'count',
        'Test Case Name': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else '')
    }).reset_index()
   
    defect_status_pivot = defect_status_data.pivot(
        index='LOB',
        columns='Defect Status',
        values='Test Case ID'
    ).fillna(0)
   
    test_cases_pivot = defect_status_data.pivot(
        index='LOB',
        columns='Defect Status',
        values='Test Case Name'
    )
   
    fig_status = go.Figure()
   
    for status in defect_status_pivot.columns:
        fig_status.add_trace(go.Bar(
            name=status,
            x=defect_status_pivot.index,
            y=defect_status_pivot[status],
            customdata=test_cases_pivot[status],
            hovertemplate="<b>LOB:</b> %{x}<br>" +
                         "<b>Status:</b> " + status + "<br>" +
                         "<b>Count:</b> %{y}<br>" +
                         "<b>Test Cases:</b><br>%{customdata}"
        ))
   
    fig_status.update_layout(
        title='Defect Status by LOB',
        barmode='stack',
        template='plotly_white',
        showlegend=True,
        legend_title='Defect Status'
    )
    st.plotly_chart(fig_status, use_container_width=True)

# AI Analysis Section
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI Gap Analysis")
       
        # Calculate test coverage metrics with test case names
        test_coverage = df.groupby(['LOB', 'Test Case Name']).size().reset_index(name='execution_count')
        lob_coverage = test_coverage.groupby('LOB').agg({
            'Test Case Name': 'count',
            'execution_count': 'sum'
        }).reset_index()
       
        # Calculate defect patterns with test case details
        defect_patterns = df[df['Execution Status'] == 'Fail'].groupby(
            ['LOB', 'Test Case Name', 'Defect Type', 'Defect Description']
        ).size().reset_index(name='count')
        defect_patterns = defect_patterns.sort_values('count', ascending=False).head(10)
       
        # Identify test cases with high failure rates
        tc_failure_rates = df.groupby(['Test Case Name', 'LOB']).agg({
            'Execution Status': lambda x: (x == 'Fail').mean() * 100,
            'Defect Type': lambda x: list(x[x != ''].unique()),
            'Defect Description': lambda x: list(x[x != ''].unique())
        }).reset_index()
        high_risk_tcs = tc_failure_rates[tc_failure_rates['Execution Status'] > 50].head(5)
       
        # Get uncovered scenarios (test cases with no recent executions)
        recent_date = df['Execution Date'].max() - pd.Timedelta(days=30)
        recent_executions = df[df['Execution Date'] > recent_date]['Test Case Name'].unique()
        all_test_cases = df['Test Case Name'].unique()
        uncovered_tcs = set(all_test_cases) - set(recent_executions)
       
        # Calculate defect resolution metrics
        defect_resolution = df[df['Execution Status'] == 'Fail'].groupby(['LOB', 'Defect Status']).size().unstack(fill_value=0)
        resolution_rate = (defect_resolution['Closed'] / defect_resolution.sum(axis=1) * 100).round(2)
       
        analysis_prompt = f"""
        Based on the actual test execution data, provide a comprehensive gap analysis:
 
        Test Coverage Overview:
        {lob_coverage.to_string()}
       
        Top 10 Most Frequent Defect Patterns:
        {defect_patterns.to_string()}
       
        High-Risk Test Cases (>50% failure rate):
        {high_risk_tcs.to_string()}
       
        Test Cases Not Executed in Last 30 Days:
        {list(uncovered_tcs)[:5]}  # Showing first 5 uncovered test cases
       
        Defect Resolution by LOB:
        Resolution Rate (%):
        {resolution_rate.to_string()}
       
        Please provide:
        1. Coverage Gap Analysis
           - Identify areas with insufficient test coverage
           - Analyze test distribution across LOBs
           - Highlight critical functionality gaps
           - Recommend coverage improvements
       
        2. Test Case Risk Assessment
           - Analyze patterns in high-risk test cases
           - Evaluate impact on business functionality
           - Identify common failure modes
           - Suggest stability improvements
       
        3. Defect Resolution Analysis
           - Evaluate defect resolution efficiency
           - Identify bottlenecks in defect lifecycle
           - Analyze patterns in unresolved defects
           - Recommend process improvements
       
        4. Test Execution Patterns
           - Analyze test execution frequency
           - Identify under-tested scenarios
           - Evaluate test data coverage
           - Suggest execution strategy improvements
       
        5. Quality Metrics Assessment
           - Evaluate overall test effectiveness
           - Analyze defect detection efficiency
           - Assess test reliability
           - Recommend quality improvements
       
        6. Resource Optimization
           - Identify areas needing additional testing
           - Suggest resource allocation improvements
           - Recommend automation opportunities
           - Propose test optimization strategies
       
        7. Action Plan
           - Prioritized list of gaps to address
           - Specific recommendations for each gap
           - Timeline suggestions for improvements
           - Resource requirements and allocation
       
        Focus on providing data-driven insights and specific, actionable recommendations based on the actual test results.
        Prioritize gaps based on business impact and provide concrete steps for improvement.
        """
       
        with st.spinner("Generating gap analysis..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)
 
def lob_analysis_tab(df):
    """Tab 4: LOB-Wise Failure Analysis"""
    st.header("📊 LOB-Wise Failure Analysis")
   
    # Add floating prompt
    st.session_state.prompt=True
    add_floating_prompt_to_tab("lob", get_ai_analysis)
   
    # Add weekend vs weekday analysis
    df['DayOfWeek'] = df['Execution Date'].dt.day_name()
    df['IsWeekend'] = df['Execution Date'].dt.dayofweek.isin([5, 6])
   
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
    st.subheader("📈 Detailed Defect Analysis")
   
    col3, col4 = st.columns(2)
   
    with col3:
        # Enhanced defect type distribution
        defect_dist = lob_df[lob_df['Execution Status'] == 'Fail'].groupby('Defect Type').agg({
            'Test Case ID': 'count',
            'Test Case Name': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else ''),
            'Defect Description': lambda x: '<br>'.join(x.unique()[:2]) + ('...' if len(x.unique()) > 2 else '')
        }).reset_index()
       
        fig_defect = px.pie(
            defect_dist,
            values='Test Case ID',
            names='Defect Type',
            title=f'Defect Type Distribution for {selected_lob}',
            custom_data=['Test Case Name', 'Defect Description']
        )
        fig_defect.update_layout(template='plotly_white')
        fig_defect.update_traces(
            textposition='outside',
            textinfo='percent+label',
            hovertemplate="<b>Type:</b> %{label}<br>" +
                         "<b>Count:</b> %{value}<br>" +
                         "<b>Test Cases:</b><br>%{customdata[0]}<br>" +
                         "<b>Sample Defects:</b><br>%{customdata[1]}"
        )
        st.plotly_chart(fig_defect)
   
    with col4:
        # Enhanced defect status distribution
        status_dist = lob_df[lob_df['Execution Status'] == 'Fail'].groupby('Defect Status').agg({
            'Test Case ID': 'count',
            'Test Case Name': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else ''),
            'Defect Description': lambda x: '<br>'.join(x.unique()[:2]) + ('...' if len(x.unique()) > 2 else '')
        }).reset_index()
       
        fig_status = px.pie(
            status_dist,
            values='Test Case ID',
            names='Defect Status',
            title=f'Defect Status Distribution for {selected_lob}',
            custom_data=['Test Case Name', 'Defect Description']
        )
        fig_status.update_layout(template='plotly_white')
        fig_status.update_traces(
            textposition='outside',
            textinfo='percent+label',
            hovertemplate="<b>Status:</b> %{label}<br>" +
                         "<b>Count:</b> %{value}<br>" +
                         "<b>Test Cases:</b><br>%{customdata[0]}<br>" +
                         "<b>Sample Defects:</b><br>%{customdata[1]}"
        )
        st.plotly_chart(fig_status)
   
    # Add weekly pattern analysis
    st.subheader("📅 Weekly Pattern Analysis")
   
    # Enhanced weekly failure distribution
    weekly_dist = lob_df[lob_df['Execution Status'] == 'Fail'].groupby(['DayOfWeek', 'Defect Type']).agg({
        'Test Case ID': 'count',
        'Test Case Name': lambda x: '<br>'.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else '')
    }).reset_index()
   
    fig_weekly = px.bar(
        weekly_dist,
        x='DayOfWeek',
        y='Test Case ID',
        color='Defect Type',
        title=f'Weekly Failure Pattern for {selected_lob}',
        labels={'Test Case ID': 'Number of Failures', 'DayOfWeek': 'Day of Week'},
        custom_data=['Test Case Name']
    )
    fig_weekly.update_layout(
        template='plotly_white',
        barmode='stack'
    )
    fig_weekly.update_traces(
        hovertemplate="<b>Day:</b> %{x}<br>" +
                     "<b>Type:</b> %{color}<br>" +
                     "<b>Failures:</b> %{y}<br>" +
                     "<b>Test Cases:</b><br>%{customdata[0]}"
    )
    st.plotly_chart(fig_weekly, use_container_width=True)
 
    # AI Analysis Section
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI LOB Analysis")
       
        # Calculate LOB-specific metrics
        total_executions = len(lob_df)
        failure_rate = len(lob_df[lob_df['Execution Status'] == 'Fail']) / total_executions * 100
        weekend_failure_rate = len(lob_df[(lob_df['Execution Status'] == 'Fail') & (lob_df['IsWeekend'])]) / len(lob_df[lob_df['IsWeekend']]) * 100
       
        # Get top failing test cases
        top_failures = lob_df[lob_df['Execution Status'] == 'Fail'].groupby(
            ['Test Case ID', 'Test Case Name']
        ).size().nlargest(5)
        top_failures_str = '\n'.join([f"- {id} ({name}): {count} failures"
                                    for (id, name), count in top_failures.items()])
       
        # Get defect type distribution
        defect_dist = lob_df[lob_df['Execution Status'] == 'Fail'].groupby('Defect Type').agg({
            'Test Case ID': 'count',
            'Test Case Name': lambda x: list(x.unique())
        })
       
        # Format the analysis template
        analysis_template = f"""
        ### LOB Analysis Summary for {selected_lob}
       
        #### Key Metrics
        - Total Test Executions: {total_executions:,}
        - Overall Failure Rate: {failure_rate:.1f}%
        - Weekend Failure Rate: {weekend_failure_rate:.1f}%
       
        #### Top Failing Test Cases
        ```
        {top_failures_str}
        ```
       
        #### Defect Type Distribution
        ```
        {defect_dist['Test Case ID'].to_string()}
        ```
       
        #### Recommendations
        1. {'Review weekend test execution strategy' if weekend_failure_rate > failure_rate else 'Maintain current execution schedule'}
        2. Focus on stabilizing top failing test cases
        3. Address predominant defect types
        4. Monitor and improve overall failure rate
        """
       
        st.markdown(analysis_template)

def predictive_analysis_tab(df):
    """Tab 5: Predictive Analysis"""
    st.header("🔮 Predictive Analysis")
   
    # Add floating prompt
    st.session_state.prompt=True
    add_floating_prompt_to_tab("predictive", get_ai_analysis)
   
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
   
    # Display most common test cases affected by this defect type
    st.subheader(f"Most Common Test Cases with {selected_defect_type} Issues")
    test_case_counts = filtered_defects.groupby(['Test Case ID', 'Test Case Name']).size().reset_index(name='count')
    test_case_counts = test_case_counts.sort_values('count', ascending=False)
   
    # Create bar chart for test cases
    fig_test_cases = px.bar(
        test_case_counts.head(10),
        x='Test Case ID',
        y='count',
        title=f'Top 10 Test Cases with {selected_defect_type} Issues',
        labels={'count': 'Occurrence Count', 'Test Case ID': 'Test Case ID'},
        hover_data=['Test Case Name']
    )
    st.plotly_chart(fig_test_cases, use_container_width=True)
   
    # Display LOB distribution for this defect type
    st.subheader(f"LOB Distribution for {selected_defect_type} Issues")
    lob_counts = filtered_defects.groupby('LOB').size().reset_index(name='count')
    lob_counts = lob_counts.sort_values('count', ascending=False)
   
    # Create pie chart for LOB distribution
    fig_lob = px.pie(
        lob_counts,
        values='count',
        names='LOB',
        title=f'LOB Distribution for {selected_defect_type} Issues'
    )
    st.plotly_chart(fig_lob, use_container_width=True)
   
    # Display common defect descriptions
    st.subheader(f"Common {selected_defect_type} Issue Patterns")
    description_counts = filtered_defects.groupby('Defect Description').size().reset_index(name='count')
    description_counts = description_counts.sort_values('count', ascending=False)
    st.dataframe(description_counts)
   
    # Identify most affected LOB
    most_affected_lob = lob_counts.iloc[0]['LOB'] if not lob_counts.empty else "N/A"
   
    # Get defect type distribution
    defect_type_dist = df[df['Execution Status'] == 'Fail'].groupby('Defect Type').size()
   
    # Get automation issues
    automation_issues = df[
        (df['Execution Status'] == 'Fail') & 
        (df['Defect Type'] == 'Automation')
    ].groupby('Defect Description').size().nlargest(5)
   
    # Get manual/functional issues
    manual_issues = df[
        (df['Execution Status'] == 'Fail') & 
        (df['Defect Type'] != 'Automation')
    ].groupby('Defect Description').size().nlargest(5)
   
    # Calculate test case stability
    test_stability = df.groupby('Test Case Name').agg({
        'Execution Status': lambda x: (x == 'Pass').mean() * 100
    }).sort_values('Execution Status')
   
    # Get defect status distribution
    defect_status_dist = df[df['Execution Status'] == 'Fail'].groupby('Defect Status').size()
   
    # Add beautiful prompt
    add_floating_prompt_to_tab("root_cause", get_ai_analysis)
   
    # AI Analysis Section
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI Root Cause Analysis")
       
        analysis_prompt = f"""
        Based on the actual test execution data, provide a detailed analysis:
 
        Test Execution Summary:
        - Total Test Cases: {len(df)}
        - Failed Test Cases: {len(df[df['Execution Status'] == 'Fail'])}
        - Most Affected LOB: {most_affected_lob}
       
        Defect Distribution:
        {defect_type_dist.to_string()}
       
        Top 5 Most Critical Issues:
        {automation_issues.to_string()}
       
        Top 5 UI/Functional Issues:
        {manual_issues.to_string()}
       
        Defect Status Overview:
        {defect_status_dist.to_string()}
       
        Least Stable Test Cases (Bottom 5):
        {test_stability.head().to_string()}
 
        Please provide:
        1. Critical Issue Analysis
           - Identify patterns in the most frequent failures
           - Analyze root causes based on defect descriptions
           - Assess impact on different LOBs
       
        2. Test Case Stability Assessment
           - Evaluate patterns in unstable test cases
           - Identify common factors in failing scenarios
           - Suggest improvements for test reliability
       
        3. Defect Management Insights
           - Analyze defect resolution patterns
           - Identify bottlenecks in defect lifecycle
           - Recommend process improvements
       
        4. Risk Assessment
           - Evaluate impact on business functionality
           - Identify high-risk areas needing immediate attention
           - Suggest preventive measures
       
        5. Actionable Recommendations
           - Specific steps for improving test stability
           - Process improvements for defect management
           - Test case enhancement suggestions
           - Resource allocation recommendations
       
        Focus on providing data-driven insights and specific, actionable recommendations based on the actual test results.
        """
       
        with st.spinner("Generating comprehensive analysis..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)

# Add beautiful prompt functions directly to app.py
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
    return """<svg class="arrow-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
    </svg>"""

def beautiful_prompt_section(context: Optional[str] = None, analysis_function: Optional[Callable] = None):
    """Create beautiful floating prompt section with arrow send button"""
    
    # Add custom CSS
    create_beautiful_prompt_css()
    
    # Create container
    with st.container():
        # Add CSS to hide the Streamlit input
        st.markdown(
            """
            <style>
            /* Hide the default Streamlit input */
            .stTextInput {
                position: absolute;
                left: -9999px;
                width: 1px;
                height: 1px;
                overflow: hidden;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Create a hidden text input to capture the user's query
        input_key = f"hidden_input_{context if context else 'general'}"
        user_input = st.text_input("", key=input_key, label_visibility="collapsed")
        
        st.markdown('<div class="beautiful-chat">', unsafe_allow_html=True)
        
        # Add context-specific placeholder text
        placeholder_text = {
            "failure": "Ask about failure patterns or specific issues...",
            "trend": "Ask about trends and patterns over time...",
            "gap": "Ask about testing gaps and coverage...",
            "lob": "Ask about LOB performance and issues...",
            "predictive": "Ask about predictions and risks...",
            "root_cause": "Ask about root causes and solutions..."
        }.get(context, "Ask a question about the analysis...")
        
        # Create input container with send button
        st.markdown(
            f'''
            <div class="input-container">
                <input type="text" class="chat-input" 
                    placeholder="{placeholder_text}" 
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
        "predictive": "predictions and future risks",
        "root_cause": "root causes and solutions"
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

# Main application
st.title("🔍 QA Failure Analysis Dashboard")

# Set fixed CSV path instead of file upload
csv_path = "qa_failure_analysis.csv"  # Set your fixed CSV path here

# Add toggle switches for connections
col1, col2 = st.columns(2)

with col1:
    connect_data = st.toggle("Connect for Data", value=False)
    if connect_data:
        st.success("Data Retrieved Successfully from Rally")

with col2:
    connect_ai = st.toggle("Connect With Azure Open AI", value=False)
    if connect_ai:
        st.success("Connected with Azure Open AI")
        st.session_state.ai_analysis = True

# Process data based on toggle instead of file upload
if connect_data:
    # Load and validate data from fixed path
    df = load_and_validate_data(csv_path)
   
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
    st.info("Please toggle 'Connect for Data' to begin the analysis.")
   
    # Display sample CSV format
    st.subheader("Required CSV Format:")
    sample_data = {
        'Execution Date': ['2024-01-01', '2024-01-01'],
        'Application': ['Story 1', 'Story 2'],
        'Test Case ID': ['TC001', 'TC002'],
        'LOB': ['Banking', 'Insurance'],
        'Execution Status': ['Pass', 'Fail'],
        'Defect ID': ['', 'D_001'],
        'Defect Description': ['', 'Sample defect'],
        'Defect Type': ['', 'Automation'],
        'Defect Status': ['', 'Open']
    }
    st.dataframe(pd.DataFrame(sample_data))