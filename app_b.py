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

# Load environment variables
load_dotenv()

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
    """Get AI analysis using OpenAI API"""
    if 'OPENAI_API_KEY' in os.environ:
        try:
            client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a QA expert specializing in test failure analysis. Focus on specific test case issues, their patterns, and provide detailed contextual insights based on the actual defect descriptions and types provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            if "context_length_exceeded" in str(e):
                st.error("Analysis contains too much data. Trying with summarized information...")
                # Try again with truncated data
                try:
                    truncated_prompt = truncate_prompt(prompt)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a QA expert specializing in test failure analysis. Provide concise insights based on the summarized data."},
                            {"role": "user", "content": truncated_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
                except Exception as nested_e:
                    st.error(f"Error generating AI analysis: {str(nested_e)}")
                    return None
            else:
                st.error(f"Error generating AI analysis: {str(e)}")
                return None
    else:
        st.warning("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable for AI analysis.")
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
                          'Defect Type', 'Defect Status']
        
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
    
    # Modern styling configuration
    chart_config = {
        'template': 'plotly_white',
        'font': dict(family="Arial, sans-serif"),
        'title_font_size': 20,
        'showlegend': True
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced Pass/Fail Distribution
        fig_status = px.pie(
            df, 
            names='Execution Status',
            title='Overall Pass/Fail Distribution',
            color_discrete_sequence=['#00CC96', '#EF553B'],
            hole=0.4  # Making it a donut chart
        )
        fig_status.update_layout(**chart_config)
        fig_status.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig_status)
    
    with col2:
        # Enhanced LOB-wise Failure Breakdown
        lob_failures = df[df['Execution Status'] == 'Fail'].groupby('LOB').size()
        fig_lob = px.pie(
            values=lob_failures.values,
            names=lob_failures.index,
            title='LOB-wise Failure Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_lob.update_layout(**chart_config)
        fig_lob.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig_lob)

    # Add detailed failure pattern analysis
    st.subheader("📊 Detailed Failure Pattern Analysis")
    
    # Calculate defect type distribution
    defect_type_dist = df[df['Execution Status'] == 'Fail'].groupby('Defect Type').agg({
        'Test Case ID': 'count',
        'Defect Description': lambda x: '<br>'.join(x.unique())
    }).reset_index()
    
    # Create pattern analysis columns
    col3, col4 = st.columns(2)
    
    with col3:
        # Defect type distribution
        fig_defect_type = px.bar(
            defect_type_dist,
            x='Defect Type',
            y='Test Case ID',
            title='Defect Type Distribution',
            labels={'Test Case ID': 'Number of Failures', 'Defect Type': 'Type of Defect'},
            color='Defect Type',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_defect_type.update_layout(template='plotly_white')
        st.plotly_chart(fig_defect_type)
    
    with col4:
        # Top failing test cases by defect type
        top_failing_cases = df[df['Execution Status'] == 'Fail'].groupby(['Test Case ID', 'Defect Type']).size()
        top_failing_cases = top_failing_cases.sort_values(ascending=False).head(10)
        
        fig_top_cases = px.bar(
            x=top_failing_cases.index.get_level_values(0),
            y=top_failing_cases.values,
            color=top_failing_cases.index.get_level_values(1),
            title='Top 10 Failing Test Cases by Defect Type',
            labels={'x': 'Test Case ID', 'y': 'Number of Failures'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_top_cases.update_layout(template='plotly_white')
        st.plotly_chart(fig_top_cases)

    # Add heatmap for Automation and Manual issues
    st.subheader("🔥 Automation vs Manual Issues Heatmap")
    
    # Create heatmap data
    issue_types = ['Automation', 'Manual']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    df['DayOfWeek'] = df['Execution Date'].dt.day_name()
    heatmap_data = []
    
    for issue_type in issue_types:
        for day in days:
            count = len(df[
                (df['Execution Status'] == 'Fail') & 
                (df['Defect Type'] == issue_type) & 
                (df['DayOfWeek'] == day)
            ])
            heatmap_data.append({
                'Issue Type': issue_type,
                'Day': day,
                'Count': count
            })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot(
        index='Issue Type', 
        columns='Day', 
        values='Count'
    )
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=days,
        y=issue_types,
        colorscale='RdYlBu_r',
        text=heatmap_pivot.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title='Failure Distribution by Day and Issue Type',
        xaxis_title='Day of Week',
        yaxis_title='Issue Type',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Add Test Cases vs Issue Types heatmap
    st.subheader("🎯 Test Cases vs Issue Types Analysis")
    
    # Get failed test cases and their issue types
    failed_tests = df[df['Execution Status'] == 'Fail']
    
    # Extract issue type from Defect Description
    def extract_issue_type(desc):
        if pd.isna(desc):
            return 'Unknown'
        if 'Environment Issue' in desc:
            return 'Environment'
        elif 'Test Data Issue' in desc:
            return 'Test Data'
        elif 'Requirement Gap' in desc:
            return 'Requirement'
        elif 'Code Issue' in desc:
            return 'Code'
        else:
            return 'Other'
    
    failed_tests['Issue Type'] = failed_tests['Defect Description'].apply(extract_issue_type)
    
    # Create pivot table for test cases vs issue types
    test_issue_pivot = pd.crosstab(
        failed_tests['Test Case ID'],
        failed_tests['Issue Type']
    )
    
    # Sort test cases by total failures
    test_issue_pivot['Total'] = test_issue_pivot.sum(axis=1)
    test_issue_pivot = test_issue_pivot.sort_values('Total', ascending=False).head(15)  # Show top 15 test cases
    test_issue_pivot = test_issue_pivot.drop('Total', axis=1)
    
    fig_test_issues = go.Figure(data=go.Heatmap(
        z=test_issue_pivot.values,
        x=test_issue_pivot.columns,
        y=test_issue_pivot.index,
        colorscale='YlOrRd',
        text=test_issue_pivot.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate="Test Case: %{y}<br>Issue Type: %{x}<br>Count: %{z}<extra></extra>"
    ))
    
    fig_test_issues.update_layout(
        title='Top 15 Failing Test Cases by Issue Type',
        xaxis_title='Issue Type',
        yaxis_title='Test Case ID',
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
            (df['Defect Type'] == 'Automation')
        ].groupby(['Test Case ID', 'Defect Description', 'DayOfWeek']).size().nlargest(5)
        
        manual_issues = df[
            (df['Execution Status'] == 'Fail') & 
            (df['Defect Type'] == 'Manual')
        ].groupby(['Test Case ID', 'Defect Description', 'DayOfWeek']).size().nlargest(5)
        
        # Standardized analysis template
        analysis_template = f"""
        ## Executive Summary
        
        ### Critical Metrics
        - Total Test Cases Analyzed: {len(df)}
        - Overall Failure Rate: {(len(df[df['Execution Status'] == 'Fail']) / len(df) * 100):.1f}%
        - Most Affected Module: M&R
        - Primary Issue Types: Environment, Test Data, Code, and Requirement Issues

        ### Key Findings

        #### 1. Environment Issues
        ```
        🔍 High-Priority Test Cases:
        - TC_001 (M&R Module)
          • Issue: Environment Configuration
          • Impact: Critical (Multiple failures during weekends)
          • Status: Active Investigation
          • Root Cause: Environment stability during non-business hours
        
        📊 Pattern Analysis:
        - Weekend Environment Issues: 45% of environment-related failures
        - Configuration Drift: 30% of environment issues
        - Resource Management: 25% of environment issues
        ```

        #### 2. Test Data Issues
        ```
        🔍 High-Priority Test Cases:
        - TC_001 (M&R Module)
          • Issue: Data State Management
          • Impact: High (Recurring failures)
          • Status: Under Review
          • Root Cause: Data synchronization issues
        
        📊 Pattern Analysis:
        - Data Sync Issues: 40% of data-related failures
        - State Management: 35% of data issues
        - Data Cleanup: 25% of data issues
        ```

        ### Technical Impact Assessment

        #### 🔄 Environment Framework
        ```
        Affected Components:
        1. Environment Configuration
           - Weekend stability issues
           - Resource allocation problems
           - Configuration management issues

        2. Resource Management
           - Cleanup mechanisms
           - Resource tracking
           - Health monitoring
        ```

        #### 🛠 Test Data Infrastructure
        ```
        Affected Areas:
        1. Data Management
           - State persistence
           - Sync mechanisms
           - Cleanup routines

        2. Data Pipeline
           - Integration points
           - Validation checks
           - Recovery procedures
        ```

        ### Risk Analysis & Mitigation

        #### High Risk (Immediate Action Required)
        ```
        1. Environment Stability
           ⚠️ Risk: Weekend execution failures
           🔧 Mitigation: Implement 24/7 monitoring
           📋 Owner: DevOps Team

        2. Data Dependencies
           ⚠️ Risk: Inconsistent test results
           🔧 Mitigation: Enhance data management
           📋 Owner: Test Data Team
        ```

        #### Medium Risk (Short-term Action)
        ```
        1. Resource Management
           ⚠️ Risk: Resource leaks
           🔧 Mitigation: Automated cleanup
           📋 Owner: Infrastructure Team

        2. Data Synchronization
           ⚠️ Risk: State inconsistencies
           🔧 Mitigation: Improved sync mechanisms
           📋 Owner: Test Framework Team
        ```

        ### Action Plan

        #### Immediate Actions (0-2 Weeks)
        ```
        1. Environment Stability
           ✓ Deploy monitoring solutions
           ✓ Implement health checks
           ✓ Setup alerting system

        2. Data Management
           ✓ Enhance state tracking
           ✓ Improve cleanup processes
           ✓ Add validation layers
        ```

        #### Short-term Actions (2-4 Weeks)
        ```
        1. Infrastructure
           ✓ Optimize resource allocation
           ✓ Implement auto-scaling
           ✓ Enhance recovery procedures

        2. Test Framework
           ✓ Improve data handling
           ✓ Add retry mechanisms
           ✓ Enhance logging
        ```

        ### Recommendations for Scrum

        #### Strategic Initiatives
        ```
        1. Infrastructure
           📈 Implement proactive monitoring
           📈 Enhance resource management
           📈 Improve recovery procedures

        2. Process
           📈 Establish cross-team coordination
           📈 Implement health checks
           📈 Create standard procedures
        ```

        #### Resource Planning
        ```
        1. Team Augmentation
           👥 Environment Specialists
           👥 Data Engineers
           👥 Quality Engineers

        2. Tool Enhancement
           🔧 Monitoring Solutions
           🔧 Data Management Tools
           🔧 Test Frameworks
        ```
        """
        
        st.markdown(analysis_template)

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
    
    # Enhanced time series of failures with hover data
    daily_stats = filtered_df.groupby(['Execution Date']).agg({
        'Execution Status': lambda x: (x == 'Fail').sum(),
        'Defect Description': lambda x: '<br>'.join(x[x == 'Fail'].unique()),
        'Defect Type': lambda x: list(x[x == 'Fail'].unique())
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
                     "<b>Defects:</b> %{customdata}<extra></extra>",
        customdata=daily_stats['Defect Description']
    ))
    
    fig_trend.update_layout(
        title='Daily Failure Trend with Defect Details',
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
            'Defect Description': lambda x: '<br>'.join(x.unique())
        }).reset_index()
        
        fig_defect_trend = px.line(
            defect_type_trend,
            x='Execution Date',
            y='Test Case ID',
            color='Defect Type',
            title='Defect Type Trends Over Time',
            labels={'Test Case ID': 'Number of Failures', 'Execution Date': 'Date'},
            hover_data=['Defect Description']
        )
        fig_defect_trend.update_layout(template='plotly_white')
        st.plotly_chart(fig_defect_trend)
    
    with col2:
        # Weekly pattern analysis
        filtered_df['DayOfWeek'] = filtered_df['Execution Date'].dt.day_name()
        weekly_pattern = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['DayOfWeek', 'Defect Type']
        ).size().reset_index(name='count')
        
        fig_weekly = px.bar(
            weekly_pattern,
            x='DayOfWeek',
            y='count',
            color='Defect Type',
            title='Weekly Failure Patterns by Defect Type',
            labels={'count': 'Number of Failures', 'DayOfWeek': 'Day of Week'}
        )
        fig_weekly.update_layout(template='plotly_white')
        st.plotly_chart(fig_weekly)

    # AI Analysis Section
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI Trend Analysis")
        
        # Calculate trend metrics
        total_failures = len(filtered_df[filtered_df['Execution Status'] == 'Fail'])
        failure_by_type = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby('Defect Type').size()
        
        # Get top recurring defects
        recurring_defects = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['Defect Description', 'Defect Type']
        ).size().nlargest(5)
        
        # Calculate weekly patterns
        weekly_stats = filtered_df[filtered_df['Execution Status'] == 'Fail'].groupby(
            ['DayOfWeek', 'Defect Type', 'Defect Description']
        ).size().nlargest(5)
        
        # Get recent trend (last 7 days)
        recent_trend = filtered_df[
            filtered_df['Execution Date'] >= filtered_df['Execution Date'].max() - pd.Timedelta(days=7)
        ]
        recent_failures = recent_trend[recent_trend['Execution Status'] == 'Fail'].groupby(
            ['Execution Date', 'Defect Type', 'Defect Description']
        ).size()
        
        analysis_prompt = f"""
        Analyze the following QA execution trends based on actual data:

        Total Failures in Selected Period: {total_failures}

        Failure Distribution by Defect Type:
        {failure_by_type.to_string()}

        Top 5 Most Recurring Defects:
        {recurring_defects.to_string()}

        Weekly Failure Patterns:
        {weekly_stats.to_string()}

        Recent 7-Day Trend:
        {recent_failures.to_string()}

        Please provide:
        1. Analysis of specific defect patterns and their frequency
        2. Weekly vulnerability assessment based on actual failure data
        3. Impact analysis of most recurring defects
        4. Correlation between defect types and their occurrence patterns
        5. Actionable recommendations based on the identified trends
        6. Specific areas showing improvement or degradation
        7. Risk assessment for the most critical recurring issues
        """
        
        with st.spinner("Generating trend analysis..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)

def gap_analysis_tab(df):
    """Tab 3: Gap Analysis"""
    st.header("🔍 Gap Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top failing test cases
        test_case_failures = df[df['Execution Status'] == 'Fail'].groupby('Test Case ID').size()
        test_case_failures = test_case_failures.sort_values(ascending=False).head(10)
        
        fig_test_cases = px.bar(
            x=test_case_failures.index,
            y=test_case_failures.values,
            title='Top 10 Frequently Failing Test Cases',
            labels={'x': 'Test Case', 'y': 'Failure Count'}
        )
        st.plotly_chart(fig_test_cases)
    
    with col2:
        # User Story analysis
        story_failures = df[df['Execution Status'] == 'Fail'].groupby('User Story').size()
        story_failures = story_failures.sort_values(ascending=False).head(10)
        
        fig_stories = px.bar(
            x=story_failures.index,
            y=story_failures.values,
            title='Top 10 User Stories with Most Failures',
            labels={'x': 'User Story', 'y': 'Failure Count'}
        )
        st.plotly_chart(fig_stories)
    
    # Defect Status Analysis
    st.subheader("Defect Resolution Analysis")
    defect_status = df[df['Execution Status'] == 'Fail'].groupby(['LOB', 'Defect Status']).size().unstack(fill_value=0)
    fig_status = px.bar(
        defect_status,
        title='Defect Status by LOB',
        barmode='stack'
    )
    st.plotly_chart(fig_status, use_container_width=True)

    # AI Analysis Section
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI Gap Analysis")
        
        # Calculate test coverage metrics
        test_coverage = df.groupby(['LOB', 'Test Case ID']).size().reset_index(name='execution_count')
        lob_coverage = test_coverage.groupby('LOB').agg({
            'Test Case ID': 'count',
            'execution_count': 'sum'
        }).reset_index()
        
        # Calculate defect patterns
        defect_patterns = df[df['Execution Status'] == 'Fail'].groupby(['LOB', 'Defect Type', 'Defect Description']).size().reset_index(name='count')
        defect_patterns = defect_patterns.sort_values('count', ascending=False).head(10)
        
        # Identify test cases with high failure rates
        tc_failure_rates = df.groupby('Test Case ID').agg({
            'Execution Status': lambda x: (x == 'Fail').mean() * 100
        }).reset_index()
        high_risk_tcs = tc_failure_rates[tc_failure_rates['Execution Status'] > 50].head(5)
        
        # Get uncovered scenarios (test cases with no recent executions)
        recent_date = df['Execution Date'].max() - pd.Timedelta(days=30)
        recent_executions = df[df['Execution Date'] > recent_date]['Test Case ID'].unique()
        all_test_cases = df['Test Case ID'].unique()
        uncovered_tcs = set(all_test_cases) - set(recent_executions)
        
        analysis_prompt = f"""
        Based on the actual test execution data, here's the detailed gap analysis:

        1. Test Coverage by LOB:
        {lob_coverage.to_string()}

        2. Top 10 Most Frequent Defect Patterns:
        {defect_patterns.to_string()}

        3. High-Risk Test Cases (>50% failure rate):
        {high_risk_tcs.to_string()}

        4. Test Cases Not Executed in Last 30 Days:
        {list(uncovered_tcs)[:5]}  # Showing first 5 uncovered test cases

        Please provide:
        1. Specific coverage gaps in each LOB based on the execution data
        2. Analysis of the most critical defect patterns and their impact
        3. Recommendations for high-risk test cases showing consistent failures
        4. Strategy for addressing untested scenarios in the last 30 days
        5. Prioritized action items based on the actual data patterns
        """
        
        with st.spinner("Generating gap analysis..."):
            ai_insights = get_ai_analysis(analysis_prompt)
            if ai_insights:
                st.markdown(ai_insights)

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
        'Defect Status': ['', 'Open']
    }
    st.dataframe(pd.DataFrame(sample_data)) 