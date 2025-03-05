"""
Deep Research Agent Streamlit Interface

A modern web interface for the comprehensive workflow system,
featuring custom styling, interactive controls, and real-time progress updates.
"""

import streamlit as st
import asyncio
import time
import json
import threading
from datetime import datetime
import tempfile
import base64
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Import workflow system
from workflow_system import (
    WorkflowConfig,
    FormatConfig,
    generate_research_report,
    export_markdown,
    export_docx,
    export_pdf,
    export_pptx
)

# Initialize session state to store the report and progress
if 'report' not in st.session_state:
    st.session_state.report = None
if 'has_searched' not in st.session_state:
    st.session_state.has_searched = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'current_step' not in st.session_state:
    st.session_state.current_step = "Initializing"
if 'workflow_logs' not in st.session_state:
    st.session_state.workflow_logs = []
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = 0
if 'progress_thread' not in st.session_state:
    st.session_state.progress_thread = None
if 'cancellation_token' not in st.session_state:
    st.session_state.cancellation_token = False
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = time.time()
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0

# Page configuration
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-top: 0;
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #1E88E5;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 10px;
    }
    .metadata {
        font-size: 0.8rem;
        color: #757575;
        font-style: italic;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .feature-box {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1.2rem;
        border: 1px solid #e0e0e0;
    }
    .feature-title {
        font-weight: bold;
        color: #1E88E5;
        font-size: 1.1rem;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .advanced-options {
        background-color: #e3f2fd;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .export-button {
        background-color: #1E88E5;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 10px 5px;
        text-align: center;
    }
    .log-container {
        max-height: 300px;
        overflow-y: auto;
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 10px;
        font-family: monospace;
        margin-top: 10px;
    }
    .log-entry {
        padding: 5px;
        border-bottom: 1px solid #ddd;
    }
    .log-entry-time {
        color: #757575;
        font-size: 0.8rem;
    }
    .log-entry-content {
        margin-left: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def create_download_link(file_path: str, link_text: str) -> str:
    """Create a download link for a file."""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    file_name = os.path.basename(file_path)
    mime_type = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.md': 'text/markdown'
    }.get(os.path.splitext(file_path)[1], 'application/octet-stream')

    href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}" class="export-button">{link_text}</a>'
    return href


def display_header():
    """Display the application header."""
    st.markdown('<p class="main-header">🔍 Deep Research Agent</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate comprehensive research reports on any topic using a multi-agent workflow</p>',
                unsafe_allow_html=True)


def display_intro():
    """Display the intro section with features."""
    # Create a three-column layout for features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
        <p class="feature-title">🧠 Multi-Agent Workflow</p>
        <ul>
            <li><strong>Planner:</strong> Creates research strategy</li>
            <li><strong>Researcher:</strong> Gathers credible sources</li>
            <li><strong>Fact Checker:</strong> Validates claims</li>
            <li><strong>Writer & Editor:</strong> Creates polished content</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
        <p class="feature-title">📊 Research Customization</p>
        <ul>
            <li><strong>Depth Control:</strong> Basic to exhaustive</li>
            <li><strong>Citation Styles:</strong> APA, MLA, Chicago</li>
            <li><strong>Topic Analysis:</strong> Comprehensive breakdown</li>
            <li><strong>Source Validation:</strong> Credibility scoring</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
        <p class="feature-title">📄 Multiple Export Formats</p>
        <ul>
            <li><strong>Markdown:</strong> Clean, portable text</li>
            <li><strong>Word Document:</strong> Professional reports</li>
            <li><strong>PDF:</strong> Publication-ready documents</li>
            <li><strong>PowerPoint:</strong> Presentation slides</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# Function to simulate progress updates in a background thread
def update_progress(cancellation_token):
    """Update progress in a background thread."""
    try:
        start_time = time.time()
        progress_values = [0, 5, 10, 15, 20, 25, 30, 35, 40,
                           45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 100]
        steps = ["Initializing", "Planning Research", "Gathering Sources", "Validating Facts",
                 "Analyzing Information", "Writing Content", "Editing Document", "Formatting Output", "Finalizing"]

        # Add some initial logs
        st.session_state.workflow_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": "Research workflow initialized"
        })

        # Simulated progress updates
        for i, progress_value in enumerate(progress_values):
            # Check if we should stop
            if cancellation_token():
                print("Progress update cancelled")
                break

            # Update progress in session state
            st.session_state.progress = progress_value

            # Update the current step based on progress
            step_index = min(int(i / len(progress_values)
                             * len(steps)), len(steps) - 1)
            current_step = steps[step_index]
            st.session_state.current_step = current_step

            # Add logs for each step change
            if i == 0 or (i > 0 and current_step != steps[min(int((i-1) / len(progress_values) * len(steps)), len(steps) - 1)]):
                st.session_state.workflow_logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "message": f"Working on: {current_step}"
                })

            # Calculate time elapsed
            st.session_state.execution_time = int(time.time() - start_time)

            # Print debug info to console
            print(
                f"Progress update: {progress_value}%, Step: {current_step}, Time: {st.session_state.execution_time}s")

            # Sleep before next update - longer time for less UI pressure
            # This is a background thread, so we can sleep longer without affecting the UI
            time.sleep(1.5)
    except Exception as e:
        print(f"Error in progress update thread: {str(e)}")
        # Add error to logs
        st.session_state.workflow_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": f"Progress update error: {str(e)}"
        })


# Function to run the async report generation with threading
def run_async_report_generation(research_topic, document_format, citation_style, research_depth):
    """Run the report generation using threading."""
    try:
        # Reset progress tracking
        st.session_state.progress = 0
        st.session_state.current_step = "Initializing"
        st.session_state.workflow_logs = []
        st.session_state.execution_time = 0
        st.session_state.cancellation_token = False

        # Add initial log
        st.session_state.workflow_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": f"Starting research on: {research_topic}"
        })

        # Start the progress simulation in a background thread
        progress_thread = threading.Thread(
            target=update_progress,
            args=(lambda: st.session_state.cancellation_token,),
            daemon=True  # Make thread daemon so it doesn't block app shutdown
        )
        progress_thread.start()

        # Create workflow configuration
        config = WorkflowConfig(
            query=research_topic,
            research_depth=research_depth,
            format_config=FormatConfig(
                format_type=document_format,
                citation_style=citation_style,
                include_toc=True,
                include_abstract=True
            ),
            verbose=True
        )

        # Log that we're starting the workflow
        st.session_state.workflow_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": "Executing research workflow..."
        })

        # Run the workflow using the synchronous version
        print(f"Starting research workflow for: {research_topic}")
        report = generate_research_report(
            topic=research_topic,
            document_format=document_format,
            citation_style=citation_style,
            research_depth=research_depth
        )
        print(f"Research workflow completed for: {research_topic}")

        # Signal the progress thread to stop
        st.session_state.cancellation_token = True

        # Update session state with the report
        st.session_state.report = report
        st.session_state.progress = 100
        st.session_state.current_step = "Completed"

        # Add completion log
        st.session_state.workflow_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": "Research workflow completed successfully"
        })

        return report
    except Exception as e:
        # Log the error
        error_message = f"Error: {str(e)}"
        print(f"Research error: {error_message}")

        st.session_state.workflow_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": error_message
        })

        # Signal the progress thread to stop
        st.session_state.cancellation_token = True

        # Update session state to show error
        st.session_state.current_step = "Error"
        st.session_state.progress = 0

        # Don't raise the exception, just return None
        return None


def export_report(report_content, format_type, file_prefix="research_report"):
    """Export the report to the specified format and provide a download link."""
    if not report_content:
        st.error("No report content available to export")
        return None

    # Create a temporary file for the export
    temp_dir = tempfile.mkdtemp()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Handle both dictionary reports and string content
        if isinstance(report_content, dict):
            # This is a report dictionary
            content = report_content.get(
                "content") or report_content.get("final_document", "")
            if not content:
                st.error("No content available in the report dictionary")
                return None
            report_dict = report_content
        else:
            # This is just the content string
            content = report_content
            report_dict = {"content": content, "final_document": content}

        if format_type == "markdown":
            file_path = os.path.join(temp_dir, f"{file_prefix}_{timestamp}.md")
            with open(file_path, "w") as f:
                f.write(content)
        elif format_type == "docx":
            file_path = os.path.join(
                temp_dir, f"{file_prefix}_{timestamp}.docx")
            export_docx(report_dict, file_path)
        elif format_type == "pdf":
            file_path = os.path.join(
                temp_dir, f"{file_prefix}_{timestamp}.pdf")
            export_pdf(report_dict, file_path)
        elif format_type == "pptx":
            file_path = os.path.join(
                temp_dir, f"{file_prefix}_{timestamp}.pptx")
            export_pptx(report_dict, file_path)
        else:
            st.error(f"Unsupported format: {format_type}")
            return None

        # Create a download link
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        return file_bytes, os.path.basename(file_path), f"application/{format_type}"

    except Exception as e:
        st.error(f"Error exporting report: {str(e)}")
        return None


def display_research_form():
    """Display the research form with all options."""
    with st.form("research_form"):
        research_topic = st.text_input(
            "Enter a research topic:",
            placeholder="e.g., The impact of artificial intelligence on healthcare"
        )

        # Create two columns for form options
        col1, col2 = st.columns(2)

        with col1:
            document_format = st.selectbox(
                "Output Format",
                options=["markdown", "docx", "pdf", "pptx"],
                index=0,
                help="Select the output document format"
            )

            citation_style = st.selectbox(
                "Citation Style",
                options=["APA", "MLA", "Chicago"],
                index=0,
                help="Select the academic citation style for references"
            )

        with col2:
            research_depth = st.select_slider(
                "Research Depth",
                options=["basic", "comprehensive", "exhaustive"],
                value="comprehensive",
                help="Basic: Quick overview with key facts\nComprehensive: Detailed analysis with supporting evidence\nExhaustive: In-depth academic research with cross-validation"
            )

            # A placeholder for future options
            st.write(
                "Full Langgraph multi-agent workflow will be used for research")

        # Submit button
        submitted = st.form_submit_button(
            "Start Research", use_container_width=True)

        if submitted:
            if not research_topic:
                st.error("Please enter a research topic")
                return

            # Reset session state for a new research
            st.session_state.report = None
            st.session_state.has_searched = True
            st.session_state.progress = 0
            st.session_state.current_step = "Initializing"
            st.session_state.workflow_logs = []
            st.session_state.execution_time = 0
            st.session_state.cancellation_token = False

            # Display a message that research is starting
            st.info(f"Starting research on: {research_topic}")

            # Log the start of research
            print(f"Starting research on: {research_topic}")
            print(
                f"Format: {document_format}, Style: {citation_style}, Depth: {research_depth}")

            # Start the research process in a background thread
            research_thread = threading.Thread(
                target=run_async_report_generation,
                args=(research_topic, document_format,
                      citation_style, research_depth),
                daemon=True  # Make thread daemon so it doesn't block app shutdown
            )
            research_thread.start()

            # Force a rerun to show the progress page
            st.rerun()


def display_progress():
    """Display the research progress with detailed information."""
    st.markdown('<p class="section-header">Research Progress</p>',
                unsafe_allow_html=True)

    # Progress bar with custom styling - simplified to reduce rendering load
    progress_value = st.session_state.progress / 100
    progress_color = "#4CAF50" if progress_value > 0.7 else "#2196F3" if progress_value > 0.4 else "#FFC107"

    # Use Streamlit's built-in progress bar instead of custom HTML
    st.progress(progress_value)

    # Display step and time information in a more efficient way
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Step", st.session_state.current_step)
    with col2:
        st.metric("Progress", f"{st.session_state.progress}%")
    with col3:
        execution_time_str = f"{st.session_state.execution_time // 60}m {st.session_state.execution_time % 60}s"
        st.metric("Time Elapsed", execution_time_str)

    # Display workflow logs with improved styling but more efficient rendering
    with st.expander("Workflow Logs", expanded=True):
        # Use a more efficient way to display logs
        # Only show the last 10 logs to reduce rendering load
        for log in st.session_state.workflow_logs[-10:]:
            st.markdown(f"**[{log['time']}]** {log['message']}")

        if len(st.session_state.workflow_logs) > 10:
            st.info(
                f"Showing the last 10 of {len(st.session_state.workflow_logs)} logs.")

    # Cancel button with warning styling
    if st.button("Cancel Research", key="cancel_button", type="primary", help="Click to cancel the current research process"):
        st.session_state.cancellation_token = True
        st.warning("Research cancellation requested. Please wait...")
        # Don't rerun here - let the timed refresh handle it


def display_report():
    """Display the research report with tabs for different sections."""
    if st.session_state.report is None:
        return

    report = st.session_state.report

    # Report header
    st.markdown(f'<p class="section-header">Research Report: {report["title"].replace("Research Report: ", "")}</p>',
                unsafe_allow_html=True)

    # Create tabs for different report sections
    report_tabs = st.tabs(["Full Report", "Export Options", "Metadata"])

    # Full Report tab
    with report_tabs[0]:
        st.markdown(report["content"])

    # Export Options tab
    with report_tabs[1]:
        st.markdown(
            '<p class="section-header">Export Research Report</p>', unsafe_allow_html=True)

        # Create a two-column layout for export options
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.markdown("<p>Export as Markdown</p>", unsafe_allow_html=True)
            md_file = export_report(report["content"], "markdown")
            if md_file:
                st.markdown(create_download_link(
                    md_file, "📄 Download Markdown"), unsafe_allow_html=True)

            st.markdown("<p>Export as PDF</p>", unsafe_allow_html=True)
            pdf_file = export_report(report["content"], "pdf")
            if pdf_file:
                st.markdown(create_download_link(
                    pdf_file, "📄 Download PDF"), unsafe_allow_html=True)

        with export_col2:
            st.markdown("<p>Export as Word Document</p>",
                        unsafe_allow_html=True)
            docx_file = export_report(report["content"], "docx")
            if docx_file:
                st.markdown(create_download_link(
                    docx_file, "📄 Download DOCX"), unsafe_allow_html=True)

            st.markdown("<p>Export as PowerPoint</p>", unsafe_allow_html=True)
            pptx_file = export_report(report["content"], "pptx")
            if pptx_file:
                st.markdown(create_download_link(
                    pptx_file, "📄 Download PPTX"), unsafe_allow_html=True)

    # Metadata tab
    with report_tabs[2]:
        st.markdown('<p class="section-header">Report Metadata</p>',
                    unsafe_allow_html=True)

        metadata = report["metadata"]

        # Create a more visual display of metadata
        meta_col1, meta_col2 = st.columns(2)

        with meta_col1:
            st.metric("Generated At", metadata.get(
                "timestamp", "N/A").split('T')[0])
            st.metric("Workflow Type", metadata.get(
                "workflow_type", "Advanced"))
            st.metric("Research Depth", metadata.get(
                "research_depth", "comprehensive"))

        with meta_col2:
            st.metric("Citation Style", metadata.get("citation_style", "APA"))
            st.metric("Sources Analyzed", metadata.get(
                "sources_analyzed", "N/A"))
            st.metric("Validated Facts", metadata.get(
                "validated_facts_count", "N/A"))

        # Display raw metadata in JSON format
        with st.expander("Raw Metadata", expanded=False):
            st.json(metadata)


def main():
    """Main Streamlit application."""
    # Initialize session state variables if they don't exist
    if "has_searched" not in st.session_state:
        st.session_state.has_searched = False
    if "report" not in st.session_state:
        st.session_state.report = None
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    if "current_step" not in st.session_state:
        st.session_state.current_step = "Initializing"
    if "workflow_logs" not in st.session_state:
        st.session_state.workflow_logs = []
    if "execution_time" not in st.session_state:
        st.session_state.execution_time = 0
    if "cancellation_token" not in st.session_state:
        st.session_state.cancellation_token = False

    # Display header
    display_header()

    # Display content based on state
    if not st.session_state.has_searched:
        # Show intro and research form
        display_intro()
        display_research_form()
    elif st.session_state.report is None:
        # Research is in progress - show progress
        display_progress()

        # Add a manual refresh button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Refresh Progress", key="manual_refresh"):
                st.rerun()
    else:
        # Research is complete - show report
        display_report()

        # Add a button to start a new research
        if st.button("Start New Research", key="new_research"):
            st.session_state.report = None
            st.session_state.has_searched = False
            st.session_state.progress = 0
            st.session_state.current_step = "Initializing"
            st.session_state.workflow_logs = []
            st.session_state.execution_time = 0
            st.session_state.cancellation_token = False
            st.rerun()


if __name__ == "__main__":
    main()
