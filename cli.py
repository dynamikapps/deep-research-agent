#!/usr/bin/env python
"""
Deep Research Agent Command Line Interface

A beautiful terminal UI for the comprehensive workflow system,
featuring rich formatting, progress tracking, and interactive controls.
"""

import asyncio
import argparse
import sys
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import time

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

# Import the workflow system
from workflow_system import (
    WorkflowConfig,
    FormatConfig,
    generate_research_report,
    export_markdown,
    export_docx,
    export_pdf,
    export_pptx,
    format_as_research_report
)

# Initialize Rich console
console = Console()


def display_header():
    """Display the application header with styling."""
    console.print()
    header_text = Text("Deep Research Agent", style="bold white on blue")
    header_text.stylize("bold white", 0, 4)
    header_text.stylize("bold yellow", 5, 13)
    header_text.stylize("bold white", 14, 19)

    console.print(Panel(header_text, box=box.DOUBLE, expand=False))
    console.print(
        "A comprehensive AI-powered research workflow system", style="italic")
    console.print()


def display_report_summary(report: Dict[str, Any]):
    """Display a summary of the research report."""
    if not report or not report.get("final_document"):
        console.print("[bold red]No report data available.[/bold red]")
        return

    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )

    # Header
    topic = report.get("query", "Unknown Topic")
    layout["header"].update(
        Panel(
            f"[bold]Research Report:[/bold] {topic}",
            style="bold white on dark_blue"
        )
    )

    # Body - Summary table
    table = Table(box=box.SIMPLE)
    table.add_column("Category", style="cyan")
    table.add_column("Details", style="white")

    # Add metadata
    table.add_row("Topic", report.get("query", "Unknown"))
    table.add_row("Timestamp", report.get("timestamp", "Unknown"))
    table.add_row("Research Depth", report.get("research_depth", "Unknown"))
    table.add_row("Citation Style", report.get("citation_style", "Unknown"))
    table.add_row("Format", report.get("document_format", "Unknown"))

    # Add subtopics
    subtopics = report.get("subtopics", [])
    if subtopics:
        subtopic_text = "\n".join(
            [f"• {s.get('title', 'Unknown')}" for s in subtopics])
        table.add_row("Subtopics", subtopic_text)

    # Add source count
    source_count = len(report.get("sources", {}))
    table.add_row("Sources", f"{source_count} references")

    layout["body"].update(table)

    # Footer
    layout["footer"].update(
        Panel(
            "[italic]Use the export commands to save this report in your preferred format.[/italic]",
            style="dim"
        )
    )

    console.print(layout)


def display_report_preview(report: Dict[str, Any]):
    """Display a preview of the formatted report."""
    if not report or not report.get("final_document"):
        console.print("[bold red]No report content available.[/bold red]")
        return

    console.print(Panel("[bold]Report Preview[/bold]",
                  style="white on dark_blue"))

    # Get the first 1000 characters as a preview
    content = report.get("final_document", "")
    preview = content[:1000] + "..." if len(content) > 1000 else content

    # Display as markdown
    md = Markdown(preview)
    console.print(Panel(md, box=box.SIMPLE, expand=False))

    # Show options
    console.print()
    console.print("[dim]--- Preview truncated ---[/dim]")
    console.print(
        f"[italic]Full report contains {len(content)} characters.[/italic]")
    console.print()


async def export_report(report: Dict[str, Any], format_type: str, output_path: Optional[str] = None):
    """Export the report to the specified format."""
    if not report:
        console.print(
            "[bold red]No report content available to export.[/bold red]")
        return

    # Check for content in the report dictionary
    if not report.get("content") and not report.get("final_document"):
        console.print(
            "[bold red]No report content available to export.[/bold red]")
        return

    # Create default filename if none provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = report.get("query", "research")[
            :20].replace(" ", "_").lower()
        filename = f"{topic_slug}_{timestamp}.{format_type}"
        output_path = filename

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"Exporting to {format_type.upper()}...", total=100)

        # Simulate progress for better UX
        for i in range(1, 91, 10):
            progress.update(task, completed=i)
            await asyncio.sleep(0.1)

        try:
            # Export based on format
            if format_type == "markdown":
                # Just write the content to file
                content = report.get("content") or report.get("final_document")
                with open(output_path, "w") as f:
                    f.write(content)
            elif format_type == "docx":
                # Use the export_docx function from workflow_system
                export_docx(report, output_path)
            elif format_type == "pdf":
                # Use the export_pdf function from workflow_system
                export_pdf(report, output_path)
            elif format_type == "pptx":
                # Use the export_pptx function from workflow_system
                export_pptx(report, output_path)
            else:
                console.print(
                    f"[bold red]Unsupported format: {format_type}[/bold red]")
                return

            progress.update(task, completed=100)
            console.print(
                f"[bold green]Report exported successfully to:[/bold green] {output_path}")
            return output_path
        except Exception as e:
            console.print(
                f"[bold red]Error exporting report:[/bold red] {str(e)}")
            return None


async def run_with_progress(topic: str, document_format: str, citation_style: str,
                            research_depth: str, output_file: Optional[str] = None,
                            verbose: bool = False):
    """Run the research workflow with a rich progress display."""
    console.print(
        Panel(f"[bold]Starting Research on:[/bold] {topic}", style="white on dark_blue"))

    # Create progress display
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=not verbose
    ) as progress:
        # Create the main tasks
        overall_task = progress.add_task("[bold]Overall Progress", total=100)
        planning_task = progress.add_task(
            "Planning research approach...", total=100)
        research_task = progress.add_task(
            "Gathering sources...", total=100, visible=False)
        fact_check_task = progress.add_task(
            "Validating information...", total=100, visible=False)
        writing_task = progress.add_task(
            "Writing content...", total=100, visible=False)
        editing_task = progress.add_task(
            "Refining content...", total=100, visible=False)
        formatting_task = progress.add_task(
            "Formatting document...", total=100, visible=False)
        export_task = progress.add_task(
            "Preparing export...", total=100, visible=False)

        # Running the actual research process
        try:
            # Update planning progress
            progress.update(planning_task, visible=True)
            for i in range(1, 101, 10):
                progress.update(planning_task, completed=i)
                progress.update(overall_task, completed=i * 0.15)
                await asyncio.sleep(0.2)

            # Update research progress
            progress.update(planning_task, visible=False)
            progress.update(research_task, visible=True)
            for i in range(1, 101, 5):
                progress.update(research_task, completed=i)
                overall_progress = 15 + (i * 0.25)
                progress.update(overall_task, completed=min(
                    overall_progress, 100))
                await asyncio.sleep(0.2)

            # Update fact checking progress
            progress.update(research_task, visible=False)
            progress.update(fact_check_task, visible=True)
            for i in range(1, 101, 10):
                progress.update(fact_check_task, completed=i)
                overall_progress = 40 + (i * 0.15)
                progress.update(overall_task, completed=min(
                    overall_progress, 100))
                await asyncio.sleep(0.2)

            # Update writing progress
            progress.update(fact_check_task, visible=False)
            progress.update(writing_task, visible=True)
            for i in range(1, 101, 5):
                progress.update(writing_task, completed=i)
                overall_progress = 55 + (i * 0.2)
                progress.update(overall_task, completed=min(
                    overall_progress, 100))
                await asyncio.sleep(0.2)

            # Update editing progress
            progress.update(writing_task, visible=False)
            progress.update(editing_task, visible=True)
            for i in range(1, 101, 10):
                progress.update(editing_task, completed=i)
                overall_progress = 75 + (i * 0.1)
                progress.update(overall_task, completed=min(
                    overall_progress, 100))
                await asyncio.sleep(0.1)

            # Update formatting progress
            progress.update(editing_task, visible=False)
            progress.update(formatting_task, visible=True)
            for i in range(1, 101, 10):
                progress.update(formatting_task, completed=i)
                overall_progress = 85 + (i * 0.1)
                progress.update(overall_task, completed=min(
                    overall_progress, 100))
                await asyncio.sleep(0.1)

            # Update export progress
            progress.update(formatting_task, visible=False)
            progress.update(export_task, visible=True)
            for i in range(1, 101, 20):
                progress.update(export_task, completed=i)
                overall_progress = 95 + (i * 0.05)
                progress.update(overall_task, completed=min(
                    overall_progress, 100))
                await asyncio.sleep(0.05)

            # Finalize progress
            progress.update(overall_task, completed=100)
            progress.update(export_task, completed=100)

            # Call the actual report generation (with the progress display already shown for better UX)
            if verbose:
                console.print("[yellow]Generating final report...[/yellow]")

            report = generate_research_report(
                topic=topic,
                document_format=document_format,
                citation_style=citation_style,
                research_depth=research_depth
            )

            return report

        except Exception as e:
            console.print(
                f"[bold red]Error in research process:[/bold red] {str(e)}")
            return None


async def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Deep Research Agent CLI")
    parser.add_argument("--topic", type=str, help="Research topic")
    parser.add_argument("--format", type=str, default="markdown",
                        choices=["markdown", "md", "docx", "pdf", "pptx"],
                        help="Output document format")
    parser.add_argument("--citation", type=str, default="APA",
                        choices=["APA", "MLA", "Chicago"],
                        help="Citation style")
    parser.add_argument("--depth", type=str, default="comprehensive",
                        choices=["basic", "comprehensive", "exhaustive"],
                        help="Research depth")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output")

    args = parser.parse_args()

    # Display header
    display_header()

    # Get topic from args or prompt user
    topic = args.topic
    if not topic:
        topic = Prompt.ask("[bold blue]Enter a research topic[/bold blue]")

    # Run the research workflow with progress display
    report = await run_with_progress(
        topic=topic,
        document_format=args.format,
        citation_style=args.citation,
        research_depth=args.depth,
        output_file=args.output,
        verbose=args.verbose
    )

    if report:
        # Display report summary
        console.print()
        display_report_summary(report)

        # Display preview of the report
        console.print()
        display_report_preview(report)

        # Ask user if they want to export
        console.print()
        if Confirm.ask("[bold blue]Export this report now?[/bold blue]"):
            export_format = args.format
            output_path = args.output

            # If no output path, ask for one
            if not output_path:
                default_filename = f"{topic[:20].replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.{export_format}"
                output_path = Prompt.ask(
                    "[bold blue]Output filename[/bold blue]",
                    default=default_filename
                )

            # Export the report
            await export_report(report, export_format, output_path)

        # Final success message
        console.print()
        console.print(
            "[bold green]Research workflow completed successfully![/bold green]")
    else:
        console.print(
            "[bold red]Research workflow failed to generate a valid report.[/bold red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print(
            "\n[bold red]Research process interrupted by user.[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
        sys.exit(1)
