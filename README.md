# Deep Research Agent

A comprehensive research workflow system powered by AI, featuring a multi-agent architecture using Langgraph for orchestration and Pydantic for strong typing.

## Overview

The Deep Research Agent is an advanced research tool that leverages multiple specialized AI agents working together to generate comprehensive research reports on any topic. Each agent handles a specific part of the research process, from planning and source gathering to fact checking, writing, editing, and formatting.

### Key Features

- **Multi-Agent Workflow**: Specialized agents collaborate in a structured workflow
- **Strong Type Safety**: Pydantic models ensure consistent data throughout the workflow
- **Multiple Output Formats**: Export to Markdown, Word, PDF, or PowerPoint
- **Rich Terminal UI**: Detailed progress reporting with the Rich library
- **Streamlit Web Interface**: Modern, responsive web UI for easy interaction
- **Modular Architecture**: Easy to extend with new capabilities

## Architecture

The system uses a Langgraph-based workflow with the following specialized agents:

1. **Planner Agent**: Creates a structured research plan with subtopics and approach
2. **Researcher Agent**: Gathers credible sources for each subtopic
3. **Fact Checker Agent**: Validates claims against sources for accuracy
4. **Writer Agent**: Creates content based on validated research
5. **Editor Agent**: Refines and improves the content
6. **Formatter Agent**: Structures the document according to academic standards
7. **Document Exporter**: Converts to various output formats

## Project Structure

```
deep_research_agent/
├── cli.py                # Command line interface using Rich
├── streamlit_app.py      # Streamlit web application
├── workflow_system.py    # Core workflow orchestration system
├── requirements.txt      # Project dependencies
├── .env.example          # Example environment variables
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/deep_research_agent.git
   cd deep_research_agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

## Usage

### Command Line Interface

The CLI uses the Rich library to provide a beautiful terminal experience:

```bash
python cli.py --topic "The impact of artificial intelligence on healthcare"
```

Command line options:

- `--topic`: Research topic to investigate
- `--format`: Output format (markdown, docx, pdf, pptx)
- `--citation`: Citation style (APA, MLA, Chicago)
- `--depth`: Research depth (basic, comprehensive, exhaustive)
- `--output`: Output file path
- `--verbose`: Enable verbose output

### Streamlit Web Interface

To run the Streamlit web app:

```bash
streamlit run streamlit_app.py
```

The web interface offers:

- Interactive research form
- Real-time progress tracking
- Comprehensive report display
- Multiple export options
- Detailed metadata views

## Core Components

### Workflow System

The `workflow_system.py` module contains:

- **Pydantic Models**: Strongly typed data structures for the research workflow
- **LangGraph Implementation**: Orchestration of specialized agents
- **Export Functionality**: Conversion to various document formats
- **Workflow Engine**: Manages the entire research process

### Interfaces

Two interfaces are provided:

1. **CLI (cli.py)**: A Rich-powered terminal interface with progress tracking
2. **Streamlit (streamlit_app.py)**: A modern web interface with interactive controls

## Example Workflow

1. The user submits a research topic
2. The Planner Agent creates a research plan with subtopics
3. The Researcher Agent gathers credible sources for each subtopic
4. The Fact Checker Agent validates claims against sources
5. The Writer Agent creates content based on validated research
6. The Editor Agent refines and improves the content
7. The Formatter Agent structures the document
8. The output is exported in the requested format

## Extending the System

The modular architecture makes it easy to extend with new capabilities:

1. Add new agent types in `workflow_system.py`
2. Create new Pydantic models for structured data
3. Modify the Langgraph workflow to incorporate new steps
4. Update the interfaces to expose new functionality

## License

[MIT License](LICENSE)

## Acknowledgements

- [OpenAI](https://openai.com/) for the language models
- [Langgraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- [Rich](https://rich.readthedocs.io/) for terminal UI
- [Streamlit](https://streamlit.io/) for the web interface
