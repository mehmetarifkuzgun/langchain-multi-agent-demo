"""
Configuration settings for the multi-agent system.
Modify these settings to customize the behavior of your agents.
"""

# Ollama Model Configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Agent Configuration
AGENT_SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "timeout": 60  # seconds
}

# Workflow Configuration
WORKFLOW_SETTINGS = {
    "max_iterations": 3,
    "quality_threshold": 8.0,  # Minimum score for content approval
    "enable_iterative_improvement": True
}

# Output Configuration
OUTPUT_SETTINGS = {
    "save_results": True,
    "output_directory": "results",
    "timestamp_format": "%Y%m%d_%H%M%S"
}

# Agent Prompts - Customize these to change agent behavior
AGENT_PROMPTS = {
    "research_agent": {
        "system_message": "You are a thorough research specialist with expertise in gathering and analyzing information.",
        "task_description": "Conduct comprehensive research and provide structured findings."
    },
    
    "writer_agent": {
        "system_message": "You are a professional content writer skilled in creating engaging and informative articles.",
        "task_description": "Create well-structured, high-quality written content based on research data."
    },
    
    "critic_agent": {
        "system_message": "You are an experienced editor and critic with a keen eye for quality and improvement.",
        "task_description": "Provide constructive feedback and suggestions for content improvement."
    },
    
    "coordinator_agent": {
        "system_message": "You are a project coordinator managing the workflow between different agents.",
        "task_description": "Orchestrate the collaboration between agents to achieve the best results."
    }
}

# Example Topics for Testing
EXAMPLE_TOPICS = [
    "The Future of Artificial Intelligence in Education",
    "Sustainable Urban Planning for Smart Cities",
    "The Role of Blockchain in Supply Chain Management",
    "Mental Health in the Remote Work Era",
    "Climate Change Adaptation Strategies",
    "The Ethics of Gene Editing Technology",
    "Renewable Energy Storage Solutions",
    "Digital Privacy in the Age of Big Data"
]
