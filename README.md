# Overview

This is a Python-based multi-agent AI system built with CrewAI that orchestrates three specialized agents (Research Analyst, Content Writer, and Quality Reviewer) to perform collaborative content creation tasks. The system leverages SambaNova's AI models through OpenAI-compatible APIs to enable sophisticated research, writing, and review workflows.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Multi-Agent Framework
- **CrewAI Integration**: Uses CrewAI framework to orchestrate multiple AI agents working together
- **Agent Specialization**: Three distinct agent roles with specific goals and backstories:
  - Research Analyst: Data gathering and analysis
  - Content Writer: Content creation from research findings
  - Quality Reviewer: Content review and quality assurance
- **Sequential Processing**: Agents work in a coordinated workflow to complete complex tasks

## AI Model Integration
- **SambaNova LLM**: Primary AI provider using the "openai/gpt-oss-120b" model
- **OpenAI Compatibility**: Uses OpenAI client library with custom base URL for SambaNova API
- **Model Configuration**: Standardized LLM settings with temperature=0.1 and top_p=0.1 for consistent, focused outputs
- **Centralized LLM Factory**: Single function to create configured LLM instances for all agents

## Configuration Management
- **Environment Variables**: Secure API key storage using dotenv
- **Modular Design**: Separate agent definitions with individual configurations
- **Reusable Components**: Shared LLM configuration across all agents

# External Dependencies

## AI Services
- **SambaNova AI**: Primary LLM provider accessed via API at https://api.sambanova.ai/v1
- **OpenAI Client Library**: Used for API communication compatibility

## Python Libraries
- **CrewAI**: Multi-agent orchestration framework
- **OpenAI**: API client for LLM communication
- **python-dotenv**: Environment variable management

## Authentication
- **API Key Authentication**: Requires SAMBANOVA_API_KEY environment variable for service access
