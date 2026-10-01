# AI Project Prompt Generator

A two-stage AI prompt engineering application powered by Streamlit, LangChain, and OpenRouter.

## Workflow

Raw Project Prompt
        ↓
LLM #1
        ↓
Magic Prompt
        ↓
LLM #2
        ↓
Project Instructions

## Features

- Generate an engineered Magic Prompt
- Generate persistent Project Instructions
- Download both outputs as `.txt` files
- Uses OpenRouter
- Uses separate LLM calls for each generation stage
- No prompt data is persisted by the application

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenRouter

## Deployment

Designed for deployment using Streamlit Community Cloud.

## Environment Variables

The application requires:

OPENROUTER_API_KEY
