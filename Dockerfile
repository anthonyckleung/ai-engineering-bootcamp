FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation and Python optimization
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONOPTIMIZE=1
ENV UV_LINK_MODE=copy

# Set Python path to include the src directory for imports
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Copy only dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy application code
COPY src/chatbot-ui ./src/chatbot-ui/

# Pre-compile Python files to bytecode
RUN python -m compileall ./src/chatbot-ui

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user and set permissions
RUN addgroup --system app && \
adduser --system --ingroup app --home /home/app app && \
    mkdir -p /home/app && \
    chown -R app:app /app

# Set HOME environment variable (NEW!)
ENV HOME=/home/app

# Switch to non-root user
USER app

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/chatbot-ui/streamlit_app.py", "--server.address=0.0.0.0"]