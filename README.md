# AI Engineering Sprint 0

This sprint is dedicated to setting up your development environment and making sure that all configuration works as intended.

## Documentation

Project design and concept is found [here](https://github.com/anthonyckleung/ai-engineering-bootcamp-sprint-00/tree/24d468d3856edf92cc0b15c16ca362e245049354/documentation/project_design)

## Running the code
- Clone the repository.
- Run:
```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=your_google_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

To build the project, run:

```bash
make build-docker-streamlit
```

To run the project:

```bash
make run-docker-streamlit
```

