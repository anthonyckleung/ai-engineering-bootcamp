from langsmith import Client 
import os


coordinator_eval_dataset = [
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Whats is the weather today?"}
            ]
        },
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "What is job 10397 about?"}
            ]
        },
        "outputs": {
            "next_agent": "job_posting_qa_agent",
            "coordinator_final_answer": False
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you classify a posting with ID 10397 and tell me if it's real or fake?"}
            ]
        },
        "outputs": {
            "next_agent": "classifier_agent",
            "coordinator_final_answer": False
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you classify multiple job postings?"}
            ]
        },
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Give me examples of fake job postings?"}
            ]
        },
        "outputs": {
            "next_agent": "classifier_agent",
            "coordinator_final_answer": False
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "What of kind things can you classify?"}
            ]
        },
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you help me with my request?"}
            ]
        },
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True
        }
    },
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "Can you find me 2 suspicious data job postings?"}
            ]
        },
        "outputs": {
            "next_agent": "job_posting_qa_agent",
            "coordinator_final_answer": False
        }
    }
]