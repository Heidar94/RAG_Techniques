"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using a local LLM with LM Studio.
"""

import json
import os
import sys
from typing import List, Dict, Any
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import BaseModel, Field

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Initialize LLM (LM Studio)
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    model_name="google/gemma-3-12b",
    temperature=0.3,
    max_tokens=1000
)

class EvaluationScores(BaseModel):
    relevance: int = Field(..., ge=1, le=5, description="Relevance score from 1 to 5")
    completeness: int = Field(..., ge=1, le=5, description="Completeness score from 1 to 5")
    accuracy: int = Field(..., ge=1, le=5, description="Accuracy score from 1 to 5")

def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using test questions and metrics.
    
    Args:
        retriever: The retriever component to evaluate
        num_questions: Number of test questions to generate
    
    Returns:
        Dict containing evaluation metrics
    """
    # Create evaluation chain with JSON output
    output_parser = PydanticOutputParser(pydantic_object=EvaluationScores)
    
    # Create a more strict evaluation prompt
    eval_prompt = PromptTemplate(
        template="""
        Evalúa los siguientes resultados de recuperación para la pregunta.
        Pregunta: {question}
        Contexto Recuperado: {context}
        
        Califica en una escala del 1-5 (5 siendo el mejor) para:
        1. Relevancia: ¿Qué tan relevante es la información recuperada para la pregunta?
        2. Completitud: ¿El contexto contiene toda la información necesaria?
        3. Precisión: ¿La información recuperada es precisa y correcta?
        
        Responde ÚNICAMENTE con un objeto JSON válido que contenga EXACTAMENTE las siguientes claves: 
        - "relevance" (número entero del 1 al 5)
        - "completeness" (número entero del 1 al 5)
        - "accuracy" (número entero del 1 al 5)
        
        No incluyas ningún otro texto, explicación o formato adicional.
        """,
        input_variables=["question", "context"]
    )
    
    # Create the evaluation chain with output fixing
    eval_chain = eval_prompt | llm | output_parser
    
    # Generate test questions
    question_gen_prompt = PromptTemplate.from_template(
        "Genera {num_questions} preguntas diversas sobre el tema del documento:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    
    try:
        # Generate questions
        questions_text = question_chain.invoke({"num_questions": num_questions})
        questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        
        # Evaluate each question
        results = []
        for question in questions:
            try:
                # Get retrieval results
                context_docs = retriever.get_relevant_documents(question)
                context_text = "\n\n".join([doc.page_content for doc in context_docs])
                
                # Get evaluation with retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Get evaluation
                        eval_data = eval_chain.invoke({
                            "question": question,
                            "context": context_text
                        })
                        
                        # Convert Pydantic model to dict
                        eval_data = eval_data.dict()
                        
                        results.append({
                            "question": question,
                            "context": context_text,
                            "evaluation": eval_data
                        })
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt == max_retries - 1:  # Last attempt
                            print(f"Error en la evaluación (intento {attempt + 1}/{max_retries}) para: {question[:100]}...")
                            print(f"Error: {str(e)}")
                            continue
                
            except Exception as e:
                print(f"Error procesando pregunta '{question}': {str(e)}")
                continue
        
        # Calculate average scores
        avg_scores = calculate_average_scores(results)
        
        return {
            "questions": [r["question"] for r in results],
            "results": results,
            "average_scores": avg_scores
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "details": "Error durante la evaluación del RAG"
        }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate average scores across all evaluation results.
    """
    if not results:
        return {}
    
    total_scores = {
        "relevance": 0,
        "completeness": 0,
        "accuracy": 0,
        "count": 0
    }
    
    for result in results:
        eval_data = result.get("evaluation", {})
        if not isinstance(eval_data, dict):
            continue
            
        total_scores["relevance"] += float(eval_data.get("relevance", 0))
        total_scores["completeness"] += float(eval_data.get("completeness", 0))
        total_scores["accuracy"] += float(eval_data.get("accuracy", 0))
        total_scores["count"] += 1
    
    if total_scores["count"] == 0:
        return {}
    
    return {
        "average_relevance": total_scores["relevance"] / total_scores["count"],
        "average_completeness": total_scores["completeness"] / total_scores["count"],
        "average_accuracy": total_scores["accuracy"] / total_scores["count"]
    }

if __name__ == "__main__":
    print("Módulo de evaluación RAG para LM Studio")
    print("Importa este módulo y usa la función evaluate_rag() con tu retriever.")