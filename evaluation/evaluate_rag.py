from typing import Dict, Any, List
import re
import json
import logging
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from pydantic import BaseModel, Field

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationScores(BaseModel):
    """Modelo Pydantic para validar las puntuaciones de evaluación."""
    relevance: int = Field(..., ge=1, le=5, description="Relevance score from 1 to 5")
    completeness: int = Field(..., ge=1, le=5, description="Completeness score from 1 to 5")
    conciseness: int = Field(..., ge=1, le=5, description="Conciseness score from 1 to 5")

def parse_evaluation(text: str) -> Dict[str, int]:
    """Parsea la respuesta de evaluación a un diccionario de puntuaciones."""
    try:
        # Buscar JSON en la respuesta
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "relevance": int(data.get("relevance", 1)),
                "completeness": int(data.get("completeness", 1)),
                "conciseness": int(data.get("conciseness", 1))
            }
    except Exception as e:
        logger.warning(f"Error al parsear evaluación: {str(e)}")
    
    # Valores por defecto en caso de error
    return {"relevance": 1, "completeness": 1, "conciseness": 1}

def extract_questions_from_text(text: str) -> List[str]:
    """
    Extrae preguntas numeradas de un texto.
    
    Args:
        text: Texto que contiene preguntas numeradas
        
    Returns:
        Lista de preguntas extraídas
    """
    questions = []
    # Buscar líneas que empiecen con número seguido de punto y espacio
    pattern = r'^\d+\.\s*(.+?)(?=\n\d+\.|\Z)'
    matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        question = match.group(1).strip()
        # Limpiar cualquier carácter de nueva línea dentro de la pregunta
        question = ' '.join(question.split())
        questions.append(question)
    
    return questions

def evaluate_rag(retriever: VectorStore, num_questions: int = 5, text_with_questions: str = None) -> Dict[str, Any]:
    """
    Evalúa un sistema RAG usando preguntas extraídas de un texto.
    
    Args:
        retriever: Componente de recuperación a evaluar
        num_questions: Número máximo de preguntas a evaluar
        text_with_questions: Texto que contiene las preguntas numeradas
        
    Returns:
        Diccionario con métricas de evaluación
    """
    # Inicializar el modelo de lenguaje
    llm = ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="lm-studio",
        model_name="deepseek/deepseek-r1-0528-qwen3-8b",
        temperature=0.3,
        max_tokens=1000
    )
    
    # Extraer preguntas del texto
    questions = extract_questions_from_text(text_with_questions)
    questions = questions[:num_questions]  # Limitar al número solicitado
    
    if not questions:
        raise ValueError("No se encontraron preguntas en el texto proporcionado")
    
    logger.info(f"Preguntas extraídas: {questions}")
    
    # Prompt para evaluación
    eval_prompt = PromptTemplate(
        template="""
        Evalúa los siguientes resultados de recuperación para la pregunta.
        
        Pregunta: {question}
        Contexto Recuperado: {context}
        
        Califica en una escala del 1-5 (5 siendo el mejor) para:
        1. Relevancia: ¿Qué tan relevante es la información recuperada para la pregunta?
        2. Completitud: ¿El contexto contiene toda la información necesaria?
        3. Concisión: ¿La información recuperada es concisa y sin información irrelevante?
        
        Proporciona las calificaciones en formato JSON con estas claves exactas:
        {{
            "relevance": <puntuación 1-5>,
            "completeness": <puntuación 1-5>,
            "conciseness": <puntuación 1-5>
        }}
        """,
        input_variables=["question", "context"]
    )
    
    eval_chain = eval_prompt | llm | StrOutputParser()
    
    try:
        # Evaluar cada pregunta
        results = []
        for question in questions:  
            try:
                # Obtener contexto usando el método correcto del retriever
                context_docs = retriever.get_relevant_documents(question)
                context_text = "\n\n".join([doc.page_content for doc in context_docs])
                
                # Evaluar
                eval_text = eval_chain.invoke({
                    "question": question,
                    "context": context_text
                })
                
                # Parsear evaluación
                evaluation = parse_evaluation(eval_text)
                
                results.append({
                    "question": question,
                    "context": context_text,
                    "evaluation": evaluation
                })
                
                logger.debug(f"Pregunta evaluada: {question[:50]}...")
                
            except Exception as e:
                logger.error(f"Error al evaluar pregunta: {str(e)}")
                results.append({
                    "question": question,
                    "error": str(e),
                    "evaluation": {"relevance": 1, "completeness": 1, "conciseness": 1}
                })
        
        # Calcular promedios
        if results:
            avg_scores = {
                "average_relevance": round(sum(r["evaluation"].get("relevance", 1) for r in results) / len(results), 2),
                "average_completeness": round(sum(r["evaluation"].get("completeness", 1) for r in results) / len(results), 2),
                "average_conciseness": round(sum(r["evaluation"].get("conciseness", 1) for r in results) / len(results), 2)
            }
        else:
            avg_scores = {
                "average_relevance": 0,
                "average_completeness": 0,
                "average_conciseness": 0
            }
        
        return {
            "questions": questions,
            "results": results,
            "average_scores": avg_scores
        }
        
    except Exception as e:
        logger.error(f"Error en la evaluación RAG: {str(e)}")
        return {
            "error": str(e),
            "questions": [],
            "results": [],
            "average_scores": {
                "average_relevance": 0,
                "average_completeness": 0,
                "average_conciseness": 0
            }
        }

# Función auxiliar para pruebas
if __name__ == "__main__":
    
    pass