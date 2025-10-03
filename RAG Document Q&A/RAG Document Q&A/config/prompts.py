"""
Prompt templates for the Document Intelligence Platform
"""

from langchain_core.prompts import ChatPromptTemplate

class PromptTemplates:
    """Collection of prompt templates for different use cases"""
    
    @staticmethod
    def get_default_qa_prompt():
        """Get the default Q&A prompt template"""
        return ChatPromptTemplate.from_template(
            """
            You are an expert research assistant analyzing academic documents. 
            Provide comprehensive, accurate answers based strictly on the provided context.
            
            GUIDELINES:
            - Answer the question using only the information from the provided context
            - If the information is not in the context, clearly state this
            - Provide detailed explanations when appropriate
            - Include relevant technical details from the research papers
            - Structure your response clearly and professionally
            
            CONTEXT: {context}
            
            QUESTION: {input}
            
            Please provide a thorough, well-structured answer:
            """
        )
    
    @staticmethod
    def get_summary_prompt():
        """Get prompt template for document summarization"""
        return ChatPromptTemplate.from_template(
            """
            You are an expert document analyzer. Please provide a comprehensive summary of the following document content.
            
            INSTRUCTIONS:
            - Create a detailed summary highlighting key points
            - Identify main topics and themes
            - Extract important findings or conclusions
            - Structure the summary with clear sections
            
            DOCUMENT CONTENT: {context}
            
            Please provide a structured summary:
            """
        )
    
    @staticmethod
    def get_technical_analysis_prompt():
        """Get prompt template for technical document analysis"""
        return ChatPromptTemplate.from_template(
            """
            You are a technical expert analyzing research documents. Provide a detailed technical analysis.
            
            FOCUS AREAS:
            - Methodology and approaches used
            - Technical specifications and requirements
            - Implementation details
            - Performance metrics and results
            - Limitations and future work
            
            CONTEXT: {context}
            
            QUESTION: {input}
            
            Provide a technical analysis addressing the question:
            """
        )
    
    @staticmethod
    def get_comparison_prompt():
        """Get prompt template for comparing documents or concepts"""
        return ChatPromptTemplate.from_template(
            """
            You are an expert analyst comparing documents or concepts. Provide a detailed comparison.
            
            COMPARISON GUIDELINES:
            - Identify similarities and differences
            - Highlight unique aspects of each
            - Provide balanced analysis
            - Structure comparison clearly
            
            CONTEXT: {context}
            
            COMPARISON REQUEST: {input}
            
            Provide a structured comparison:
            """
        )