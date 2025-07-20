from typing import List, Dict, Any, Optional
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser, Document
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from pydantic import BaseModel, Field
import json
import time
import os
from pathlib import Path


class AgentResponse(BaseModel):
    """Standard response format for all agents"""
    agent_name: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class JSONOutputParser(BaseOutputParser):
    """Parser to extract JSON from LLM responses"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            # Try to find JSON in the response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
            else:
                # If no JSON found, return the text as content
                return {"content": text.strip()}
        except json.JSONDecodeError:
            return {"content": text.strip()}


class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, model_name: str = "llama3.1:8b"):
        self.name = name
        self.llm = Ollama(model=model_name, temperature=0.7)
        self.output_parser = JSONOutputParser()
    
    def _create_chain(self, template: str) -> LLMChain:
        """Create a LangChain chain with the given template"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "context"]
        )
        return LLMChain(llm=self.llm, prompt=prompt, output_parser=self.output_parser)
    
    def process(self, input_text: str, context: str = "") -> AgentResponse:
        """Process input and return standardized response"""
        raise NotImplementedError("Subclasses must implement process method")


class RAGAgent(BaseAgent):
    """Agent specialized in Retrieval-Augmented Generation"""
    
    def __init__(self, documents_path: Optional[str] = None):
        super().__init__("RAG Agent")
        self.embeddings = OllamaEmbeddings(model="llama3.1:8b")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vectorstore = None
        self.documents = []
        
        # Initialize with documents if path provided
        if documents_path:
            self.load_documents(documents_path)
        
        template = """You are a knowledgeable assistant with access to relevant documents. Use the retrieved context to answer questions accurately.

Question: {input}
Retrieved Context: {context}

Please provide a comprehensive answer based on the retrieved information. If the context doesn't contain enough information, clearly state what information is missing.

Format your response as JSON:
{{
    "answer": "Your detailed answer based on the context",
    "sources_used": ["source1", "source2"],
    "confidence": "high/medium/low",
    "additional_info_needed": "What additional information would be helpful"
}}

Response:"""
        
        self.chain = self._create_chain(template)
    
    def load_documents(self, documents_path: str):
        """Load documents from a directory or file"""
        try:
            if os.path.isfile(documents_path):
                loader = TextLoader(documents_path)
                documents = loader.load()
            elif os.path.isdir(documents_path):
                loader = DirectoryLoader(
                    documents_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                documents = loader.load()
            else:
                raise ValueError(f"Path {documents_path} is not a valid file or directory")
            
            # Split documents into chunks
            self.documents = self.text_splitter.split_documents(documents)
            
            # Create vector store
            if self.documents:
                self.vectorstore = FAISS.from_documents(
                    self.documents,
                    self.embeddings
                )
                print(f"✅ Loaded {len(self.documents)} document chunks into RAG system")
            else:
                print("⚠️ No documents found to load")
                
        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
    
    def add_text_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add text documents directly to the RAG system"""
        try:
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
            
            # Split documents into chunks
            chunked_docs = self.text_splitter.split_documents(documents)
            self.documents.extend(chunked_docs)
            
            # Update or create vector store
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(chunked_docs, self.embeddings)
            else:
                new_vectorstore = FAISS.from_documents(chunked_docs, self.embeddings)
                self.vectorstore.merge_from(new_vectorstore)
            
            print(f"✅ Added {len(chunked_docs)} new document chunks to RAG system")
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if self.vectorstore is None:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            return []
    
    def process(self, input_text: str, context: str = "") -> AgentResponse:
        """Process input with RAG retrieval"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(input_text)
        
        # Combine retrieved context with any additional context
        retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])
        full_context = f"{context}\n\nRetrieved Information:\n{retrieved_context}" if context else retrieved_context
        
        # Generate response using the chain
        result = self.chain.run(input=input_text, context=full_context)
        
        return AgentResponse(
            agent_name=self.name,
            content=str(result),
            metadata={
                "task_type": "rag_query",
                "retrieved_docs_count": len(relevant_docs),
                "has_vectorstore": self.vectorstore is not None,
                "input_query": input_text
            }
        )


class ResearchAgent(BaseAgent):
    """Agent specialized in conducting research and gathering information with RAG support"""
    
    def __init__(self, rag_agent: Optional[RAGAgent] = None):
        super().__init__("Research Agent")
        self.rag_agent = rag_agent
        
        template = """You are a research specialist. Your task is to conduct thorough research on the given topic.

Topic: {input}
Context: {context}

Please provide a comprehensive research summary including:
1. Key concepts and definitions
2. Main points and important facts
3. Different perspectives or approaches
4. Potential challenges or considerations

Format your response as JSON with the following structure:
{{
    "summary": "Brief overview of the topic",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "main_points": ["point1", "point2", "point3"],
    "perspectives": ["perspective1", "perspective2"],
    "challenges": ["challenge1", "challenge2"],
    "sources_consulted": "Information about sources used"
}}

Research Summary:"""
        
        self.chain = self._create_chain(template)
    
    def process(self, input_text: str, context: str = "") -> AgentResponse:
        enhanced_context = context
        
        # If RAG agent is available, retrieve relevant information
        if self.rag_agent and self.rag_agent.vectorstore is not None:
            rag_response = self.rag_agent.process(input_text)
            enhanced_context = f"{context}\n\nRAG Retrieved Information:\n{rag_response.content}"
        
        result = self.chain.run(input=input_text, context=enhanced_context)
        
        return AgentResponse(
            agent_name=self.name,
            content=str(result),
            metadata={
                "task_type": "research", 
                "input_topic": input_text,
                "rag_enhanced": self.rag_agent is not None
            }
        )


class WriterAgent(BaseAgent):
    """Agent specialized in creating written content"""
    
    def __init__(self):
        super().__init__("Writer Agent")
        
        template = """You are a professional writer. Your task is to create high-quality content based on the research provided.

Topic: {input}
Research Data: {context}

Please write a well-structured article that:
1. Has a clear introduction, body, and conclusion
2. Uses the research data effectively
3. Is engaging and informative
4. Maintains a professional tone

Format your response as JSON:
{{
    "title": "Article Title",
    "introduction": "Opening paragraph",
    "body": "Main content paragraphs",
    "conclusion": "Closing paragraph",
    "word_count": "estimated word count"
}}

Article:"""
        
        self.chain = self._create_chain(template)
    
    def process(self, input_text: str, context: str = "") -> AgentResponse:
        result = self.chain.run(input=input_text, context=context)
        
        return AgentResponse(
            agent_name=self.name,
            content=str(result),
            metadata={"task_type": "writing", "input_topic": input_text}
        )


class CriticAgent(BaseAgent):
    """Agent specialized in reviewing and providing feedback"""
    
    def __init__(self):
        super().__init__("Critic Agent")
        
        template = """You are a professional critic and editor. Your task is to review the content and provide constructive feedback.

Original Topic: {input}
Content to Review: {context}

Please provide a thorough review including:
1. Overall assessment of quality
2. Strengths of the content
3. Areas for improvement
4. Specific suggestions for enhancement
5. A numerical score (1-10)

Format your response as JSON:
{{
    "overall_assessment": "Brief overall evaluation",
    "strengths": ["strength1", "strength2", "strength3"],
    "improvements": ["improvement1", "improvement2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "score": 8,
    "final_verdict": "Recommendation (approve/revise/reject)"
}}

Review:"""
        
        self.chain = self._create_chain(template)
    
    def process(self, input_text: str, context: str = "") -> AgentResponse:
        result = self.chain.run(input=input_text, context=context)
        
        return AgentResponse(
            agent_name=self.name,
            content=str(result),
            metadata={"task_type": "review", "input_topic": input_text}
        )


class CoordinatorAgent(BaseAgent):
    """Agent that coordinates the workflow between other agents with RAG support"""
    
    def __init__(self, rag_agent: Optional[RAGAgent] = None):
        super().__init__("Coordinator Agent")
        self.rag_agent = rag_agent
        self.research_agent = ResearchAgent(rag_agent)
        self.writer_agent = WriterAgent()
        self.critic_agent = CriticAgent()
    
    def orchestrate_workflow(self, topic: str) -> Dict[str, AgentResponse]:
        """Orchestrate the complete workflow from research to final content"""
        
        print(f"🚀 Starting multi-agent workflow for topic: '{topic}'")
        if self.rag_agent and self.rag_agent.vectorstore is not None:
            print("📚 RAG system is active - will use retrieved knowledge")
        print("-" * 60)
        
        # Step 0: RAG Query (if available)
        rag_response = None
        if self.rag_agent and self.rag_agent.vectorstore is not None:
            print("🔍 Phase 0: RAG Agent retrieving relevant information...")
            rag_response = self.rag_agent.process(topic)
            print(f"✅ Information retrieved by {rag_response.agent_name}")
            print(f"📖 Retrieved info preview: {rag_response.content[:100]}...")
            print()
        
        # Step 1: Research
        print("📚 Phase 1: Research Agent conducting research...")
        research_response = self.research_agent.process(topic)
        print(f"✅ Research completed by {research_response.agent_name}")
        print(f"📊 Research summary: {research_response.content[:100]}...")
        print()
        
        # Step 2: Writing
        print("✍️ Phase 2: Writer Agent creating content...")
        writer_context = research_response.content
        if rag_response:
            writer_context = f"{research_response.content}\n\nAdditional RAG Context:\n{rag_response.content}"
        writer_response = self.writer_agent.process(topic, writer_context)
        print(f"✅ Content created by {writer_response.agent_name}")
        print(f"📝 Content preview: {writer_response.content[:100]}...")
        print()
        
        # Step 3: Review
        print("🔍 Phase 3: Critic Agent reviewing content...")
        critic_response = self.critic_agent.process(topic, writer_response.content)
        print(f"✅ Review completed by {critic_response.agent_name}")
        print(f"📋 Review summary: {critic_response.content[:100]}...")
        print()
        
        # Step 4: Final coordination
        print("🎯 Phase 4: Coordinator finalizing workflow...")
        workflow_result = {
            "rag": rag_response,
            "research": research_response,
            "content": writer_response,
            "review": critic_response,
            "workflow_status": "completed",
            "topic": topic
        }
        
        print("🎉 Multi-agent workflow completed successfully!")
        print("-" * 60)
        
        return workflow_result
    
    def process(self, input_text: str, context: str = "") -> AgentResponse:
        """Process input through the complete workflow"""
        workflow_result = self.orchestrate_workflow(input_text)
        
        return AgentResponse(
            agent_name=self.name,
            content=json.dumps(workflow_result, indent=2, default=str),
            metadata={
                "task_type": "coordination",
                "workflow_phases": ["research", "writing", "review"],
                "topic": input_text
            }
        )


class MultiAgentSystem:
    """Main system that manages all agents and provides the interface with RAG support"""
    
    def __init__(self, documents_path: Optional[str] = None):
        # Initialize RAG agent
        self.rag_agent = RAGAgent(documents_path) if documents_path else RAGAgent()
        
        # Initialize coordinator with RAG support
        self.coordinator = CoordinatorAgent(self.rag_agent)
        
        # Update agents dictionary
        self.agents = {
            "rag": self.rag_agent,
            "research": self.coordinator.research_agent,
            "writer": self.coordinator.writer_agent,
            "critic": self.coordinator.critic_agent,
            "coordinator": self.coordinator
        }
    
    def add_documents_to_rag(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the RAG system"""
        self.rag_agent.add_text_documents(texts, metadatas)
    
    def load_documents_from_path(self, documents_path: str):
        """Load documents from a file or directory path"""
        self.rag_agent.load_documents(documents_path)
    
    def run_workflow(self, topic: str) -> Dict[str, AgentResponse]:
        """Run the complete multi-agent workflow"""
        return self.coordinator.orchestrate_workflow(topic)
    
    def run_single_agent(self, agent_name: str, input_text: str, context: str = "") -> AgentResponse:
        """Run a specific agent independently"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}")
        
        agent = self.agents[agent_name]
        return agent.process(input_text, context)
    
    def list_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get status of the RAG system"""
        return {
            "has_vectorstore": self.rag_agent.vectorstore is not None,
            "documents_count": len(self.rag_agent.documents),
            "embeddings_model": "llama3.1:8b"
        }


def main():
    """Main function to demonstrate the multi-agent system with RAG"""
    
    print("🤖 LangChain Multi-Agent System with Ollama and RAG")
    print("=" * 60)
    
    # Initialize the system
    system = MultiAgentSystem()
    
    # Add some sample documents to demonstrate RAG
    sample_documents = [
        """
        Artificial Intelligence in Education: Transforming Learning
        
        AI technologies are revolutionizing education through personalized learning experiences,
        intelligent tutoring systems, and automated assessment tools. Machine learning algorithms
        can adapt to individual student needs, providing customized content and pacing.
        
        Key benefits include:
        - Personalized learning paths
        - Real-time feedback and assessment
        - Improved accessibility for diverse learners
        - Enhanced teacher productivity through automation
        """,
        """
        Challenges of AI in Education
        
        While AI offers significant benefits, there are important challenges to consider:
        
        1. Privacy and Data Security: Student data protection is paramount
        2. Digital Divide: Ensuring equitable access to AI-powered tools
        3. Teacher Training: Educators need support to integrate AI effectively
        4. Ethical Considerations: Bias in algorithms and decision-making transparency
        5. Cost and Infrastructure: Implementation requires significant investment
        """,
        """
        Future of AI-Enhanced Learning
        
        The future of AI in education includes:
        - Virtual and Augmented Reality integration
        - Natural Language Processing for better human-computer interaction
        - Predictive analytics for early intervention
        - Collaborative AI systems that support both students and teachers
        - Continuous assessment and adaptive curricula
        
        Research shows that AI-enhanced learning can improve student outcomes
        by 20-30% when implemented effectively.
        """
    ]
    
    # Add documents to RAG system
    print("📚 Adding sample documents to RAG system...")
    system.add_documents_to_rag(
        sample_documents,
        [
            {"source": "AI_Education_Overview", "type": "benefits"},
            {"source": "AI_Education_Challenges", "type": "challenges"},
            {"source": "AI_Education_Future", "type": "future_trends"}
        ]
    )
    
    # Show RAG status
    rag_status = system.get_rag_status()
    print(f"RAG Status: {rag_status}")
    print()
    
    # Example topic for demonstration
    topic = "The Impact of Artificial Intelligence on Modern Education"
    
    print(f"Available agents: {system.list_agents()}")
    print()
    
    try:
        # Demonstrate RAG agent independently
        print("🔍 Testing RAG Agent independently...")
        rag_result = system.run_single_agent("rag", topic)
        print(f"RAG Response preview: {rag_result.content[:200]}...")
        print()
        
        # Run the complete workflow
        results = system.run_workflow(topic)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 WORKFLOW RESULTS")
        print("=" * 60)
        
        for phase, response in results.items():
            if isinstance(response, AgentResponse):
                print(f"\n🔹 {response.agent_name.upper()}")
                print("-" * 40)
                
                # Try to parse JSON content for better display
                try:
                    content_data = json.loads(response.content)
                    print(json.dumps(content_data, indent=2))
                except (json.JSONDecodeError, TypeError):
                    print(response.content)
                
                print(f"\n⏱️ Timestamp: {time.ctime(response.timestamp)}")
                print(f"📋 Metadata: {response.metadata}")
            elif phase == "workflow_status":
                print(f"\n🎯 Workflow Status: {response}")
            elif phase == "topic":
                print(f"\n📝 Topic: {response}")
        
        print("\n" + "=" * 60)
        print("✅ Multi-agent workflow with RAG completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running workflow: {str(e)}")
        print("Make sure Ollama is running and llama3.1:8b model is available")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
