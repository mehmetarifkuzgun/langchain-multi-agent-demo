import streamlit as st
import json
import time
from typing import Dict, Any, List, Optional
from multi_agent_system import MultiAgentSystem, AgentResponse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import traceback


class MultiAgentUI:
    """Streamlit UI for the Multi-Agent System"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'system' not in st.session_state:
            st.session_state.system = None
        if 'workflow_results' not in st.session_state:
            st.session_state.workflow_results = None
        if 'workflow_history' not in st.session_state:
            st.session_state.workflow_history = []
        if 'rag_documents' not in st.session_state:
            st.session_state.rag_documents = []
    
    def setup_page(self):
        """Setup the Streamlit page configuration"""
        st.set_page_config(
            page_title="🤖 Multi-Agent System with RAG",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .agent-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        .agent-header {
            font-size: 18px;
            font-weight: bold;
            color: #2E86AB;
            margin-bottom: 10px;
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        .workflow-step {
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #2E86AB;
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with system controls"""
        st.sidebar.title("🛠️ System Configuration")
        
        # Initialize System Button
        if st.sidebar.button("🚀 Initialize Multi-Agent System", type="primary"):
            with st.spinner("Initializing system..."):
                try:
                    st.session_state.system = MultiAgentSystem()
                    st.sidebar.success("✅ System initialized successfully!")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
        
        # System Status
        if st.session_state.system:
            st.sidebar.success("🟢 System Online")
            
            # RAG Configuration Section
            st.sidebar.subheader("📚 RAG System")
            rag_status = st.session_state.system.get_rag_status()
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Documents", rag_status["documents_count"])
            with col2:
                status_icon = "✅" if rag_status["has_vectorstore"] else "❌"
                st.metric("Vector Store", status_icon)
            
            # Document Upload Section
            st.sidebar.subheader("📄 Add Documents")
            
            # Text input for documents
            document_text = st.sidebar.text_area(
                "Add document text:",
                height=100,
                placeholder="Enter your document text here..."
            )
            
            document_metadata = st.sidebar.text_input(
                "Document source/title:",
                placeholder="e.g., 'Research Paper 2024'"
            )
            
            if st.sidebar.button("➕ Add Document"):
                if document_text.strip():
                    try:
                        metadata = {"source": document_metadata or "User Input", "timestamp": datetime.now().isoformat()}
                        st.session_state.system.add_documents_to_rag([document_text], [metadata])
                        st.session_state.rag_documents.append({
                            "text": document_text[:100] + "..." if len(document_text) > 100 else document_text,
                            "source": document_metadata or "User Input",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.sidebar.success("📄 Document added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"❌ Error adding document: {str(e)}")
                else:
                    st.sidebar.warning("⚠️ Please enter some text")
            
            # Show added documents
            if st.session_state.rag_documents:
                st.sidebar.subheader("📋 Added Documents")
                for i, doc in enumerate(st.session_state.rag_documents[-3:]):  # Show last 3
                    with st.sidebar.expander(f"📄 {doc['source'][:30]}..."):
                        st.write(f"**Added:** {doc['timestamp']}")
                        st.write(f"**Preview:** {doc['text']}")
        else:
            st.sidebar.warning("🔴 System Not Initialized")
    
    def render_main_interface(self):
        """Render the main interface"""
        st.title("🤖 Multi-Agent System with RAG")
        st.markdown("---")
        
        if not st.session_state.system:
            st.warning("⚠️ Please initialize the system using the sidebar.")
            return
        
        # Main input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            topic = st.text_input(
                "🎯 Enter your topic or question:",
                placeholder="e.g., 'The Impact of Artificial Intelligence on Modern Education'",
                key="main_topic"
            )
        
        with col2:
            st.write("")  # Spacing
            run_workflow = st.button("🚀 Run Complete Workflow", type="primary")
        
        # Agent selection for individual runs
        st.subheader("🎛️ Individual Agent Testing")
        agent_col1, agent_col2, agent_col3 = st.columns([2, 2, 1])
        
        with agent_col1:
            selected_agent = st.selectbox(
                "Select Agent:",
                options=["rag", "research", "writer", "critic"],
                format_func=lambda x: {
                    "rag": "🔍 RAG Agent",
                    "research": "📚 Research Agent", 
                    "writer": "✍️ Writer Agent",
                    "critic": "🔍 Critic Agent"
                }[x]
            )
        
        with agent_col2:
            agent_context = st.text_input(
                "Additional Context (optional):",
                placeholder="Extra context for the agent..."
            )
        
        with agent_col3:
            st.write("")  # Spacing
            run_single = st.button("▶️ Run Agent")
        
        # Process workflow or single agent
        if run_workflow and topic:
            self.run_complete_workflow(topic)
        elif run_single and topic:
            self.run_single_agent(selected_agent, topic, agent_context)
        
        # Display results
        if st.session_state.workflow_results:
            self.display_workflow_results()
    
    def run_complete_workflow(self, topic: str):
        """Run the complete multi-agent workflow"""
        with st.spinner("🔄 Running complete workflow..."):
            try:
                start_time = time.time()
                results = st.session_state.system.run_workflow(topic)
                end_time = time.time()
                
                st.session_state.workflow_results = {
                    "results": results,
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "duration": end_time - start_time,
                    "type": "complete_workflow"
                }
                
                # Add to history
                st.session_state.workflow_history.append(st.session_state.workflow_results)
                
                st.success(f"✅ Workflow completed in {end_time - start_time:.2f} seconds!")
                
            except Exception as e:
                st.error(f"❌ Error running workflow: {str(e)}")
                st.code(traceback.format_exc())
    
    def run_single_agent(self, agent_name: str, topic: str, context: str = ""):
        """Run a single agent"""
        with st.spinner(f"🔄 Running {agent_name} agent..."):
            try:
                start_time = time.time()
                result = st.session_state.system.run_single_agent(agent_name, topic, context)
                end_time = time.time()
                
                st.session_state.workflow_results = {
                    "results": {agent_name: result},
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "duration": end_time - start_time,
                    "type": "single_agent",
                    "agent_name": agent_name
                }
                
                st.success(f"✅ {agent_name.title()} agent completed in {end_time - start_time:.2f} seconds!")
                
            except Exception as e:
                st.error(f"❌ Error running {agent_name} agent: {str(e)}")
                st.code(traceback.format_exc())
    
    def display_workflow_results(self):
        """Display the workflow results in a structured way"""
        results_data = st.session_state.workflow_results
        
        st.markdown("---")
        st.header("📊 Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Duration", f"{results_data['duration']:.2f}s")
        with col2:
            st.metric("🎯 Topic", results_data['topic'][:20] + "..." if len(results_data['topic']) > 20 else results_data['topic'])
        with col3:
            st.metric("🕐 Time", datetime.fromisoformat(results_data['timestamp']).strftime("%H:%M:%S"))
        with col4:
            result_count = len([k for k, v in results_data['results'].items() if isinstance(v, AgentResponse)])
            st.metric("🤖 Agents", result_count)
        
        # Quick summary for complete workflow
        if results_data['type'] == 'complete_workflow':
            st.markdown("### 📋 Quick Summary")
            
            # Extract key info for summary
            summary_info = self.extract_summary_info(results_data['results'])
            
            if summary_info:
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    if summary_info.get('title'):
                        st.info(f"**Article:** {summary_info['title'][:50]}...")
                
                with summary_col2:
                    if summary_info.get('word_count'):
                        st.info(f"**Length:** {summary_info['word_count']} words")
                
                with summary_col3:
                    if summary_info.get('score'):
                        score = summary_info['score']
                        if score >= 8:
                            st.success(f"**Quality:** {score}/10 (Excellent)")
                        elif score >= 6:
                            st.warning(f"**Quality:** {score}/10 (Good)")
                        else:
                            st.error(f"**Quality:** {score}/10 (Needs Work)")
                
                # Show brief preview of the article
                if summary_info.get('introduction'):
                    with st.expander("👁️ Article Preview"):
                        st.write(f"**{summary_info.get('title', 'Article')}**")
                        intro_preview = summary_info['introduction'][:200] + "..." if len(summary_info['introduction']) > 200 else summary_info['introduction']
                        st.write(intro_preview)
                        st.markdown("*Click on the 'Final Article' tab to read the complete article.*")
        
        st.markdown("---")
        
        # Create tabs for different views
        if results_data['type'] == 'complete_workflow':
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["� Final Article", "�🔍 RAG Results", "📚 Research", "✍️ Writing", "🔍 Review"])
            
            with tab1:
                self.display_final_article(results_data['results'])
            
            with tab2:
                self.display_agent_result("rag", results_data['results'].get("rag"))
            
            with tab3:
                self.display_agent_result("research", results_data['results'].get("research"))
            
            with tab4:
                self.display_agent_result("content", results_data['results'].get("content"))
            
            with tab5:
                self.display_agent_result("review", results_data['results'].get("review"))
        
        else:
            # Single agent result
            agent_name = results_data['agent_name']
            result = results_data['results'][agent_name]
            self.display_agent_result(agent_name, result)
        
        # Workflow visualization
        if results_data['type'] == 'complete_workflow':
            self.display_workflow_timeline(results_data['results'])
    
    def extract_summary_info(self, results: Dict[str, Any]):
        """Extract key information for the summary display"""
        summary = {}
        
        for key, response in results.items():
            if isinstance(response, AgentResponse):
                try:
                    content_data = json.loads(response.content)
                    
                    # Extract from writer agent
                    if key == "content" or response.agent_name == "Writer Agent":
                        if isinstance(content_data, dict):
                            summary['title'] = content_data.get('title', '')
                            summary['introduction'] = content_data.get('introduction', '')
                            summary['word_count'] = content_data.get('word_count', '')
                    
                    # Extract from critic agent
                    elif key == "review" or response.agent_name == "Critic Agent":
                        if isinstance(content_data, dict):
                            summary['score'] = content_data.get('score', 0)
                            summary['assessment'] = content_data.get('overall_assessment', '')
                
                except (json.JSONDecodeError, TypeError):
                    continue
        
        return summary

    def display_final_article(self, results: Dict[str, Any]):
        """Display the complete workflow result as a formatted article"""
        st.markdown("### 📖 Complete Article")
        st.markdown("*Generated by the Multi-Agent System*")
        st.markdown("---")
        
        # Extract content from each agent
        writer_content = None
        research_content = None
        rag_content = None
        critic_content = None
        
        # Parse agent outputs
        for key, response in results.items():
            if isinstance(response, AgentResponse):
                try:
                    content_data = json.loads(response.content)
                    if key == "content" or response.agent_name == "Writer Agent":
                        writer_content = content_data
                    elif key == "research" or response.agent_name == "Research Agent":
                        research_content = content_data
                    elif key == "rag" or response.agent_name == "RAG Agent":
                        rag_content = content_data
                    elif key == "review" or response.agent_name == "Critic Agent":
                        critic_content = content_data
                except (json.JSONDecodeError, TypeError):
                    # Handle non-JSON content
                    if key == "content" or response.agent_name == "Writer Agent":
                        writer_content = {"body": response.content}
        
        # Display the main article
        if writer_content and isinstance(writer_content, dict):
            # Article title
            if "title" in writer_content:
                st.markdown(f"# {writer_content['title']}")
                st.markdown("---")
            
            # Article metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                if "word_count" in writer_content:
                    st.metric("📊 Word Count", writer_content["word_count"])
            with col2:
                st.metric("🕐 Generated", datetime.now().strftime("%B %d, %Y"))
            with col3:
                if critic_content and "score" in critic_content:
                    score = critic_content["score"]
                    st.metric("⭐ Quality Score", f"{score}/10")
            
            st.markdown("---")
            
            # Introduction
            if "introduction" in writer_content:
                st.markdown("## Introduction")
                st.markdown(writer_content["introduction"])
                st.markdown("")
            
            # Main content body
            if "body" in writer_content:
                st.markdown("## Main Content")
                # Split the body into paragraphs for better formatting
                body_text = writer_content["body"]
                paragraphs = body_text.split('\n\n') if '\n\n' in body_text else [body_text]
                
                for paragraph in paragraphs:
                    if paragraph.strip():
                        st.markdown(paragraph.strip())
                        st.markdown("")
            
            # Conclusion
            if "conclusion" in writer_content:
                st.markdown("## Conclusion")
                st.markdown(writer_content["conclusion"])
                st.markdown("")
            
            # Additional insights from research
            if research_content and isinstance(research_content, dict):
                st.markdown("---")
                st.markdown("## Key Insights")
                
                if "key_concepts" in research_content and research_content["key_concepts"]:
                    st.markdown("**Key Concepts:**")
                    for concept in research_content["key_concepts"][:5]:  # Show top 5
                        st.markdown(f"• {concept}")
                    st.markdown("")
                
                if "challenges" in research_content and research_content["challenges"]:
                    st.markdown("**Important Considerations:**")
                    for challenge in research_content["challenges"][:3]:  # Show top 3
                        st.markdown(f"• {challenge}")
                    st.markdown("")
            
            # Sources and references (from RAG)
            if rag_content and isinstance(rag_content, dict):
                if "sources_used" in rag_content and rag_content["sources_used"]:
                    st.markdown("---")
                    st.markdown("## Sources")
                    for i, source in enumerate(rag_content["sources_used"], 1):
                        st.markdown(f"{i}. {source}")
            
            # Quality assessment summary
            if critic_content and isinstance(critic_content, dict):
                st.markdown("---")
                st.markdown("## Quality Assessment")
                
                if "overall_assessment" in critic_content:
                    st.info(f"**Editor's Note:** {critic_content['overall_assessment']}")
                
                # Show score with visual indicator
                if "score" in critic_content:
                    score = critic_content["score"]
                    if score >= 8:
                        st.success(f"✅ **High Quality Article** - Score: {score}/10")
                    elif score >= 6:
                        st.warning(f"⚠️ **Good Quality Article** - Score: {score}/10")
                    else:
                        st.error(f"❌ **Needs Improvement** - Score: {score}/10")
                
                # Show strengths if available
                if "strengths" in critic_content and critic_content["strengths"]:
                    with st.expander("💪 Article Strengths"):
                        for strength in critic_content["strengths"]:
                            st.write(f"• {strength}")
        
        else:
            st.warning("❌ No article content available. The writer agent may not have produced valid output.")
            
        # Export options
        st.markdown("---")
        st.markdown("### 📥 Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Copy to Clipboard"):
                article_text = self.generate_article_text(writer_content, research_content, rag_content, critic_content)
                st.text_area("Article Text (Copy this):", article_text, height=200)
        
        with col2:
            if st.button("💾 Download as Text"):
                article_text = self.generate_article_text(writer_content, research_content, rag_content, critic_content)
                st.download_button(
                    label="📄 Download Article",
                    data=article_text,
                    file_name=f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    def generate_article_text(self, writer_content, research_content, rag_content, critic_content):
        """Generate plain text version of the article"""
        article_lines = []
        
        if writer_content and isinstance(writer_content, dict):
            # Title
            if "title" in writer_content:
                article_lines.append(writer_content["title"])
                article_lines.append("=" * len(writer_content["title"]))
                article_lines.append("")
            
            # Metadata
            article_lines.append(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
            if "word_count" in writer_content:
                article_lines.append(f"Word Count: {writer_content['word_count']}")
            if critic_content and "score" in critic_content:
                article_lines.append(f"Quality Score: {critic_content['score']}/10")
            article_lines.append("")
            article_lines.append("-" * 50)
            article_lines.append("")
            
            # Introduction
            if "introduction" in writer_content:
                article_lines.append("INTRODUCTION")
                article_lines.append("")
                article_lines.append(writer_content["introduction"])
                article_lines.append("")
            
            # Main content
            if "body" in writer_content:
                article_lines.append("MAIN CONTENT")
                article_lines.append("")
                article_lines.append(writer_content["body"])
                article_lines.append("")
            
            # Conclusion
            if "conclusion" in writer_content:
                article_lines.append("CONCLUSION")
                article_lines.append("")
                article_lines.append(writer_content["conclusion"])
                article_lines.append("")
            
            # Key insights
            if research_content and isinstance(research_content, dict):
                if "key_concepts" in research_content and research_content["key_concepts"]:
                    article_lines.append("KEY CONCEPTS")
                    article_lines.append("")
                    for concept in research_content["key_concepts"][:5]:
                        article_lines.append(f"• {concept}")
                    article_lines.append("")
            
            # Sources
            if rag_content and isinstance(rag_content, dict):
                if "sources_used" in rag_content and rag_content["sources_used"]:
                    article_lines.append("SOURCES")
                    article_lines.append("")
                    for i, source in enumerate(rag_content["sources_used"], 1):
                        article_lines.append(f"{i}. {source}")
                    article_lines.append("")
        
        return "\n".join(article_lines)

    def display_agent_result(self, agent_key: str, response: Optional[AgentResponse]):
        """Display individual agent result"""
        if not response:
            st.warning(f"No result available for {agent_key}")
            return
        
        # Agent header
        agent_names = {
            "rag": "🔍 RAG Agent",
            "research": "📚 Research Agent",
            "content": "✍️ Writer Agent", 
            "review": "🔍 Critic Agent"
        }
        
        agent_display_name = agent_names.get(agent_key, f"🤖 {agent_key.title()} Agent")
        st.subheader(agent_display_name)
        
        # Metadata
        with st.expander("📋 Agent Metadata"):
            col1, col2 = st.columns(2)
            with col1:
                st.json(response.metadata)
            with col2:
                st.write(f"**Timestamp:** {datetime.fromtimestamp(response.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Agent:** {response.agent_name}")
        
        # Try to parse and display JSON content
        try:
            content_data = json.loads(response.content)
            
            if agent_key == "rag":
                self.display_rag_result(content_data)
            elif agent_key == "research":
                self.display_research_result(content_data)
            elif agent_key == "content":
                self.display_writer_result(content_data)
            elif agent_key == "review":
                self.display_critic_result(content_data)
            else:
                st.json(content_data)
                
        except (json.JSONDecodeError, TypeError):
            # If not JSON, display as text
            st.text_area("Response Content:", response.content, height=300, disabled=True)
    
    def display_rag_result(self, data: Dict[str, Any]):
        """Display RAG agent specific results"""
        if isinstance(data, dict):
            if "answer" in data:
                st.markdown("### 💡 Answer")
                st.write(data["answer"])
                
                if "sources_used" in data:
                    st.markdown("### 📚 Sources Used")
                    for source in data["sources_used"]:
                        st.write(f"• {source}")
                
                if "confidence" in data:
                    confidence = data["confidence"]
                    color = {"high": "green", "medium": "orange", "low": "red"}.get(confidence, "gray")
                    st.markdown(f"### 📊 Confidence: <span style='color:{color}'>{confidence.title()}</span>", unsafe_allow_html=True)
                
                if "additional_info_needed" in data:
                    st.markdown("### ❓ Additional Information Needed")
                    st.info(data["additional_info_needed"])
            else:
                st.json(data)
        else:
            st.write(data)
    
    def display_research_result(self, data: Dict[str, Any]):
        """Display research agent specific results"""
        if isinstance(data, dict):
            if "summary" in data:
                st.markdown("### 📝 Summary")
                st.write(data["summary"])
            
            if "key_concepts" in data:
                st.markdown("### 🔑 Key Concepts")
                for concept in data["key_concepts"]:
                    st.write(f"• {concept}")
            
            if "main_points" in data:
                st.markdown("### 📋 Main Points")
                for point in data["main_points"]:
                    st.write(f"• {point}")
            
            if "perspectives" in data:
                st.markdown("### 👀 Different Perspectives")
                for perspective in data["perspectives"]:
                    st.write(f"• {perspective}")
            
            if "challenges" in data:
                st.markdown("### ⚠️ Challenges")
                for challenge in data["challenges"]:
                    st.write(f"• {challenge}")
        else:
            st.json(data)
    
    def display_writer_result(self, data: Dict[str, Any]):
        """Display writer agent specific results"""
        if isinstance(data, dict):
            if "title" in data:
                st.markdown(f"# {data['title']}")
            
            if "introduction" in data:
                st.markdown("## Introduction")
                st.write(data["introduction"])
            
            if "body" in data:
                st.markdown("## Content")
                st.write(data["body"])
            
            if "conclusion" in data:
                st.markdown("## Conclusion")
                st.write(data["conclusion"])
            
            if "word_count" in data:
                st.metric("📊 Word Count", data["word_count"])
        else:
            st.json(data)
    
    def display_critic_result(self, data: Dict[str, Any]):
        """Display critic agent specific results"""
        if isinstance(data, dict):
            if "overall_assessment" in data:
                st.markdown("### 📊 Overall Assessment")
                st.write(data["overall_assessment"])
            
            if "score" in data:
                score = data["score"]
                color = "green" if score >= 8 else "orange" if score >= 6 else "red"
                st.markdown(f"### 🏆 Score: <span style='color:{color};font-size:24px'>{score}/10</span>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "strengths" in data:
                    st.markdown("### ✅ Strengths")
                    for strength in data["strengths"]:
                        st.write(f"• {strength}")
            
            with col2:
                if "improvements" in data:
                    st.markdown("### 🔧 Areas for Improvement")
                    for improvement in data["improvements"]:
                        st.write(f"• {improvement}")
            
            if "suggestions" in data:
                st.markdown("### 💡 Suggestions")
                for suggestion in data["suggestions"]:
                    st.write(f"• {suggestion}")
            
            if "final_verdict" in data:
                verdict = data["final_verdict"]
                verdict_color = {"approve": "green", "revise": "orange", "reject": "red"}.get(verdict, "gray")
                st.markdown(f"### 🎯 Final Verdict: <span style='color:{verdict_color}'>{verdict.title()}</span>", unsafe_allow_html=True)
        else:
            st.json(data)
    
    def display_workflow_timeline(self, results: Dict[str, Any]):
        """Display a timeline visualization of the workflow"""
        st.markdown("---")
        st.subheader("📈 Workflow Timeline")
        
        # Create timeline data
        agents = []
        timestamps = []
        
        for key, response in results.items():
            if isinstance(response, AgentResponse):
                agents.append(response.agent_name)
                timestamps.append(response.timestamp)
        
        if agents and timestamps:
            # Create a simple timeline chart
            fig = go.Figure()
            
            for i, (agent, timestamp) in enumerate(zip(agents, timestamps)):
                fig.add_trace(go.Scatter(
                    x=[datetime.fromtimestamp(timestamp)],
                    y=[i],
                    mode='markers+text',
                    text=[agent],
                    textposition="middle right",
                    marker=dict(size=15, color=f'rgb({50 + i*50}, {100 + i*30}, {200 - i*20})'),
                    name=agent
                ))
            
            fig.update_layout(
                title="Agent Execution Timeline",
                xaxis_title="Time",
                yaxis_title="Agents",
                yaxis=dict(tickvals=list(range(len(agents))), ticktext=agents),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_history_tab(self):
        """Render workflow history"""
        st.header("📚 Workflow History")
        
        if not st.session_state.workflow_history:
            st.info("No workflow history available yet.")
            return
        
        # Display history in reverse chronological order
        for i, workflow in enumerate(reversed(st.session_state.workflow_history[-10:])):  # Show last 10
            with st.expander(f"🕐 {workflow['topic'][:50]}... ({datetime.fromisoformat(workflow['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{workflow['duration']:.2f}s")
                with col2:
                    st.metric("Type", workflow['type'].replace('_', ' ').title())
                with col3:
                    if st.button(f"🔄 Load Results", key=f"load_{i}"):
                        st.session_state.workflow_results = workflow
                        st.rerun()
    
    def run(self):
        """Main run method"""
        self.setup_page()
        
        # Main navigation
        main_tab, history_tab, about_tab = st.tabs(["🏠 Main", "📚 History", "ℹ️ About"])
        
        with main_tab:
            self.render_sidebar()
            self.render_main_interface()
        
        with history_tab:
            self.render_history_tab()
        
        with about_tab:
            st.markdown("""
            # 🤖 Multi-Agent System with RAG
            
            This application demonstrates a sophisticated multi-agent system that uses:
            
            ## 🔧 System Components
            - **RAG Agent**: Retrieves relevant information from documents
            - **Research Agent**: Conducts comprehensive research
            - **Writer Agent**: Creates well-structured content
            - **Critic Agent**: Reviews and provides feedback
            - **Coordinator Agent**: Orchestrates the entire workflow
            
            ## 🚀 Features
            - **Interactive UI**: Easy-to-use Streamlit interface
            - **Real-time Results**: See each agent's output as it processes
            - **Document Management**: Add your own documents to the RAG system
            - **Workflow History**: Track and review previous runs
            - **Individual Agent Testing**: Run agents independently
            
            ## 🛠️ Technology Stack
            - **LangChain**: For agent orchestration
            - **Ollama**: Local LLM inference
            - **FAISS**: Vector storage for RAG
            - **Streamlit**: Web interface
            - **Plotly**: Data visualization
            
            ## 📝 Usage Tips
            1. Initialize the system first using the sidebar
            2. Add relevant documents to enhance RAG performance
            3. Enter clear, specific topics for better results
            4. Use individual agents to test specific functionality
            5. Check the workflow history to compare results
            """)


def main():
    """Main function to run the Streamlit app"""
    ui = MultiAgentUI()
    ui.run()


if __name__ == "__main__":
    main()
