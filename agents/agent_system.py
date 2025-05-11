from langgraph.graph import Graph
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentSystem:
    def __init__(self):
        # Initialize LLM with local Ollama configuration
        self.llm = Ollama(
            base_url="http://localhost:11434",
            model="mistral:7b-instruct"
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="applications",
            embedding_function=self.embeddings
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agents
        self.agents = self._create_agents()
        
        # Create graph
        self.graph = self._create_graph()
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for agents."""
        return [
            Tool(
                name="extract_data",
                func=self._extract_data,
                description="Extract relevant data from documents"
            ),
            Tool(
                name="validate_data",
                func=self._validate_data,
                description="Validate extracted data"
            ),
            Tool(
                name="check_eligibility",
                func=self._check_eligibility,
                description="Check eligibility based on data"
            ),
            Tool(
                name="make_recommendation",
                func=self._make_recommendation,
                description="Make final recommendation"
            )
        ]
    
    def _create_agents(self) -> Dict[str, AgentExecutor]:
        """Create specialized agents."""
        # Base prompt template
        base_prompt = PromptTemplate.from_template(
            """You are an AI assistant helping with social support applications.
            Current task: {task}
            Available tools: {tools}
            
            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Question: {input}
            {agent_scratchpad}"""
        )
        
        # Create agents
        agents = {}
        for agent_type in ["extraction", "validation", "eligibility", "recommendation"]:
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=base_prompt
            )
            agents[agent_type] = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=True
            )
        
        return agents
    
    def _create_graph(self) -> Graph:
        """Create the agent workflow graph."""
        def extraction_agent(state):
            return self.agents["extraction"].invoke(state)
        
        def validation_agent(state):
            return self.agents["validation"].invoke(state)
        
        def eligibility_agent(state):
            return self.agents["eligibility"].invoke(state)
        
        def recommendation_agent(state):
            return self.agents["recommendation"].invoke(state)
        
        # Create graph
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("extraction", extraction_agent)
        workflow.add_node("validation", validation_agent)
        workflow.add_node("eligibility", eligibility_agent)
        workflow.add_node("recommendation", recommendation_agent)
        
        # Add edges
        workflow.add_edge("extraction", "validation")
        workflow.add_edge("validation", "eligibility")
        workflow.add_edge("eligibility", "recommendation")
        
        # Set entry point
        workflow.set_entry_point("extraction")
        
        # Compile the graph
        return workflow.compile()
    
    def _extract_data(self, query: str) -> str:
        """Extract data from documents."""
        try:
            # Split text into chunks
            texts = self.text_splitter.split_text(query)
            
            # Add to vector store
            self.vector_store.add_texts(texts)
            
            # Search for relevant information
            docs = self.vector_store.similarity_search(query)
            
            # Extract and format data
            extracted_data = {
                "text": "\n".join(doc.page_content for doc in docs),
                "metadata": [doc.metadata for doc in docs]
            }
            
            return json.dumps(extracted_data)
        except Exception as e:
            logger.error(f"Error in data extraction: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _validate_data(self, data: str) -> str:
        """Validate extracted data."""
        try:
            # Parse input data
            data_dict = json.loads(data)
            
            # Mock validation
            validation_result = {
                "is_valid": True,
                "missing_fields": [],
                "inconsistencies": []
            }
            
            # Check for required fields
            required_fields = ["name", "income", "expenses", "dependents"]
            for field in required_fields:
                if field not in data_dict:
                    validation_result["missing_fields"].append(field)
                    validation_result["is_valid"] = False
            
            return json.dumps(validation_result)
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _check_eligibility(self, data: str) -> str:
        """Check eligibility based on data."""
        try:
            # Parse input data
            data_dict = json.loads(data)
            
            # Mock eligibility check
            eligibility_result = {
                "is_eligible": True,
                "score": 85,
                "factors": [
                    {"name": "Income", "score": 80, "impact": "Positive"},
                    {"name": "Expenses", "score": 90, "impact": "Positive"},
                    {"name": "Dependents", "score": 85, "impact": "Positive"}
                ]
            }
            
            return json.dumps(eligibility_result)
        except Exception as e:
            logger.error(f"Error in eligibility check: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _make_recommendation(self, data: str) -> str:
        """Make final recommendation."""
        try:
            # Parse input data
            data_dict = json.loads(data)
            
            # Mock recommendation
            recommendation = {
                "recommendation": "Approve",
                "support_amount": 1500,
                "duration_months": 6,
                "conditions": [
                    "Monthly income verification",
                    "Quarterly review"
                ]
            }
            
            return json.dumps(recommendation)
        except Exception as e:
            logger.error(f"Error in recommendation: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def process_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete application."""
        try:
            # Convert application data to string
            input_data = json.dumps(application_data)
            
            # Run the workflow
            result = self.graph.invoke({"input": input_data})
            
            return result
        except Exception as e:
            logger.error(f"Error processing application: {str(e)}")
            return {"error": str(e)}

# Create singleton instance
agent_system = AgentSystem() 