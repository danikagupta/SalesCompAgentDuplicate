import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from pinecone import Pinecone
from src.policy_agent import PolicyAgent
from src.commission_agent import CommissionAgent
from src.contest_agent import ContestAgent
from src.ticket_agent import TicketAgent 
from src.clarify_agent import ClarifyAgent
from src.small_talk_agent import SmallTalkAgent
from src.plan_explainer_agent import PlanExplainerAgent
from src.feedback_collector_agent import FeedbackCollectorAgent
from src.create_llm_message import create_llm_message

from langgraph.graph.message import AnyMessage, add_messages


# Define the structure of the agent state using TypedDict
class AgentState(TypedDict):
    agent: str
    initialMessage: str
    responseToUser: str
    lnode: str
    category: str
    sessionHistory: Annotated[list[AnyMessage], add_messages]

# Define the structure of outputs from different agents using Pydantic (BaseModel) for runtime data validation 
# and serialization
class Category(BaseModel):
    category: str

#class PolicyResponse(BaseModel):
#    policy: str
#    response: str

#class CommissionResponse(BaseModel):
#    commission: str
#    calculation: str
#    response: str

#class ContestResponse(BaseModel):
#    contestUrl: str
#    contestRules: str
#    response: str

#class PlanExplainerResponse(BaseModel):
#    plan: str
#    response: str



def get_contest_info():
        with open('contestrules.txt', 'r') as file:
            contestrules = file.read()
        return contestrules

# Define valid categories
VALID_CATEGORIES = ["policy", "commission", "contest", "ticket", "smalltalk", "clarify", "planexplainer", "feedbackcollector"]

# Define the salesCompAgent class
class salesCompAgent():
    def __init__(self, api_key):
        # Initialize the ChatOpenAI model (from LangChain) and OpenAI client with the given API key
        # ChatOpenAI is used for chat interactions
        # OpenAI is used for creating embeddings
        self.model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
        self.client = OpenAI(api_key=api_key)

        #Pinecone configurtion using Streamlit secrets
        # Pinecone is used for storing and querying embeddings
        self.pinecone_api_key = st.secrets['PINECONE_API_KEY']
        self.pinecone_env = st.secrets['PINECONE_API_ENV']
        self.pinecone_index_name = st.secrets['PINECONE_INDEX_NAME']

        # Initialize Pinecone once
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pinecone.Index(self.pinecone_index_name)

        # Initialize the PolicyAgent, CommissionAgent, ContestAgent, TicketAgent, ClarifyAgent, SmallTalkAgent,
        # PlanExplainerAgent, FeedbackCollectorAgent
        self.policy_agent_class = PolicyAgent(self.client, self.model, self.index)
        self.commission_agent_class = CommissionAgent(self.model)
        self.contest_agent_class = ContestAgent(self.model)
        self.ticket_agent_class = TicketAgent(self.model)
        self.clarify_agent_class = ClarifyAgent(self.model) # Capable of passing reference to the main agent
        self.small_talk_agent_class = SmallTalkAgent(self.client, self.model)
        self.plan_explainer_agent_class = PlanExplainerAgent(self.client, self.model, self.index)
        self.feedback_collector_agent_class = FeedbackCollectorAgent(self.model)

        # Build the state graph
        workflow = StateGraph(AgentState)
        workflow.add_node("classifier", self.initial_classifier)
        workflow.add_node("policy", self.policy_agent_class.policy_agent)
        workflow.add_node("commission", self.commission_agent_class.commission_agent)
        workflow.add_node("contest", self.contest_agent_class.contest_agent)
        workflow.add_node("ticket", self.ticket_agent_class.ticket_agent)
        workflow.add_node("clarify", self.clarify_agent_class.clarify_agent)
        workflow.add_node("smalltalk", self.small_talk_agent_class.small_talk_agent)
        workflow.add_node("planexplainer", self.plan_explainer_agent_class.plan_explainer_agent)
        workflow.add_node("feedbackcollector", self.feedback_collector_agent_class.feedback_collector_agent)

        # Set the entry point and add conditional edges
        workflow.add_conditional_edges("classifier", self.main_router)

        # Define end points for each node
        workflow.add_edge(START, "classifier")
        workflow.add_edge("policy", END)
        workflow.add_edge("commission", END)
        workflow.add_edge("contest", END)
        workflow.add_edge("ticket", END)
        workflow.add_edge("clarify", END)
        workflow.add_edge("smalltalk", END)
        workflow.add_edge("planexplainer", END)
        workflow.add_edge("feedbackcollector", END)

        # Set up in-memory SQLite database for state saving
        #memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        #self.graph = builder.compile(checkpointer=memory)

        self.graph = workflow.compile()

    # Initial classifier function to categorize user messages
    def initial_classifier(self, state: AgentState):
        print("initial classifier")
        #print(f"state is {state}\n******\n")
        if 'messages' in state:
            print(f"messages is {state['messages']}")
        
        CLASSIFIER_PROMPT = f"""
You are an expert in sales operations with deep knowledge of sales compensation. Your job is to accurately classify customer requests into one of the following categories based on context and content, even if specific keywords are not used.
	"""

        CLASSIFIER_PROMPT = f"""
You are an expert with deep knowledge of sales compensation. Your job is to comprehend the message from the user 
even if it lacks specific keywords, always maintain a friendly, professional, and helpful tone. If a user greets 
you, greet them back by mirroring user's tone and verbosity, and offer assitance. 

Based on user query, accurately classify customer requests into one of the following categories based on context 
and content, even if specific keywords are not used.

1) **policy**: Select this category if the request is related to any formal sales compensation rules or guidelines, 
even if the word "policy" is not mentioned. This includes topics like windfall, minimum commission guarantees, bonus structures, or leave-related questions.
   - Example: "What happens to my commission if I go on leave?" (This is about policy.)
   - Example: "Is there any guarantee for minimum commission guarantee or MCG?" (This is about policy.)
   - Example: "Can you tell me what is a windfall?" (This is about policy.)
   - Example: "What is a teaming agreement?" (This is about policy.)
   - Example: "What is a split or commission split?" (This is about policy.)

2) **commission**: Select this category if the request involves the calculation or details of the user's sales 
commission, such as earnings, rates, or specific deal-related inquiries.
   - Example: "How much commission will I earn on a $500,000 deal?" (This is about commission.)
   - Example: "What is the new commission rate?" (This is about commission.)

3) **contest**: Select this category if the request is about sales contests or SPIF
   - Example: "How do I start a sales contest?" (This is about contests.)
   - Example: "I'd like to initiate a SPIF" (This is about contests.)
   - Example: "What are the rules for the upcoming contest?" (This is about contests.)

4) **ticket**: Select this category if the request involves issues or problems that you either don't know how to answer 
or require a human to be involved, such as system issues, payment errors, or situations where a service ticket is required.
   - Example: "I can't access my commission report." (This is about a ticket.)
   - Example: "My commission was calculated incorrectly." (This is about a ticket.)
   - Example: "Please explain how my commission was computed." (This is about a ticket.)

5) **smalltalk**: Select this category if the user query is a greeting or a generic comment.
    - Example: "Hi there"
    - Example: "How are you doing?"
    - Example: "Good morning"

6) **planexplainer**: Select this category if the request is a question about sales comp plan to understand how
comp plan works, even if the word "plan" is not mentioned. 
   - Example: "What is BCR (or Base Commission Rate)?" (This is about plainexplainer.)
   - Example: "What are kickers (or add-on incentives)?" (This is about planexplainer.)
   - Example: "Can you tell me if kicker retire quota or not?" (This is about planexplainer.)
   - Example: "What is a transition point?" (This is about planexplainer.)
   - Example: "What is the difference between ACR1 and ACR2?" (This is about planexplainer.)
   - Example: "What are accelerators?" (This is about planexplainer)
   - Example: "Why do I have a larger quota but lower commission rate?" (This is about planexplainer)

7) **feedbackcollector**: Select this category if the request is about providing feedback on either sales comp plan, 
policy, SPIF, or sales contest. This is NOT an issue whcih the user is trying to get resolved immediately. This is 
something which the user is not happy about and would like someone to listen, understand, and log as feedback.  
   - Example: "Policy does not make sense." (This is about feedbackcollector.)
   - Example: "This is driving the wrong behavior" (This is about feedbackcollector.)
   - Example: "My plan is not motivating enough" (This is about feedbackcollector.)
   - Example: "It's causing friction between multiple sales reps or sales teams." (This is about feedbackcollector.)
   - Example: "Our sales incentives are not as lucrative as our competitors." (This is about feedbackcollector.)
   - Example: "I used to make a lot more money at my previous employer." (This is about feedbackcollector)

8) **clarify**: Select this category if the request is unclear, ambiguous, or does not fit into the above categories. 
Ask the user for more details.
    - Example: "I'm not happy with my compensation plan"

Remember to consider the context and content of the request, even if specific keywords like 'policy' or 'commission' 
are not used.
""" 
  
        llm_messages = create_llm_message(CLASSIFIER_PROMPT)

        llm_response = self.model.with_structured_output(Category).invoke(llm_messages)

        # Extract the category from the model's response
        category = llm_response.category
        print(f"category is {category}")
        
        # Return the updated state with the category
        return{
            "lnode": "initial_classifier", 
            #"responseToUser": "Classifier successful",
            "category": category,
            "sessionHistory": f"SessionHistory: category is {category}"
        }
    
     # Main router function to direct to the appropriate agent based on the category
    def main_router(self, state: AgentState):
        my_category = state['category']
        if my_category in VALID_CATEGORIES:
            return my_category
        else:
            print(f"unknown category: {my_category}")
            return END
