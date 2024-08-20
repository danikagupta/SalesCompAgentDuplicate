# src/clarify_agent.py

from langchain_core.messages import SystemMessage, HumanMessage  # Import necessary message classes

class ClarifyAgent:
    
    def __init__(self, model, sales_comp_agent):
        """
        Initialize the ClarifyAgent with a ChatOpenAI model and a reference to the main salesCompAgent.
        
        :param model: An instance of the ChatOpenAI model used for generating responses.
        :param sales_comp_agent: A reference to the main salesCompAgent for reclassification and ticket creation.
        """
        self.model = model
        self.sales_comp_agent = sales_comp_agent

    def clarify_and_classify(self, user_query: str) -> dict:
        """
        Ask the user to clarify their query and attempt to classify it again.
        
        :param user_query: The original query from the user.
        :return: A dictionary representing the next state based on the user's input.
        """
        # Ask the user to clarify their query
        clarification_prompt = f"""
        I'm not sure I fully understood your request. Could you please clarify what you need help with?
        """

        # Generate a clarification response using the ChatOpenAI model
        llm_response = self.model.invoke([
            SystemMessage(content=clarification_prompt),
            HumanMessage(content=user_query)
        ])

        # Extract the clarified response from the user
        clarified_query = llm_response.content

        # Attempt to classify the clarified query using the main salesCompAgent's classifier
        classified_state = self.sales_comp_agent.initial_classifier({"initialMessage": clarified_query})

        # If classification is successful, return the new state
        if classified_state["category"] != "clarify":
            return classified_state
        
        # If classification still fails, ask the user if they'd like to create a ticket
        ticket_prompt = f"""
        I'm still having trouble understanding your request. Would you like to create a support ticket instead?
        Please respond with 'yes' or 'no'.
        """

        # Generate the ticket response using the ChatOpenAI model
        llm_ticket_response = self.model.invoke([
            SystemMessage(content=ticket_prompt),
            HumanMessage(content=clarified_query)
        ])

        # If the user agrees, route to the ticket agent
        if "yes" in llm_ticket_response.content.lower():
            return self.sales_comp_agent.ticket_agent({"initialMessage": clarified_query})
        
        # If the user does not agree, end the conversation
        return {
            "lnode": "clarify_agent", 
            "responseToUser": "Okay, feel free to reach out if you need further assistance.",
            "category": "clarify"
        }

    def clarify_agent(self, state: dict) -> dict:
        """
        Handle queries that require clarification and attempt to classify them again.
        
        :param state: A dictionary containing the state of the current conversation, including the user's initial message.
        :return: A dictionary with the updated state based on the user's clarified response.
        """
        # Handle clarification and reclassification
        return self.clarify_and_classify(state['initialMessage'])
