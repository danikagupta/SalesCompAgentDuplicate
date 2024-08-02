# Flow of Execution

1) User Input and Initial Classifier:
The user input is passed to the graph.stream method.
The initial_classifier method is invoked since the entry point is set to "classifier".

2) Returning from Initial Classifier:
The initial_classifier method classifies the input and returns a state with the category.
The returned state includes "category": category, where category could be "policy", "commission", etc.

3) Routing in Main Router:
The main_router method receives the state and returns the category, which corresponds to the next node in the state graph.

4) StateGraph Handles Transitions:
The StateGraph framework automatically transitions to the node corresponding to the returned category.
It then calls the method associated with that node.

Why do we need both initial_classifier and main_router?

initial_classifier: Focuses solely on classifying the user input. It doesn't decide what to do next.

main_router: Takes the classification result and determines the next state in the graph. T

This decouples the classification logic from the routing logic.