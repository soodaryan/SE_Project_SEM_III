from pathlib import Path
from utils import get_doc_tools, initialize_settings

from llama_index.core.agent import AgentRunner
from llama_index.core.objects import ObjectIndex
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.agent import FunctionCallingAgentWorker


from langchain.llms.base import BaseLLM

class CustomFunctionCallingLLM(BaseLLM):
    def __init__(self):
        # Assuming initialize_settings() sets up necessary configurations
        initialize_settings()  
        self.model = Settings.llm  # Initialize your model from Settings

    def _generate(self, prompt, **kwargs):
        # Use the model's complete method to generate the response
        return self.model.complete(prompt)

    def _llm_type(self):
        # Return the LLM type, you can customize this as needed
        return "custom"  # You can also return a more specific value based on your LLM

    def predict(self, prompt, **kwargs):
        return self._generate(prompt, **kwargs)

    def call(self, prompt: str) -> str:
        return self.predict(prompt)

    def predict_messages(self, messages, **kwargs):
        # Combine messages into one prompt and call predict
        prompt = "\n".join(msg['content'] for msg in messages)
        return self.predict(prompt, **kwargs)



def ChatBot() :
    """
    Creates Agent instance with specified context. Used to initalize the chat agent for infernece
    :return: runable agent instance
    """
    files = [
        "data/document.pdf"
    ]
    initialize_settings()

    files_to_tools_dict = {}
    for file in files:
        print(f"Getting tools for file: {file}")
        vector_tool = get_doc_tools(file, Path(file).stem)
        files_to_tools_dict[file] = [vector_tool]
    print(files_to_tools_dict)

    llm = Settings.llm
    all_tools = [t for file in files for t in files_to_tools_dict[file]]

    # defining an "object" index and retriever over these tools
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )
    llm = CustomFunctionCallingLLM()


    obj_retriever = obj_index.as_retriever(similarity_top_k=2)
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=obj_retriever,
        llm=llm, 
        system_prompt=
        """
        You are an intelligent assistant specializing in retrieving and answering queries based on provided documents. 
        Use the retrieved context to generate accurate and concise answers. 
        If the information is not in the document, explicitly state, "The document does not provide this information." 
        Ensure all answers are derived only from the retrieved content and are highly relevant to the query.
        Example Queries:

        **Query** : What is the e-Tender notice number and its purpose?
        **Answer** : The e-Tender notice number is NITJ/DRC/PUR/TT/36/2024. Its purpose is the fabrication of a machine for continuous production of textile waste-based composite materials for the Department of Textile Technology.
        
        **Query** : The technical bid is evaluated to ensure compliance with essential eligibility criteria, submission of EMD and Tender Fee, completion of required documents, adherence to equipment specifications, and validity of service and warranty policies. Only technically qualified bids proceed to the financial evaluation stage.
        """,
        verbose=False
    )
    agent = AgentRunner(agent_worker)
    return agent


def chat(query) : 
    """
    Generates responses to input queries using chatbot 
    :param query: input query
    :return: reponse
    """
    agent = ChatBot()
    resp = agent.query(query)
    print("done")
    return resp.response


def test(agent,query):
    """
    Generates responses to input queries using chatbot of choice. Used so that the chatbot is only defined once and not repeatedly on every call 
    :param query: runable agent instance
    :param query: input query
    :return: reponse
    """
    resp = agent.query(query)
    return resp.response


def main() : 
    queries = [
        "What is the e-Tender notice number and the purpose of the tender mentioned in the document?",
        "What are the eligibility criteria for bidders to participate in this tender?",
        "What are the deadlines for submitting the online bids and physically submitting the tender fee and EMD?",
        "What is the role of Annexure-G in determining supplier eligibility, and how is local content defined?",
        "What is the payment structure for the successful supplier as mentioned in the document?",
        "What are the warranty obligations for suppliers as outlined in Annexure-F?",
        "How is the technical bid evaluated, and what criteria are used for shortlisting bidders?",
        "What penalties are imposed for delays in delivery or non-performance by the supplier?",
        "What does Annexure-E specify about the blacklisting or debarment of suppliers?",
        "What does Annexure-D require from suppliers regarding manufacturer authorization?"
    ]
    responses = {}
    agent = ChatBot()
    for query in queries : 
        response = test(agent, query)
        responses[query] = response
        print(f"Q) {query}")
        print(f"A) {response}")


if __name__ == "__main__": 
    main()