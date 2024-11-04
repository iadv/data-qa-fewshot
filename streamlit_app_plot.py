import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.agents import Tool, ZeroShotAgent
from langchain.agents import AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
import os
import matplotlib.pyplot as plt
import pandas as pd

# Set up environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize the database
db = SQLDatabase.from_uri("sqlite:///Data.db")

# Increase the default width of the main area by 50%
st.set_page_config(layout="wide")

# Upload Wastage and Maintenance files.

with st.sidebar:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.subheader('**How to Use:**')
    st.write('''

    1. 📄 Upload the relevant files in CSV format (e.g., maintenance data, customer data, sales data, etc.).
    2. Wait for the files to upload.
    3. Ask Questions! 📊

    ''')
    st.markdown("")
    st.markdown("")
    st.markdown("")


st.title('🤖 Data to Insights')
st.markdown("#### Unlock Actionable Insights from Your Machine Data")
st.markdown("")
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    uploaded_file_w = st.file_uploader("**Upload Wastage File (.csv)**", type=("csv"))

with col2:
    uploaded_file_m = st.file_uploader("**Upload Maintenance File (.csv)**", type=("csv"))

if uploaded_file_w and uploaded_file_m:
    # Read the uploaded files
    dfe = pd.read_csv(uploaded_file_w)
    dfr = pd.read_csv(uploaded_file_m)
    
    # Connect to the SQLite database
    conn = sqlite3.connect('Data.db')
    
    # Save dataframes to SQL tables
    dfe.to_sql('Wastage_Data', conn, index=False, if_exists='replace')
    dfr.to_sql('Maintenance_Data', conn, index=False, if_exists='replace')
    
    # Commit and close the connection
    conn.commit()
    conn.close()
    
    st.success("Files have been successfully uploaded.")


st.markdown("")
st.markdown("")
st.markdown("")


# Step 1: Define example queries
examples = [
    {"input": "What is the amount of wastage for tea blends?", "query": """SELECT "Copy of Comp MatlGrp Desc" AS ComponentMaterialGroup, SUM("Var2Prf Amt") AS TotalWastage FROM Wastage_Data WHERE "Copy of Comp MatlGrp Desc" = 'Tea Blends' GROUP BY "Copy of Comp MatlGrp Desc";"""},
    {"input": "What's the reasons for plastic bags wastage on L1?", "query": "SELECT Level2Reason, COUNT(*) AS ReasonCount \nFROM Maintenance_Data \nWHERE Line = 'L01 - C24' \nGROUP BY Level2Reason;"},
    {"input": "What was the top contributor to wastage this month?", "query": """SELECT "Copy of Comp MatlGrp Desc" AS ComponentMaterialGroup, SUM("Var2Prf Amt") AS TotalWastage \nFROM Wastage_Data \nGROUP BY "Copy of Comp MatlGrp Desc";"""},
]

# Step 2: Create a FewShotPromptTemplate
example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    # prefix="You are a SQL expert. Given a user input, generate the appropriate SQL query.\nHere are some examples:",
    prefix="""You are an assitant for process engineers. You are an agent designed to interact with a SQL database or use your tools to return the current date or a test response. 
    Given an input question about data, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer. 
    You can order the results by a relevant column to return the most interesting examples in the database. 
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.,
    You have access to tools for interacting with the database as well as returning the current date or a test response.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    When a user asks about a material or item, they are referring to a unique entity from the column 'Copy of Comp MatlGrp Desc' column in the 'Wastage_Data' table with only these values possible: ['Tea Blends', 'ZWIP Default', 'Thermal Transfer Lbl', 'Corrugated & Display', 'Web', 'Misc Pkg Materials', 'ASSO BRAND DELTA MFG', '0', 'Cartons', 'Tea Tags', 'PS Labels', 'Poly Laminations', 'ZFIN DEFAULT', 'Plastic Bags']  
    When asked about 'downtime', 'reasons' or 'maintenance' query the 'Maintenance_Data' table.
    'Reasons' for downtime and maintenance are provided as Level 2 Reasons in the Maintenance_Data table in the column 'Level2Reason'.
    When asked about Lines or, for example, "L1", the lines you can query are only: ['L01 - C24', 'L02 - C24', 'L03 - C24', 'L03A - C24E', 'L04 - C21', 'L05  - C21', 'L19 - T2 Prima', 'L21 - Twinkle', 'L22 - Twinkle Rental', 'L23 - Twinkle 2', 'L24 - Twinkle 3', 'L35 - Fuso Combo 1', 'L36 - Fuso Combo 2']

    If the question does not seem related to the database, the current date or time, or a test_tool, just return "I don't know" as the answer. \nHere are some examples:""",
    suffix="User input: {input}\nSQL query: {agent_scratchpad}\n",
    input_variables=["input", "agent_scratchpad"]
)

# Step 3: Create a full prompt template with MessagesPlaceholder
full_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=few_shot_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Initialize the SQL agent
llm_sql = ChatOpenAI(model="gpt-4o", temperature=0.2)
sql_agent = create_sql_agent(
    llm_sql, 
    db=db, 
    prompt=full_prompt,  
    agent_type="openai-tools", 
    verbose=False
)

# Initialize the LLM for DataFrame agent
llm_df = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Example DataFrame (initially empty)
df = pd.DataFrame()

# Initialize the Pandas DataFrame agent
df_agent = create_pandas_dataframe_agent(
    llm_df,
    df,
    agent_type="zero-shot-react-description",
    verbose=True,
    return_intermediate_steps=False,
    allow_dangerous_code=True
)

# Define the plotting function
def plot_dataframe_with_agent(data: pd.DataFrame, x: str, y: str, kind: str = 'line'):
    # Update the DataFrame in the df_agent
    df_agent.df = data
    query = f"Plot the data with x={x} and y={y} as a {kind} plot."
    return df_agent.run(query)

# Tool for plotting
plot_tool = Tool(
    name="plot_dataframe_with_agent",
    func=plot_dataframe_with_agent,
    description="Plot the data with the DataFrame agent. Parameters: df (DataFrame), x (str), y (str), kind (str)"
)

# Create a function to coordinate the workflow
def coordinate_workflow(query: str, x: str, y: str, kind: str = 'line'):
    # Step 1: Use the SQL agent to get data
    data = sql_agent.run(query)
    # Step 2: Plot the data using the plotting tool
    plot_result = plot_tool.func(data, x, y, kind)
    return plot_result

# Create a coordinating tool
coordinating_tool = Tool(
    name="coordinate_workflow",
    func=coordinate_workflow,
    description="Coordinate the workflow between the SQL agent and DataFrame plotting agent."
)

# Initialize the LLM for the coordinating agent
llm_coordinator = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Create the LLM Chain for the coordinating agent
llm_chain = LLMChain(
    llm=llm_coordinator,
    prompt=PromptTemplate.from_template("Coordinate the following task:\n{input}")
)

# Create the ZeroShotAgent for coordination
coordinator_agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools=[coordinating_tool],
    verbose=True
)

# Create the agent executor
agent_executor = AgentExecutor(agent=coordinator_agent, tools=[coordinating_tool])

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything here!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        
        # Construct a dictionary with required inputs
        inputs = {"query": user_query, "x": "column_x", "y": "column_y", "kind": "line"}
        
        # Call the coordinating agent's run method with the inputs dictionary
        response = agent_executor.run(callbacks=[st_cb], **inputs)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
