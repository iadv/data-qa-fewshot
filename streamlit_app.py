import streamlit as st
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
import os
import sqlite3
from langchain.tools import Tool
from datetime import datetime

# Set up environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# ------------------- Create Sidebar Chat ----------------------

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

    1. 📄 Upload the relevant files in CSV format (e.g., maintenance data, customer data, sales data, etc.)
    2. Wait for the files to upload.
    3. Ask Questions! 📊

    ''')
    st.markdown("")
    st.markdown("")
    st.markdown("")


st.title('🤖 Data to Insights')
st.markdown("#### Unlock Actionable Insights from Your Process Data")
st.markdown("")
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    uploaded_file_A = st.file_uploader("**Upload File (e.g., wastage data) (.csv)**", type=("csv"))

with col2:
    uploaded_file_B = st.file_uploader("**Upload File (e.g., maintenance data) (.csv)**", type=("csv"))
    
    # Connect to the SQLite database
    conn = sqlite3.connect('Data.db')
    
if uploaded_file_A:
    # Read the uploaded file
    dfe = pd.read_csv(uploaded_file_A)
    
    
    # Save dataframes to SQL tables
    dfe.to_sql('Data_A', conn, index=False, if_exists='replace')
    st.success("File A successfully uploaded and data ready for analysis.")

if uploaded_file_B:
    # Read the uploaded file
    dfe = pd.read_csv(uploaded_file_B)

    dfe.to_sql('Data_B', conn, index=False, if_exists='replace')
    st.success("File B successfully uploaded and data ready for analysis.")
    
    # Commit and close the connection
    conn.commit()
    conn.close()
    
    st.success("Files have been successfully uploaded.")


st.markdown("")
st.markdown("")
st.markdown("")


# Step 1: Define example queries
examples = [
    # Wastage and Maintenance Data Queries
    {
        "input": "What is the amount of wastage for tea blends?",
        "query": """SELECT "Copy of Comp MatlGrp Desc" AS ComponentMaterialGroup, SUM("Var2Prf Amt") AS TotalWastage 
                    FROM Wastage_Data 
                    WHERE "Copy of Comp MatlGrp Desc" = 'Tea Blends' 
                    GROUP BY "Copy of Comp MatlGrp Desc";"""
    },
    {
        "input": "What's the reasons for plastic bags wastage on L1?",
        "query": """SELECT Level2Reason, COUNT(*) AS ReasonCount 
                    FROM Maintenance_Data 
                    WHERE Line = 'L01 - C24' 
                    GROUP BY Level2Reason;"""
    },
    {
        "input": "What was the top contributor to wastage this month?",
        "query": """SELECT "Copy of Comp MatlGrp Desc" AS ComponentMaterialGroup, SUM("Var2Prf Amt") AS TotalWastage 
                    FROM Wastage_Data 
                    GROUP BY "Copy of Comp MatlGrp Desc"
                    ORDER BY TotalWastage DESC
                    LIMIT 1;"""
    },
    # General Inquiries about File Contents
    {
        "input": "What's in these files?",
        "query": """SELECT 'Overview' AS Summary,
                           COUNT(*) AS TotalRecords,
                           GROUP_CONCAT(DISTINCT COLUMN_NAME) AS Columns
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = 'YourTableName';"""
    },
    {
        "input": "Can you tell me what's in the dataset?",
        "query": """SELECT 'Dataset Overview' AS Summary,
                           COUNT(DISTINCT <PrimaryKeyColumn>) AS UniqueEntries,
                           SUM(<NumericColumn>) AS TotalValue
                    FROM YourTableName;"""
    },
    {
        "input": "What kind of data is in these tables?",
        "query": """SELECT 'Table Overview' AS Summary,
                           COUNT(DISTINCT <IdentifierColumn>) AS UniqueItems,
                           AVG(<NumericColumn>) AS AverageValue,
                           COUNT(DISTINCT <CategoryColumn>) AS UniqueCategories
                    FROM YourTableName;"""
    },
    {
        "input": "What do these files contain?",
        "query": """SELECT 'File Content Overview' AS Summary,
                           COUNT(DISTINCT <EntityColumn>) AS UniqueEntities,
                           MAX(<NumericColumn>) AS MaxValue,
                           MIN(<NumericColumn>) AS MinValue
                    FROM YourTableName;"""
    },
    # Specific Datasets Queries
    {
        "input": "What's in the Customer_Data file?",
        "query": """SELECT 'Customer Data Overview' AS Overview,
                           COUNT(DISTINCT CustomerID) AS UniqueCustomers,
                           SUM(PurchaseAmount) AS TotalPurchases,
                           COUNT(DISTINCT ProductID) AS ProductsPurchased
                    FROM Customer_Data;"""
    },
    {
        "input": "What's in the Sales_Data file?",
        "query": """SELECT 'Sales Data Overview' AS Overview,
                           COUNT(DISTINCT OrderID) AS UniqueOrders,
                           SUM(SalesAmount) AS TotalSalesRevenue,
                           COUNT(DISTINCT Product) AS UniqueProductsSold
                    FROM Sales_Data;"""
    },
    {
        "input": "What's in the Stock_Data file?",
        "query": """SELECT 'Stock Data Overview' AS Overview,
                           COUNT(DISTINCT StockSymbol) AS UniqueStocks,
                           AVG(Price) AS AverageStockPrice,
                           SUM(Volume) AS TotalTradingVolume
                    FROM Stock_Data;"""
    }
]

# Step 2: Create a FewShotPromptTemplate
example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""You are a versatile assistant designed to interact with SQL databases across multiple domains, including but not limited to wastage and maintenance data. You can also analyze customer data, sales, and stock market information. Your task is to interpret the context based on column names and generate appropriate SQL queries.

    Given an input question about data, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer. 
    You can order the results by a relevant column to return the most interesting examples in the database. 
    Never query for all the columns from a specific table; only ask for the relevant columns given the question.

    You have access to tools for interacting with the database as well as returning the current date or a test response.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

    **For Wastage and Maintenance Data:**
    When a user asks about a material or item, they are referring to a unique entity from the 'Copy of Comp MatlGrp Desc' column in the 'Wastage_Data' table with only these values possible: ['Tea Blends', 'ZWIP Default', 'Thermal Transfer Lbl', 'Corrugated & Display', 'Web', 'Misc Pkg Materials', 'ASSO BRAND DELTA MFG', '0', 'Cartons', 'Tea Tags', 'PS Labels', 'Poly Laminations', 'ZFIN DEFAULT', 'Plastic Bags'].  
    When asked about 'downtime', 'reasons', or 'maintenance', query the 'Maintenance_Data' table.
    'Reasons' for downtime and maintenance are provided as Level 2 Reasons in the Maintenance_Data table in the column 'Level2Reason'.
    When asked about Lines or, for example, "L1", the lines you can query are only: ['L01 - C24', 'L02 - C24', 'L03 - C24', 'L03A - C24E', 'L04 - C21', 'L05  - C21', 'L19 - T2 Prima', 'L21 - Twinkle', 'L22 - Twinkle Rental', 'L23 - Twinkle 2', 'L24 - Twinkle 3', 'L35 - Fuso Combo 1', 'L36 - Fuso Combo 2'].

    **For General Data Analysis:**
    - Customer Data: Look for columns such as 'CustomerID', 'Name', 'PurchaseHistory', 'ContactInfo', etc., and aggregate or filter based on common customer queries like total purchases or frequent purchases.
    - Sales Data: Identify columns such as 'SalesAmount', 'Date', 'Product', 'Region', etc., and perform operations to find trends, top products, or sales over time.
    - Stock Market Data: Focus on columns like 'StockSymbol', 'Price', 'Volume', 'Date', etc., to analyze stock performance, average prices, or volume trends.

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

# Custom function to get the current date
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# Create a tool from the custom function
date_tool = Tool(
    name="get_current_date",
    func=get_current_date,
    description="Get the current date"
)

# Simple test function
def simple_test_tool():
    return "Test tool response for Ian"

# Create a tool from the simple test function
test_tool_Ian = Tool(
    name="simple_test_tool",
    func=simple_test_tool,
    description="Returns a test response"
)

# Initialize the LLM and create the SQL agent
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
db = SQLDatabase.from_uri("sqlite:///Data.db")
agent = create_sql_agent(llm, db=db, prompt=full_prompt, tools=[date_tool, test_tool_Ian], agent_type="openai-tools", verbose=False)


if "messages" not in st.session_state or st.sidebar.button("New Conversation"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi User, how can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        
        # Construct a dictionary with required inputs
        inputs = {"input": user_query, "agent_scratchpad": ""}
        
        # Call the agent's run method with the inputs dictionary
        response = agent.run(callbacks=[st_cb], **inputs)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)



#copy of prompts: 07/31
# Step 1: Define example queries
#examples = [
 #   {"input": "What is the amount of wastage for tea blends?", "query": """SELECT "Copy of Comp MatlGrp Desc" AS ComponentMaterialGroup, SUM("Var2Prf Amt") AS TotalWastage FROM Wastage_Data WHERE "Copy of Comp MatlGrp Desc" = 'Tea Blends' GROUP BY "Copy of Comp MatlGrp Desc";"""},
  #  {"input": "What's the reasons for plastic bags wastage on L1?", "query": "SELECT Level2Reason, COUNT(*) AS ReasonCount \nFROM Maintenance_Data \nWHERE Line = 'L01 - C24' \nGROUP BY Level2Reason;"},
   # {"input": "What was the top contributor to wastage this month?", "query": """SELECT "Copy of Comp MatlGrp Desc" AS ComponentMaterialGroup, SUM("Var2Prf Amt") AS TotalWastage \nFROM Wastage_Data \nGROUP BY "Copy of Comp MatlGrp Desc";"""},
#]

# Step 2: Create a FewShotPromptTemplate
#example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

#few_shot_prompt = FewShotPromptTemplate(
 #   examples=examples,
  #  example_prompt=example_prompt,
   # # prefix="You are a SQL expert. Given a user input, generate the appropriate SQL query.\nHere are some examples:",
 #   prefix="""You are an assitant for process engineers. You are an agent designed to interact with a SQL database or use your tools to return the current date or a test response. 
 #   Given an input question about data, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer. 
 #   You can order the results by a relevant column to return the most interesting examples in the database. 
 #   Never query for all the columns from a specific table, only ask for the relevant columns given the question.,
 #   You have access to tools for interacting with the database as well as returning the current date or a test response.
 #   Only use the given tools. Only use the information returned by the tools to construct your final answer.
 #   You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

  #  DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

 #   When a user asks about a material or item, they are referring to a unique entity from the column 'Copy of Comp MatlGrp Desc' column in the 'Wastage_Data' table with only these values possible: ['Tea Blends', 'ZWIP Default', 'Thermal Transfer Lbl', 'Corrugated & Display', 'Web', 'Misc Pkg Materials', 'ASSO BRAND DELTA MFG', '0', 'Cartons', 'Tea Tags', 'PS Labels', 'Poly Laminations', 'ZFIN DEFAULT', 'Plastic Bags']  
 #   When asked about 'downtime', 'reasons' or 'maintenance' query the 'Maintenance_Data' table.
 #   'Reasons' for downtime and maintenance are provided as Level 2 Reasons in the Maintenance_Data table in the column 'Level2Reason'.
 #   When asked about Lines or, for example, "L1", the lines you can query are only: ['L01 - C24', 'L02 - C24', 'L03 - C24', 'L03A - C24E', 'L04 - C21', 'L05  - C21', 'L19 - T2 Prima', 'L21 - Twinkle', 'L22 - Twinkle Rental', 'L23 - Twinkle 2', 'L24 - Twinkle 3', 'L35 - Fuso Combo 1', 'L36 - Fuso Combo 2']

 #   If the question does not seem related to the database, the current date or time, or a test_tool, just return "I don't know" as the answer. \nHere are some examples:""",
 #   suffix="User input: {input}\nSQL query: {agent_scratchpad}\n",
 #   input_variables=["input", "agent_scratchpad"]
#)



# # def get_fewshot_agent_chain(): 
    
#     # llm & db setup
# llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
#     # db = SQLDatabase.from_uri("sqlite:///Data.db")

#     # # create few shot prompts, their embeddings and store in Chromadb
#     # embeddings = OpenAIEmbeddings()
    
#     # # create example selector which chooses k= examples to include in the agent's prompt
#     # example_selector = SemanticSimilarityExampleSelector.from_examples(
#     #     few_shots_ag,
#     #     embeddings,
#     #     Chroma,
#     #     k=3,
#     #     input_keys=["input"],
#     # )


#     # # Now we can create our FewShotPromptTemplate, which takes our example selector, an example prompt for formatting each example, and a string prefix and suffix to put before and after our formatted examples:
#     # from langchain_core.prompts import (
#     #     ChatPromptTemplate,
#     #     FewShotPromptTemplate,
#     #     MessagesPlaceholder,
#     #     PromptTemplate,
#     #     SystemMessagePromptTemplate,
#     # )

#     system_prefix = """You are an agent designed to interact with a SQL database.
#     Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
#     Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
#     You can order the results by a relevant column to return the most interesting examples in the database.
#     Never query for all the columns from a specific table, only ask for the relevant columns given the question.
#     You have access to tools for interacting with the database.
#     Only use the given tools. Only use the information returned by the tools to construct your final answer.
#     You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

#     DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

#     If the question does not seem related to the database, just return "I don't know" as the answer.

# #     # Here are some examples of user inputs and their corresponding SQL queries:"""

#     # few_shot_prompt = FewShotPromptTemplate(
#     #     example_selector=example_selector,
#     #     example_prompt=PromptTemplate.from_template(
#     #         "User input: {input}\nSQL query: {query}"
#     #     ),
#     #     input_variables=["input", "dialect", "top_k"],
#     #     prefix=system_prefix,
#     #     suffix="",
#     # )

#     # # our full prompt should be a chat prompt with a human message template and an agent_scratchpad MessagesPlaceholder.
#     # full_prompt = ChatPromptTemplate.from_messages(
#     #     [
#     #         SystemMessagePromptTemplate(prompt=few_shot_prompt),
#     #         ("human", "{input}"),
#     #         MessagesPlaceholder("agent_scratchpad"),
#     #     ]
#     # )

#     # agent_executor = create_sql_agent(llm, db=db, prompt=full_prompt, agent_type="openai-tools", verbose=True)
#     # return agent_executor




