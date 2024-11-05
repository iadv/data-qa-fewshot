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

# Ensure the database is empty at the start of the session
db_path = 'Data.db'
if os.path.exists(db_path):
    os.remove(db_path)

# Connect to the SQLite database (this will create a new, empty database)
conn = sqlite3.connect(db_path)

# ------------------- Create Sidebar Chat ----------------------

# Increase the default width of the main area by 50%
st.set_page_config(layout="wide")

# stSidebarContent
# stAppViewContainer
page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    # background-image: url("https://img.freepik.com/premium-vector/sky-blue-gradient-web-brochure-template-background-illustration-simple-plain-background_784842-323.jpg");
    background-color: #f8f8fa;
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
with st.sidebar:
    st.markdown("")
    st.subheader('**_How_ _to_ _Use:_**', divider="blue")
    st.write('''
    1. 📄 Upload relevant files in CSV format (maximum 10 files).
    
    ''')
    st.markdown("")
    st.markdown("")
    uploaded_files = st.file_uploader("Upload Files (e.g. production data, maintenance log, quality log, customer data, etc.). (.csv)", type=("csv"), accept_multiple_files=True)
    st.markdown("")
    st.markdown("")
    st.write('''
    2. Wait for the files to upload.
    
    3. Ask Questions! 📊

    ''')
    st.markdown("")
    st.subheader("_Example_ _Use_ _Case_", divider="blue")
    st.write('''
        Upload wastage and maintenance data files to do root cause analysis of your production waste
    ''')
    st.markdown("")
    st.markdown("")

# st.header("Start asking questions about your business data", divider="gray")
# First Row - Buttons (Using Columns)
col1, col2 = st.columns(2)  # Create two columns

with col1:
    st.markdown(
        """
        <button style="
            background-color: #d7ecff; /* Primary blue color */
            color: black;
            padding: 10px 15px;
            border: 2px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease;
            cursor: pointer;
            width: 100%; /* Make button take full width of column */
            margin: 0 5px;
        "><b>_Text2DataAnalysis_</b> Co-Pilot</button>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <a href="https://processdatav2.vercel.app/" target="_blank">
            <button style="
                background-color: #f6f5fd; /* Transparent background for outline */
                color: black; /* Primary blue color for text */
                padding: 10px 15px;
                border: 2px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: background-color 0.3s ease, color 0.3s ease; /* Hover effects */
                cursor: pointer;
                width: 100%; /* Make button take full width of column */
                margin: 0 5px;
            "> <b>Click here</b> for other AI Co-Pilots</button>
        </a>
        """,
        unsafe_allow_html=True,
    )

# Second Row - Header
st.markdown(
    """
    <h2>Start asking questions about your business data</h2>
    <hr style="border-top: 1px solid gray; margin-top: 10px;">
    """,
    unsafe_allow_html=True,
)

# Connect to the SQLite database
conn = sqlite3.connect('Data.db')

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        df = pd.read_csv(uploaded_file)
        df.to_sql(f'File {i+1}', conn, index=False, if_exists='replace')
        st.success(f"File {i+1} successfully uploaded and data ready for analysis.")

# Commit and close the connection
conn.commit()
conn.close()


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
    {
        "input": "What is the percentage of unplanned maintenance on line 1?",
        "query": """SELECT (SUM(CASE WHEN Reason = 'Unplanned Maintenance' THEN DowntimeInMinutes ELSE 0 END) * 100.0 / SUM(DowntimeInMinutes)) AS UnplannedMaintenancePercentage \nFROM "File 2" \nWHERE Line = 'L01 - C24';"""
    },
    {
        "input": "Which SKU has highest unplanned maintenance??",
        "query": """SELECT "Description" AS SKU, SUM("DowntimeInMinutes") AS TotalUnplannedMaintenance \nFROM "File 2" \nWHERE "Reason" = 'Unplanned Maintenance' \nGROUP BY "Description" \nORDER BY TotalUnplannedMaintenance DESC \nLIMIT 1;"""
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
    # prefix="You are a SQL expert. Given a user input, generate the appropriate SQL query.\nHere are some examples:",
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
    Whether downtime and stoppages are planned, unplanned or other types is given in the column 'Reason' and include only : ['Excluded Time', 'Unplanned Maintenance', 'Stoppage', 'Changeovers', 'Work Breaks', 'Sanitation', 'Other', 'Planned Maintenance'].
    'Reasons' for downtime and maintenance are provided as Level 2 Reasons in the Maintenance_Data table in the column 'Level2Reason'.
    When asked about Lines or, for example, "L1", the lines you can query are only: ['L01 - C24', 'L02 - C24', 'L03 - C24', 'L03A - C24E', 'L04 - C21', 'L05  - C21', 'L19 - T2 Prima', 'L21 - Twinkle', 'L22 - Twinkle Rental', 'L23 - Twinkle 2', 'L24 - Twinkle 3', 'L35 - Fuso Combo 1', 'L36 - Fuso Combo 2'].
    When asked about a SKU, they are provided in the column 'Description'
    
    **For General Data Analysis:**
    - Customer Data: Look for columns such as 'CustomerID', 'Name', 'PurchaseHistory', 'ContactInfo', etc., and aggregate or filter based on common customer queries like total purchases or frequent purchases.
    - Sales Data: Identify columns such as 'SalesAmount', 'Date', 'Product', 'Region', etc., and perform operations to find trends, top products, or sales over time.
    - Stock Market Data: Focus on columns like 'StockSymbol', 'Price', 'Volume', 'Date', etc., to analyze stock performance, average prices, or volume trends.

    If the question does not seem related to the database, the current date or time, or a test_tool, just explain the issue with the question in 1 sentence and ask them to try another question related to the data as the answer. If this is the case, also tell the user to upload the data if they haven't. \nHere are some examples:""",
    suffix="User input: {input}\nSQL query: {agent_scratchpad}\n",
    input_variables=["input", "agent_scratchpad"]
)

# Step 3: Create a full prompt template with MessagesPlaceholder
full_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=few_shot_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# Initialize the LLM and create the SQL agent
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
db = SQLDatabase.from_uri("sqlite:///Data.db")
# agent = create_sql_agent(llm, db=db, prompt=full_prompt, tools=[date_tool, test_tool_Ian], agent_type="openai-tools", verbose=True)
agent = create_sql_agent(llm, db=db, prompt=full_prompt, agent_type="openai-tools", verbose=True)


if "messages" not in st.session_state or st.sidebar.button("New Conversation"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, how can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything about your data!")

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

