import streamlit as st
st.set_page_config(page_title="ML Agent UI", layout="wide", initial_sidebar_state="expanded")

import time
import pandas as pd
import os
import joblib
from PIL import Image, ImageDraw
from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from typing import Dict, Any, TypedDict, Optional



from database.db import init_db
from functions.data_cleaning import remove_unwanted_columns, data_cleaning
from functions.eda import perform_eda
from functions.feature_engineering import feature_engineering
from functions.model_training import model_selection_train_and_evaluation, select_target_column
from functions.storage import store_summary_in_db
from workflows.rag import answer_queries_using_rag
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from config import settings


# Ensure database is initialized
if not init_db():
    st.error("Failed to initialize database")
    st.stop()

# Define State Schema with improved error handling and persistence
class WorkflowState(TypedDict):
    df: pd.DataFrame
    summaries: Dict[str, Any]
    target_column: str
    model: Any
    metrics: Dict[str, float]
    current_step: str
    file_name: str
    chat_histories: Dict[str, Any]
    current_file: str
    error: Optional[str]
    progress: float  # 0.0 to 1.0
    last_successful_step: str


# Initialize session state variables
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}  # Store chat histories by filename
if "current_file" not in st.session_state:
    st.session_state.current_file = None


# Sidebar: Configuration & Instructions
with st.sidebar:
    st.header("Configuration")
    selected_model = st.selectbox(
        "Select LLM Model",
        [
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
            "gemma2-9b-it",
            "mixtral-8x7b-32768"
        ]
    )
    st.info("Choose an LLM model for your ML workflow analysis.")
    
    st.write("---")
    st.header("Instructions")
    st.markdown("""
    **Workflow Steps:**
    1. Upload CSV  
    2. Column Removal  
    3. Data Cleaning  
    4. EDA  
    5. Target Selection  
    6. Feature Engineering  
    7. Model Training  
    8. Store Summaries  
    9. Downloads  
    10. Chatbot  
    """)
    
    if st.button("Reset Workflow"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.title("Machine Learning Workflow UI")

# Initialize LLM with the selected model
llm = ChatGroq(model=selected_model, api_key=settings.groq_api_key)

# Define Node Functions
def load_data(state: WorkflowState) -> WorkflowState:
    """Node 1: Load Data with improved validation"""
    try:
        if "df" not in st.session_state:
            raise ValueError("No data loaded")
        
        if st.session_state.df.empty:
            raise ValueError("Uploaded file is empty")
            
        state["df"] = st.session_state.df
        state["summaries"] = {}
        state["current_step"] = "load_data"
        state["progress"] = 0.1
        state["last_successful_step"] = "load_data"
        state["error"] = None
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error loading data: {e}")
    return state


def remove_columns(state: WorkflowState) -> WorkflowState:
    """Node 2: Remove Unwanted Columns"""
    if "df_after_removal" not in st.session_state:
        return state
    state["df"] = st.session_state.df_after_removal
    state["summaries"]["column_removal"] = st.session_state.remove_summary
    state["current_step"] = "remove_columns"
    return state

def clean_data(state: WorkflowState) -> WorkflowState:
    """Node 3: Clean Data"""
    if "df_cleaned" not in st.session_state:
        return state
    state["df"] = st.session_state.df_cleaned
    state["summaries"]["data_cleaning"] = st.session_state.cleaning_summary
    state["current_step"] = "clean_data"
    return state

def data_eda(state: WorkflowState) -> WorkflowState:
    """Node 4: Perform EDA"""
    if "eda_summary" not in st.session_state:
        return state
    state["summaries"]["eda"] = st.session_state.eda_summary
    state["current_step"] = "data_eda"
    return state

def select_target(state: WorkflowState) -> WorkflowState:
    """Node 5: Select Target Column"""
    if "target_selected" not in st.session_state:
        return state
    state["target_column"] = st.session_state.target_selected
    state["current_step"] = "select_target"
    return state

def engineer_features(state: WorkflowState) -> WorkflowState:
    """Node 6: Feature Engineering"""
    if "df_engineered" not in st.session_state:
        return state
    state["df"] = st.session_state.df_engineered
    state["summaries"]["feature_engineering"] = st.session_state.feature_summary
    state["current_step"] = "feature_engineering"
    return state

def train_model(state: WorkflowState) -> WorkflowState:
    """Node 7: Train Model"""
    if "best_model" not in st.session_state:
        return state
    state["model"] = st.session_state.best_model
    state["metrics"] = st.session_state.best_metrics
    state["summaries"]["model_training"] = st.session_state.summary
    state["current_step"] = "train_model"
    return state

def store_summaries(state: WorkflowState) -> WorkflowState:
    """Node 8: Store Results"""
    if "summary" not in st.session_state:
        return state
    store_summary_in_db(state["summaries"])
    state["current_step"] = "store_summaries"
    return state

# Build LangGraph Workflow with visualization and conditional branching
workflow = StateGraph(WorkflowState)

# Add visualization capability
def visualize_workflow():
    graph = workflow.get_graph()
    st.write("### Workflow Visualization")
    st.graphviz_chart(graph.draw_mermaid_png())


# Add Nodes
workflow.add_node("load_data", RunnableLambda(load_data))
workflow.add_node("remove_columns", RunnableLambda(remove_columns))
workflow.add_node("clean_data", RunnableLambda(clean_data))
workflow.add_node("data_eda", RunnableLambda(data_eda))
workflow.add_node("select_target", RunnableLambda(select_target))
workflow.add_node("engineer_features", RunnableLambda(engineer_features))
workflow.add_node("train_model", RunnableLambda(train_model))
workflow.add_node("store_summaries", RunnableLambda(store_summaries))

# Set Edges (Workflow Flow)
workflow.set_entry_point("load_data")
workflow.add_edge("load_data", "remove_columns")
workflow.add_edge("remove_columns", "clean_data")
workflow.add_edge("clean_data", "data_eda")
workflow.add_edge("data_eda", "select_target")
workflow.add_edge("select_target", "engineer_features")
workflow.add_edge("engineer_features", "train_model")
workflow.add_edge("train_model", "store_summaries")

# Compile the workflow with enhanced logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_step(state: WorkflowState):
    logger.info(f"Executing step: {state['current_step']}")
    logger.info(f"Progress: {state['progress']*100:.1f}%")
    if state.get("error"):
        logger.error(f"Error in step {state['current_step']}: {state['error']}")
    return state

# Add logging node
workflow.add_node("log_step", RunnableLambda(log_step))

# Compile the workflow
app = workflow.compile()




# ---------------------- STEP 1: File Upload ----------------------
with st.expander("Step 1: Upload CSV File", expanded=True):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head(), height=200)
        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name  # Store file name
        st.session_state.current_file = uploaded_file.name
        if uploaded_file.name not in st.session_state.chat_histories:
            st.session_state.chat_histories[uploaded_file.name] = []


# ---------------------- STEP 2: Column Removal ----------------------
if "df" in st.session_state:
    with st.expander("Step 2: Column Removal", expanded=True):
        if "df_after_removal" not in st.session_state:  # Check if already processed
            with st.spinner("Analyzing columns for removal..."):
                try:
                    df_after_removal, remove_summary, removed_columns = remove_unwanted_columns(st.session_state.df, llm)
                    # Handle case where AI suggests removing no columns
                    removed_columns = removed_columns if removed_columns is not None else []
                    st.session_state.df_after_removal = df_after_removal
                    st.session_state.remove_summary = remove_summary
                    st.session_state.removed_columns = removed_columns
                    st.session_state.columns_removed = False  # Set flag to require user confirmation
                except Exception as e:
                    st.error(f"Error during column analysis: {e}")
                    st.stop()

        # Show suggested removal
        if st.session_state.removed_columns:
            st.markdown(f"**Suggested columns to remove:** {', '.join(st.session_state.removed_columns)}")
        else:
            st.markdown("**No columns suggested for removal**")

        st.markdown(f"**Reason:** {st.session_state.remove_summary}")
        
        # Show manual selection option
        all_columns = list(st.session_state.df.columns)
        manual_columns_to_remove = st.multiselect(
            "Or manually select columns to remove",
            options=all_columns,
            help="Select additional columns you want to remove"
        )
        
        # Combine suggested and manual selections
        final_columns_to_remove = list(set(st.session_state.removed_columns + manual_columns_to_remove))
        
        if final_columns_to_remove:
            st.markdown(f"**Final columns to be removed:** {', '.join(final_columns_to_remove)}")
        
        # Show current dataframe
        st.dataframe(st.session_state.df.head(), height=200)
        
        if st.button("Remove Columns"):
            try:
                # If no columns selected, use the original dataframe
                if not final_columns_to_remove:
                    st.session_state.df_after_removal = st.session_state.df
                    st.session_state.removed_columns = []
                    st.info("No columns were removed - proceeding with original dataset")
                else:
                    # Remove selected columns
                    df_after_removal = st.session_state.df.drop(columns=final_columns_to_remove)
                    st.session_state.df_after_removal = df_after_removal
                    st.session_state.removed_columns = final_columns_to_remove
                    st.success(f"Successfully removed columns: {', '.join(final_columns_to_remove)}")
                
                st.session_state.columns_removed = True
                # Show updated dataframe
                st.dataframe(st.session_state.df_after_removal.head(), height=200)
                # Automatically expand the next step
                st.session_state.expand_data_cleaning = True
            except Exception as e:
                st.error(f"Error processing columns: {e}")






# Initialize session state for expanding next step
if "expand_data_cleaning" not in st.session_state:
    st.session_state.expand_data_cleaning = False

# ---------------------- STEP 3: Data Cleaning ----------------------
if "df_after_removal" in st.session_state and st.session_state.get("columns_removed", False):
    with st.expander("Step 3: Data Cleaning", expanded=st.session_state.expand_data_cleaning):

        try:
            with st.spinner("Cleaning data..."):
                df_cleaned, cleaning_summary = data_cleaning(st.session_state.df_after_removal, llm)
            st.dataframe(df_cleaned.head(), height=200)
            st.session_state.df_cleaned = df_cleaned
            st.session_state.cleaning_summary = cleaning_summary
        except Exception as e:
            st.error(f"Error during data cleaning: {e}")

# ---------------------- STEP 4: Exploratory Data Analysis (EDA) ----------------------
if "df_cleaned" in st.session_state:
    with st.expander("Step 4: Exploratory Data Analysis (EDA)", expanded=True):
        try:
            if "eda_summary" not in st.session_state:
                with st.spinner("Performing EDA..."):
                    eda_results = perform_eda(st.session_state.df_cleaned, llm)
                st.session_state.eda_summary = eda_results['summary']
                st.session_state.eda_results = eda_results
            else:
                st.info("EDA already performed - showing previous results")
                eda_results = st.session_state.eda_results

            
            # Display main visualization
            st.write("Debug: Main visualization path:", eda_results['main_visualization'])
            if eda_results['main_visualization']:
                if os.path.exists(eda_results['main_visualization']):
                    try:
                        image = Image.open(eda_results['main_visualization'])
                        st.image(image, use_container_width=True, caption="Main EDA Visualization")

                    except Exception as e:
                        st.error(f"Error opening EDA image: {e}")
                else:
                    st.error(f"Main visualization file not found at: {eda_results['main_visualization']}")

            
            # Display all visualizations in a grid
            st.subheader("All Visualizations")
            if eda_results['visualizations']:
                # Create columns based on number of visualizations
                num_cols = 2
                cols = st.columns(num_cols)
                
                for i, vis_path in enumerate(eda_results['visualizations']):
                    try:
                        if os.path.exists(vis_path):
                            with cols[i % num_cols]:
                                image = Image.open(vis_path)
                                st.image(image, use_container_width=True, caption=f"Visualization {i+1}")
                        else:
                            st.error(f"Visualization file not found at: {vis_path}")
                    except Exception as e:
                        st.error(f"Error displaying visualization {i+1}: {str(e)}")


            
            # Display EDA summary
            st.subheader("Detailed EDA Summary")
            st.markdown("### Data Overview")
            st.write(st.session_state.df_cleaned.describe())
            
            st.markdown("### Missing Values")
            st.write(st.session_state.df_cleaned.isnull().sum())
            
            st.markdown("### LLM Analysis")
            st.write(eda_results['summary'])
            
            # Add button to move to target selection
            if st.button("Proceed to Target Selection"):
                st.session_state.expand_target_selection = True


        except Exception as e:
            st.error(f"Error during EDA: {e}")


# Initialize session state for expanding target selection
if "expand_target_selection" not in st.session_state:
    st.session_state.expand_target_selection = False

# ---------------------- STEP 5: Target Column Selection ----------------------
if "df_cleaned" in st.session_state and st.session_state.expand_target_selection:
    with st.expander("Step 5: Target Column Selection", expanded=True):


        try:
            df_cleaned = st.session_state.df_cleaned
            suggested_target = select_target_column(df_cleaned, llm)
            st.info(f"Suggested target column: **{suggested_target}**")
        except Exception as e:
            df_cleaned = st.session_state.df_cleaned
            suggested_target = df_cleaned.columns[0]
        columns_list = list(df_cleaned.columns)
        try:
            default_index = columns_list.index(suggested_target)
        except ValueError:
            default_index = 0
        target_column = st.selectbox("Select Target Column", options=columns_list, index=default_index)
        if st.button("Accept Target Column"):
            if target_column in df_cleaned.columns and not df_cleaned[target_column].isnull().all() and df_cleaned[target_column].nunique() > 1:
                st.session_state.target_selected = target_column
                st.success(f"Target column '{target_column}' accepted.")
            else:
                st.error("Invalid target column selection.")

# ---------------------- STEP 6: Feature Engineering ----------------------
if "target_selected" in st.session_state:
    with st.expander("Step 6: Feature Engineering", expanded=True):
        try:
            with st.spinner("Engineering features..."):
                df_engineered, feature_summary = feature_engineering(
                    st.session_state.df_cleaned, llm, st.session_state.target_selected
                )
            st.dataframe(df_engineered.head(), height=200)
            st.session_state.df_engineered = df_engineered
            st.session_state.feature_summary = feature_summary
        except Exception as e:
            st.error(f"Error during Feature Engineering: {e}")

# ---------------------- STEP 7: Model Training & Evaluation ----------------------
if "df_engineered" in st.session_state:
    with st.expander("Step 7: Model Training & Evaluation", expanded=True):
        # Check if training has already been completed
        if "training_completed" not in st.session_state:
            st.session_state.training_completed = False
            
        if not st.session_state.training_completed:
            try:
                with st.spinner("Training model..."):
                    best_model, _, X, y, summary, best_metrics = model_selection_train_and_evaluation(
                        st.session_state.df_engineered, st.session_state.target_selected, llm
                    )
                # Store results in session state
                st.session_state.best_model = best_model
                st.session_state.summary = summary
                st.session_state.best_metrics = best_metrics
                st.session_state.training_completed = True
            except Exception as e:
                st.error(f"Error during Model Training: {e}")
        
        # Display results if training is complete
        if st.session_state.training_completed:
            st.markdown("#### Model Training Summary")
            st.json(st.session_state.best_metrics)
            
            if st.button("Retrain Model"):
                # Reset training state to allow retraining
                st.session_state.training_completed = False
                st.rerun()


# ---------------------- STEP 8: Store Summaries in DB ----------------------
if all(key in st.session_state for key in ["summary", "remove_summary", "cleaning_summary", "eda_summary", "feature_summary", "best_metrics", "file_name"]):
    with st.expander("Step 8: Store Summaries in DB", expanded=True):
        try:
            with st.spinner("Storing summaries in the database..."):
                store_summary_in_db(
                    summary=st.session_state.summary,
                    remove_summary=st.session_state.remove_summary,
                    cleaning_summary=st.session_state.cleaning_summary,
                    eda_summary=st.session_state.eda_summary,
                    feature_summary=st.session_state.feature_summary,
                    best_metrics=st.session_state.best_metrics,
                    file_name=st.session_state.file_name
                )
            st.success("Summaries stored successfully!")
        except Exception as e:
            st.error(f"Error storing summaries: {e}")

# ---------------------- STEP 9: Downloads ----------------------
if "df_cleaned" in st.session_state and "best_model" in st.session_state:
    with st.expander("Step 9: Downloads", expanded=True):
        try:
            cleaned_data_path = os.path.join(os.getcwd(), "files_and_models", "processed_data", "cleaned_data.csv")
            model_path = os.path.join(os.getcwd(), "files_and_models", "saved_models", "best_model.pkl")
            st.session_state.df_cleaned.to_csv(cleaned_data_path, index=False)
            joblib.dump(st.session_state.best_model, model_path)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Cleaned Data",
                    data=open(cleaned_data_path, "rb").read(),
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "Download Best Model",
                    data=open(model_path, "rb").read(),
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.error(f"Error during downloads: {e}")

# Ensure session state variables are initialized
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "current_file" not in st.session_state or st.session_state.current_file is None:
    st.session_state.current_file = "default_chat"

# Ensure a chat history exists for the current file
if st.session_state.current_file not in st.session_state.chat_histories:
    st.session_state.chat_histories[st.session_state.current_file] = []

# ---------------------- STEP 10: Chatbot ----------------------
with st.expander("Step 10: Chatbot", expanded=True):
    st.markdown("### Chat with ML Assistant")
    st.markdown("Ask questions about your data and ML workflow. For example, ask about data preprocessing, feature engineering, model evaluation, or overall workflow insights.")

    chat_container = st.container(height=400)

    # Display existing chat history
    if st.session_state.chat_histories[st.session_state.current_file]:
        with chat_container:
            for message in st.session_state.chat_histories[st.session_state.current_file]:
                with st.chat_message(message["role"]):
                    role_display = "You" if message["role"] == "user" else "Assistant"
                    st.markdown(f"**{role_display} ({message['timestamp']}):** {message['content']}")
                st.divider()
    else:
        chat_container.info("No chat history yet. Ask a question below!")

    # User input
    user_prompt = st.chat_input("Ask me anything about the ML workflow")
    if user_prompt:
        timestamp = time.strftime("%H:%M:%S")
        # Append the user's message to the chat history
        st.session_state.chat_histories[st.session_state.current_file].append({
            "role": "user",
            "content": user_prompt,
            "timestamp": timestamp
        })

        with chat_container:
            with st.chat_message("user"):
                st.markdown(f"**You ({timestamp}):** {user_prompt}")
            st.divider()

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

            try:
                # Enhanced RAG Context Retrieval
                with st.spinner("Analyzing query and retrieving relevant context..."):
                    try:
                        # Get detailed context using RAG
                        rag_response = answer_queries_using_rag([user_prompt], llm, settings.postgresql_url)
                        retrieved_context = rag_response.get(user_prompt, "No relevant context found")
                        st.session_state.last_retrieved_context = retrieved_context
                        
                        # Construct enhanced chat prompt
                        messages = [
                            SystemMessage(
                                content="""You are an AI assistant specialized in machine learning workflows. 
                                Provide direct, concise answers using only the available information.
                                Instructions:
                                1. Answer the query directly using only the provided context
                                2. Use bullet points for structured responses
                                3. Keep responses under 50 words unless absolutely necessary
                                4. Do not include code examples or implementation details
                                5. Do not provide generic information, analysis, or suggestions
                                6. If no specific feature engineering steps are mentioned, state "No feature engineering details available"
                                7. For feature engineering queries:
                                   - List only the specific steps performed in this workflow
                                   - Include only what is explicitly documented in the context
                                   - Do not add any general information about feature engineering
                                   - If steps are listed, present them exactly as documented
                                8. Focus on providing precise, factual information from the workflow"""
                            ),
                            HumanMessage(
                                content=f"User Query: {user_prompt}\n\nRetrieved Context:\n{retrieved_context}"
                            )
                        ]

                        
                        # Log successful context retrieval
                        st.session_state.rag_status = "success"
                    except Exception as e:
                        st.error(f"Error retrieving context: {e}")
                        st.session_state.rag_status = "error"
                        messages = [
                            SystemMessage(
                                content="You are an AI assistant specialized in machine learning workflows."
                            ),
                            HumanMessage(
                                content=f"User Query: {user_prompt}\n\nNote: Context retrieval failed. Provide general guidance based on your training."
                            )
                        ]


                # Generate the assistant's response
                assistant_response = llm.invoke(messages)  # Returns an AIMessage object
                full_response = assistant_response.content  # Extract text content

                # Stream the response smoothly
                for chunk in full_response.split():
                    message_placeholder.markdown(f"**Assistant:** {chunk}▌")
                    time.sleep(0.05)
                message_placeholder.markdown(f"**Assistant ({timestamp}):** {full_response}")

                # Append the assistant's response to the chat history
                st.session_state.chat_histories[st.session_state.current_file].append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": timestamp
                })

            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                st.session_state.chat_histories[st.session_state.current_file].append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": timestamp
                })
