import streamlit as st
import requests
import json

# Configure the page
st.set_page_config(
    page_title="Sanjeevani: Yours Friendliest Doctor",
    page_icon="🌿",
    layout="wide"
)

# Define the API endpoints
API_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_URL}/chat"
HEALTH_ENDPOINT = f"{API_URL}/health"

def check_api_health():
    """Check the health status of the API"""
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            st.success("Backend API is healthy and running! ✅")
        else:
            st.error("Backend API is not responding correctly! ❌")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend API. Make sure it's running! ❌")

def chat_with_bot(query):
    """Send a chat query to the API and get response"""
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"query": query}
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to the backend API. Make sure it's running!"

def main():
    # Add a title
    st.title("🌿 Sanjeevani: Your AI Dietician")
    st.markdown("---")

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    # Health Check Button
    with col1:
        if st.button("🏥 Check API Health", use_container_width=True):
            check_api_health()

    # Clear Chat Button
    with col2:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.past = []
            st.session_state.generated = []

    # Initialize chat history
    if 'past' not in st.session_state:
        st.session_state.past = []
    if 'generated' not in st.session_state:
        st.session_state.generated = []

    # Chat input using text_input
    user_input = st.text_input("Ask about health, diet, or wellness...")
    
    # Handle Send button
    if st.button("Send") and user_input:
        # Add user input to history
        st.session_state.past.append(user_input)
        
        # Get bot response
        response = chat_with_bot(user_input)
        
        # Add bot response to history
        st.session_state.generated.append(response)

    # Display chat history
    if st.session_state.generated:
        chat_container = st.container()
        with chat_container:
            for i in range(len(st.session_state.generated)):
                # User message
                st.markdown(f"**You:** {st.session_state.past[i]}")
                st.markdown("---")
                
                # Bot response
                st.markdown(f"**Assistant:** {st.session_state.generated[i]}")
                st.markdown("---")

    # Add footer
    st.markdown("---")
    st.markdown("Built with Streamlit & FastAPI 🚀")

if __name__ == "__main__":
    main()