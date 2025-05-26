import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("Sentiment Analysis App")
st.write("This app analyzes text sentiment using a DistilBERT model")

user_input = st.text_area("Enter text to analyze:", "I love this API!")

if st.button("Analyze Sentiment"):
    if user_input:
        try:
            response = requests.post(
                API_URL,
                json={"text": user_input},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                
                st.subheader("Result")
                if sentiment == "positive":
                    st.success(f"✅ Positive sentiment")
                else:
                    st.error(f"❌ Negative sentiment")
                
                st.json(result)
            else:
                st.error(f"Error: {response.text}")
                
        except Exception as e:
            st.error(f"Failed to connect to API: {str(e)}")
    else:
        st.warning("Please enter some text")