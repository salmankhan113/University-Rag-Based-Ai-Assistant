import streamlit as st
from chatbot import app, extract_text

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 College Assistant")
st.caption("Ask questions about academics, fees, or general topics.")

if "programme" not in st.session_state:
    st.session_state.programme = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

if st.session_state.programme is None:
    st.subheader("Which programme are you in?")

    programme = st.selectbox(
        "Select your programme",
        ["BCA", "BBA", "B.Com (H)"]
    )

    if st.button("Start Chat", type="primary"):
        st.session_state.programme = programme
        st.rerun()

else:
    st.success(
        f"You're set as a {st.session_state.programme} student."
    )

    if st.button("Change Programme"):
        st.session_state.programme = None
        st.session_state.messages = []
        st.session_state.thread_id = __import__("uuid").uuid4().hex
        st.rerun()

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input("Ask your question...")

    if user_query:
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    config = {
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }

                    result = app.invoke(
                        {
                            "programme": st.session_state.programme,
                            "messages": [
                                ("human", user_query)
                            ]
                        },
                        config=config
                    )

                    answer = extract_text(
                        result["messages"][-1].content
                    )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:
                    st.error(f"An error occurred: {error}")
