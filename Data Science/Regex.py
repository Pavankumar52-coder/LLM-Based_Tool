import streamlit as st
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

# Load models and pipelines
general_model_name = "gpt2"
summarization_model_name = "facebook/bart-large-cnn"
qa_model_name = "deepset/roberta-base-squad2"

# General Text Generation
general_tokenizer = AutoTokenizer.from_pretrained(general_model_name)
general_model = AutoModelWithLMHead.from_pretrained(general_model_name)

# Summarization
summarizer = pipeline("summarization", model=summarization_model_name)

# Question Answering
qa_pipeline = pipeline("question-answering", model=qa_model_name)

# Functions
def generate_general_text(prompt):
    inputs = general_tokenizer(prompt, return_tensors="pt")
    outputs = general_model.generate(
        inputs["input_ids"],
        max_length=150,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )
    response = general_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def summarize_text(text):
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

def answer_question(context, question):
    response = qa_pipeline(question=question, context=context)
    return response['answer']

# Streamlit UI
def main():
    st.title("Enhanced LLM-Based Text Tool")
    st.subheader("Generate text, answer questions, or summarize content.")

    # Input
    task_type = st.radio("Select the task:", ["General Text Generation", "Q&A", "Summarization"])
    user_input = st.text_area("Enter your text:", "")

    # Generate Button
    if st.button("Generate"):
        if not user_input.strip():
            st.warning("Please provide some input.")
        else:
            with st.spinner("Processing..."):
                if task_type == "General Text Generation":
                    response = generate_general_text(user_input)
                elif task_type == "Summarization":
                    response = summarize_text(user_input)
                elif task_type == "Q&A":
                    question = st.text_input("Enter your question for the Q&A task:", "")
                    if question.strip():
                        response = answer_question(user_input, question)
                    else:
                        st.warning("Please enter a question for Q&A.")
                        return

                st.success("Generated Response:")
                st.text_area("Response:", response, height=200)

if __name__ == "__main__":
    main()