import gradio as gr
from generation import generate


def generate_code_no_history(prompt: str, history: list[list[str]]):
    print("History", history)
    print("Prompt", prompt)
    return generate(prompt)


def generate_code_with_history(prompt: str, history: list[list[str]]):
    return generate(prompt, history)


css = """
  .tabitem>div>div:first-child {
      height: 500px !important;
  }
"""

with gr.Blocks(css=css, title="Codestral Mamba") as demo:
  gr.HTML("<h1><center>Codestral Mamba by MistralAI<h1><center>")
  with gr.Tab("Codestral Mamba with history"):
    gr.ChatInterface(
        fn=generate_code_with_history,
        examples=[
        ],
        clear_btn=None,
        retry_btn=None,
        undo_btn=None,
        submit_btn="Send",
    )

  with gr.Tab("Codestral Mamba no history"):
    gr.ChatInterface(
        fn=generate_code_no_history,
        examples=[
        ],
        clear_btn=None,
        retry_btn=None,
        undo_btn=None,
        submit_btn="Send",
    )
demo.queue().launch(debug=True)
