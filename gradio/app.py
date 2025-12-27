import gradio as gr
def greet(name):
    return "Hello " + name + "!!!!"
demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
if __name__ == "__main__":
    demo.launch(server_name='127.0.0.1', server_port=7861, share=True)
