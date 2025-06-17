from agents import Agent, OpenAIChatCompletionsModel,Runner,set_tracing_disabled
import os 
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chainlit as cl
# -------------------------------------
load_dotenv()
set_tracing_disabled(disabled=True)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# ----------------------------------------------
memory=[]
# -------------------------------------------------
models = {
    "DeepSeek": "deepseek/deepseek-r1:free",
    "Gemini_2_Flash":"google/gemini-2.0-flash-exp:free",
    "Mistral":"mistralai/devstral-small:free",
    "Qwen":"qwen/qwen3-14b:free",
    "Meta_Llama":"meta-llama/llama-4-maverick:free",
}
# ------------------------------------------------------
@cl.on_chat_start
async def chat_bot():
   await cl.Message(content="hi👋").send()
   
   settings= await cl.ChatSettings(
       [
           cl.input_widget.Select(
               id= "Model",
               label="choose LLM model",
               values=list(models.keys()),
               initial_index=0
           )
           
           
       ]
   ).send()
   
   await setup_chat(settings)
# -------------------------------------------------------
@cl.on_settings_update
async def setup_chat(settings):
    model_name= settings["Model"]
    cl.user_session.set("model",models[model_name])
    await cl.Message(content=f"you have selected {model_name} AI Model.🧠").send()
#------------------------------------------------------------------------------------ 
@cl.on_message
async def my_talk(msg:cl.Message):
    user_input=msg.content
    memory.append({"role":"user", "content":user_input})
    selected_model =cl.user_session.get("model")
    
    client= AsyncOpenAI(
       api_key=OPENROUTER_API_KEY,
       base_url="https://openrouter.ai/api/v1"
)
    agent= Agent(
       name="multi-agent",
       instructions="you are a helpful assistent.",
       model=OpenAIChatCompletionsModel(model=selected_model, openai_client=client)
)
 
    result= Runner.run_sync(agent,memory)
    await cl.Message(content=result.final_output).send()
 
 
    





