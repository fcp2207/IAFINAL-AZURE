import chainlit as cl
# import requests
import httpx 
# from backend.app.ollama_client import OllamaClient
# from backend.app.image_utils import image_to_base64

API_URL = "http://ia-backend.ashybeach-87c13459.eastus.azurecontainerapps.io/predict"  # URL del servidor FastAPI


# @cl.on_chat_start
# async def on_chat_start():
#     await cl.Message(
#         content="Welcome to ollama client!"
#     ).send()

#     availabel_models = OllamaClient.list_models()
#     response = await cl.AskActionMessage(
#         content="Which model would you like to use?",
#         actions=[cl.Action(
#             name=model, 
#             payload={"value": model}, 
#             label=model) for model in availabel_models]
            
#     ).send()
    
#     if response is None:
#         return
#     payload = response.get("payload")
#     model_selected = payload.get("value")
#     system_instruction = (
#         # "You are a helpful assistant capable of providing insightful, relevant, and accurate responses based on the input you receive. You will use your reasoning and knowledge to assist the user, offering guidance, explanations, and solutions as needed, without requiring additional prompts or instructions. Your goal is to support the user in a natural, conversational manner."
#         "Eres un experto en motores electricos con 25 años de experiencia"    
#     )
#     ollama_client = OllamaClient(
#         model_name=model_selected, 
#         tools={
#             # "add_two_numbers": add_two_numbers,
#             # "subtract_two_numbers": subtract_two_numbers,
#             # "get_weather": get_weather,
#             # "make_appointment": make_appointment,
#             # "generate_image": generate_image
#         },
#         system_instruction=system_instruction
#     )
#     cl.user_session.set("ollama_client", ollama_client)


@cl.on_message
async def on_message(message: cl.Message):
    # 🔹 Enviar mensaje de carga
    waiting_msg = cl.Message(content="⏳ Generando respuesta, por favor espera...")
    await waiting_msg.send()

    try:
        ### Enviar pregunta a FastAPI
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(API_URL, json={"input_text": message.content})

        # Procesar la respuesta
        if response.status_code == 200:
            data = response.json()
            answer = data.get("message", "Error en la respuesta.")
        else:
            answer = "❌ Error al conectar con el modelo."

        print(f"✅ Respuesta obtenida: {answer}")  # Depuración

        # 🔹 Eliminar el mensaje de carga antes de enviar la respuesta final
        await waiting_msg.remove()

        # 🔹 Enviar la respuesta final correctamente
        await cl.Message(content=answer).send()

    except Exception as e:
        print(f"⚠️ Error en la solicitud: {e}")
        await waiting_msg.remove()
        await cl.Message(content="❌ Ocurrió un error al procesar tu solicitud.").send()