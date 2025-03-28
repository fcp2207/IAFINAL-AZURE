import chainlit as cl
import httpx 

API_URL = "https://ia-backend.ashybeach-87c13459.eastus.azurecontainerapps.io/predict"


@cl.on_message
async def on_message(message: cl.Message):
    # 🔹 Log para saber que se recibió un mensaje desde Chainlit
    print(f"📥 Mensaje recibido desde Chainlit: {message.content}")

    # 🔹 Enviar mensaje de carga
    waiting_msg = cl.Message(content="⏳ Generando respuesta, por favor espera...")
    await waiting_msg.send()

    try:
        # 🔹 Log para saber que se está enviando request al backend
        print(f"🚀 Enviando solicitud POST a {API_URL}")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(API_URL, json={"input_text": message.content})

        # 🔹 Log de respuesta
        print(f"📬 Código de respuesta: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            answer = data.get("message", "⚠️ Error en la respuesta del backend.")
        else:
            answer = "❌ Error al conectar con el modelo."

        print(f"✅ Respuesta obtenida: {answer}")

        await waiting_msg.remove()
        await cl.Message(content=answer).send()

    except Exception as e:
        print(f"⚠️ Error en la solicitud: {e}")
        await waiting_msg.remove()
        await cl.Message(content="❌ Ocurrió un error al procesar tu solicitud.").send()
