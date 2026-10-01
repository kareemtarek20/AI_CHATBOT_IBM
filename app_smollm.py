import threading
import warnings
import webbrowser

import torch
from flask import Flask, jsonify, render_template_string, request
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Loading model (first run downloads ~800 MB)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
if torch.cuda.is_available():
    model.to("cuda")
print("Model ready.")

app = Flask(__name__)

conversations = {}

SYSTEM_PROMPT = (
    "You are a friendly, helpful chatbot. Keep your replies short and conversational."
)


def generate_reply(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": message})

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(input_text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][enc["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not reply:
        reply = "Hmm, I'm not sure how to respond. Try asking something else!"
    return reply


CHAT_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SmolLM2 Chatbot</title>
<style>
  body { font-family: Arial, sans-serif; background:#f2f4f8; margin:0;
         display:flex; flex-direction:column; height:100vh; }
  header { background:#7c3aed; color:white; padding:14px 20px; font-size:18px; }
  #chat { flex:1; overflow-y:auto; padding:20px; }
  .msg { max-width:70%; margin:8px 0; padding:10px 14px; border-radius:14px; line-height:1.4; }
  .user { background:#7c3aed; color:white; margin-left:auto; border-bottom-right-radius:4px; }
  .bot  { background:white; color:#222; border:1px solid #ddd; border-bottom-left-radius:4px; }
  #inputbar { display:flex; gap:10px; padding:14px 20px; background:#e8eaed; }
  #msg { flex:1; padding:12px; border-radius:10px; border:1px solid #ccc; font-size:15px; }
  button { padding:12px 22px; border:none; border-radius:10px; background:#7c3aed;
           color:white; font-size:15px; cursor:pointer; }
  button:disabled { background:#c4b0ec; }
</style>
</head>
<body>
<header>Simple Chatbot &mdash; HuggingFaceTB/SmolLM2-360M-Instruct</header>
<div id="chat"></div>
<div id="inputbar">
  <input id="msg" placeholder="Type a message..." autofocus
         onkeydown="if(event.key==='Enter')send()">
  <button id="sendbtn" onclick="send()">Send</button>
</div>
<script>
function add(text, cls){
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  const chat = document.getElementById('chat');
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}
async function send(){
  const input = document.getElementById('msg');
  const text = input.value.trim();
  if(!text) return;
  add(text, 'user');
  input.value = '';
  document.getElementById('sendbtn').disabled = true;
  add('...', 'bot');
  try{
    const r = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await r.json();
    document.getElementById('chat').lastChild.textContent = data.reply;
  }catch(e){
    document.getElementById('chat').lastChild.textContent = 'Error talking to server.';
  }
  document.getElementById('sendbtn').disabled = false;
  input.focus();
}
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(CHAT_HTML)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400
    session_id = data.get("session", "default")
    history = conversations.setdefault(session_id, [])
    reply = generate_reply(message, history)
    history.append({"user": message, "bot": reply})
    if len(history) > 4:
        history.pop(0)
    return jsonify({"reply": reply})


@app.route("/predict", methods=["POST"])
@app.route("/generate", methods=["POST"])
def predict():
    return chat()


if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
