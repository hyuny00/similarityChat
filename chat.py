import chatSetup
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Chatbot is up and running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question')
    answer = chatSetup.return_answer(question)
    response = {"answer": answer}
    return jsonify(response)


if __name__ == "__main__":
    print("chat start")

    app.run(debug=True)

