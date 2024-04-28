from flask import Flask, json, request, json
import sentence_splitter
app = Flask(__name__)



@app.route("/evaluate", methods=["POST"])
def split_sentences():
    json_data = json.loads(request.data)
    sentencesplitter_instance = sentence_splitter.SentenceSplitter()
    response=sentencesplitter_instance.split_sentences(json_data['text'])

    result = {"sentences": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response





if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=False,)