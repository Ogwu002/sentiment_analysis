from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

SLACK_WEBHOOK_URL = 'your_slack_webhook_url'

@app.route('/slack', methods=['POST'])
def slack():
    data = request.json
    text = data['text']

    # Perform sentiment analysis (using the logistic regression model as an example)
    preprocessed_text = preprocess_text(text)
    prediction = logistic_regression_model.predict(preprocessed_text)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

    # Send result back to Slack
    response = {
        'response_type': 'in_channel',
        'text': f'Sentiment: {sentiment}'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

    data =

 request.json
  text = data['text']

    # Perform sentiment analysis (using the logistic regression model as an example)
  preprocessed_text = preprocess_text(text)
  prediction = logistic_regression_model.predict(preprocessed_text)
  sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

    # Send result back to Slack
  response = {
        'response_type': 'in_channel',
        'text': f'Sentiment: {sentiment}'
    }
return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)