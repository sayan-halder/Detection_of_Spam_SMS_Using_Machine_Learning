from flask import Flask, request, render_template
from googletrans import Translator
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__, template_folder='template')

lang = {'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Azerbaijani': 'az',
        'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca',
        'Cebuano': 'ceb', 'Chichewa': 'ny', 'Chinese (simplified)': 'zh-cn', 'Chinese (traditional)': 'zh-tw',
        'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en',
        'Esperanto': 'eo', 'Estonian': 'et', 'Filipino': 'tl', 'Finnish': 'fi', 'French': 'fr', 'Frisian': 'fy',
        'Galician': 'gl', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian creole': 'ht',
        'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'he', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu',
        'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja',
        'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko', 'Kurdish (kurmanji)': 'ku',
        'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Luxembourgish': 'lb',
        'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi',
        'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar (burmese)': 'my', 'Nepali': 'ne', 'Norwegian': 'no',
        'Odia': 'or', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa',
        'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm', 'Scots gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st',
        'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so',
        'Spanish': 'es', 'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tajik': 'tg', 'Tamil': 'ta',
        'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uyghur': 'ug', 'Uzbek': 'uz',
        'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'}


# Define the route for the web page
@app.route('/')
def index():
    # Render the HTML template for the web page
    return render_template('index.html')


# Define the Flask endpoint
@app.route('/spam_or_not_spam_msgs', methods=['GET'])
def get_spam_or_not_spam():
    # Language Translation
    def language_convert(msg, msg_lang):
        translator = Translator()
        try:
            lang_code = lang.get(msg_lang)
            if lang_code:
                text_to_translate = translator.translate(msg, src=lang_code, dest='en')
                # print(text_to_translate.text)
                return text_to_translate.text
            else:
                return "Unsupported Language"
        except Exception as e:
            print(f"Translation Error: {e}")
            return "Translation Error"

    # Prediction
    def predict_spam(sample_message):
        f1 = open('Classifier.pickle', 'rb')
        classifier = pickle.load(f1)
        f1.close()

        f2 = open('CountVectorizer.pickle', 'rb')
        cv = pickle.load(f2)
        f2.close()

        sample_message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_message)
        sample_message = sample_message.lower()
        sample_message_words = sample_message.split()
        sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
        ps = PorterStemmer()
        final_message = [ps.stem(word) for word in sample_message_words]
        # print(final_message)
        final_message = ' '.join(final_message)
        temp = cv.transform([final_message]).toarray()
        return classifier.predict(temp)

    msg_lang = request.args.get("selectedMenu")

    msg = request.args.get("smsContent")

    sms = language_convert(msg, str(msg_lang))

    if predict_spam(sms):
        templateData = {
            'result': 0
        }
    else:
        templateData = {
            'result': 1
        }

    return render_template('index.html', **templateData)


# Define the main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=4001)
